#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import re
from dataclasses import dataclass
from typing import Any

from common import settings
from common.string_utils import remove_redundant_spaces


@dataclass
class TableEntityFilterPlan:
    filters: list[dict[str, Any]]
    debug: dict[str, Any]


_COLUMN_ALIASES = {
    "设备编码": ("设备编码", "设备编号", "设备code", "设备 CODE", "device code"),
    "故障代码": ("故障代码", "故障编码", "报警代码", "fault code"),
    "故障描述": ("故障描述", "故障现象", "报警描述", "异常描述", "fault description"),
    "产线": ("产线", "线体", "生产线", "line"),
    "步骤序号": ("步骤序号", "步骤编号", "步骤", "step"),
}


def table_entity_filter_enabled(config: dict | None = None, request_payload: dict | None = None,
                                kbs: list[Any] | None = None) -> bool:
    # Explicit flag in request or config takes priority
    for source in (request_payload or {}, config or {}):
        if "enable_table_entity_filter" in source:
            return _coerce_bool(source.get("enable_table_entity_filter"))
        table_cfg = source.get("table_entity_filter")
        if isinstance(table_cfg, dict) and "enabled" in table_cfg:
            return _coerce_bool(table_cfg.get("enabled"))
    # Auto-enable for table KBs so frontend chats benefit without manual config
    if _has_table_kb(kbs or []):
        return True
    return False


def build_table_entity_filter(
    question: str,
    kbs: list[Any],
    field_map: dict[str, str] | None,
    config: dict | None = None,
    request_payload: dict | None = None,
) -> TableEntityFilterPlan | None:
    debug: dict[str, Any] = {"enabled": False}
    if not table_entity_filter_enabled(config, request_payload, kbs=kbs):
        debug["reason"] = "disabled"
        return None
    debug["enabled"] = True

    if settings.DOC_ENGINE_INFINITY or settings.DOC_ENGINE_OCEANBASE:
        debug["reason"] = "unsupported_doc_engine"
        return TableEntityFilterPlan([], debug)

    if not field_map:
        debug["reason"] = "missing_field_map"
        return TableEntityFilterPlan([], debug)

    entities = _extract_entities(question)
    debug["entities"] = entities
    if not entities:
        debug["reason"] = "no_entities"
        return TableEntityFilterPlan([], debug)

    filters, mapped = _entities_to_filters(entities, field_map)
    debug["mapped_fields"] = mapped
    if not filters:
        debug["reason"] = "no_mapped_filters"
        return TableEntityFilterPlan([], debug)

    plans = _build_fallback_filters(filters)
    debug["attempt_count"] = len(plans)
    debug["reason"] = "ok"
    return TableEntityFilterPlan(plans, debug)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _has_table_kb(kbs: list[Any]) -> bool:
    for kb in kbs or []:
        parser_id = getattr(kb, "parser_id", None)
        if isinstance(kb, dict):
            parser_id = kb.get("parser_id", parser_id)
        if str(parser_id or "").lower() == "table":
            return True
    return False


def _strip_value(value: str) -> str:
    value = remove_redundant_spaces(value or "")
    value = value.strip(' \t\r\n,，。:：;；"\'""''')
    # strip trailing noise words that don't belong to entity value
    for noise in ("时", "故障", "怎么处理", "如何", "处理", "维修", "的", "上", "出现"):
        if value.endswith(noise) and len(value) > len(noise) + 1:
            value = value[:-len(noise)]
    value = value.strip(' \t\r\n,，。:：;；"\'""''')
    return value


# Leading fault-code prefix that the 出现“...”时 故障描述 pattern swallows whole,
# e.g. “NQ_WC120_036，12213 NG” -> 12213 NG. The pattern matches ASCII ["'] only,
# so smart quotes wrapping the value are also trimmed here.
_FAULT_CODE_PREFIX = re.compile(
    '^[“”‘’"\']*\\s*[A-Z]{2,3}_WC[A-Z0-9]+_[A-Z0-9]+\\s*[，,；;:：]\\s*'
)
_VALUE_QUOTES = "“”‘’\"' "


# Canonical fault-code patterns. WC segments are alphanumeric, not pure digits,
# so codes like BD_WC73OP27_8809 (WC segment "73OP27" contains letters) match.
_FAULT_CODE_WC_PATTERN = r"([A-Z]{2,3}_WC[A-Z0-9]+_[A-Z0-9]+)"
# Bare N/P/B codes (DRS 故障描述 style). Bounded - no greedy \S* - so it captures
# just the code (e.g. "N504"), not trailing description text.
_FAULT_CODE_BARE_PATTERN = r"([NBP]\d{3,6}[A-Z]?)"


def _clean_fault_desc(value: str) -> str:
    value = _FAULT_CODE_PREFIX.sub("", value)
    return value.strip(_VALUE_QUOTES)


def _first_match(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = next((g for g in match.groups() if g), "")
        value = _strip_value(value)
        if value:
            return value
    return ""


def _extract_entities(question: str) -> dict[str, Any]:
    q = remove_redundant_spaces(question or "")
    entities: dict[str, Any] = {}

    device_code = _first_match(
        [
            r"设备编码\s*(?:为|是|:|：)?\s*([A-Za-z0-9][A-Za-z0-9_#./ -]{1,100}?)(?=的|上|，|,|。|出现|发生|时|$)",
            r"设备编号\s*(?:为|是|:|：)?\s*([A-Za-z0-9][A-Za-z0-9_#./ -]{1,100}?)(?=的|上|，|,|。|出现|发生|时|$)",
        ],
        q,
    )
    if device_code:
        entities["设备编码"] = device_code

    fault_code = _first_match(
        [
            r"故障代码\s*(?:为|是|:|：)?\s*([A-Za-z0-9][A-Za-z0-9_#./ -]{1,80}?)(?=对应|的|和|，|,|。|出现|发生|时|$)",
            r"报警代码\s*(?:为|是|:|：)?\s*([A-Za-z0-9][A-Za-z0-9_#./ -]{1,80}?)(?=对应|的|和|，|,|。|出现|发生|时|$)",
        ],
        q,
    )
    if fault_code:
        entities["故障代码"] = fault_code

    line = _first_match(
        [
            r"产线\s*([A-Za-z0-9][A-Za-z0-9_#./ -]{1,80}?)(?=的|上|，|,|。|出现|发生|时|$)",
            r"生产线\s*([A-Za-z0-9][A-Za-z0-9_#./ -]{1,80}?)(?=的|上|，|,|。|出现|发生|时|$)",
        ],
        q,
    )
    if line:
        entities["产线"] = line

    step = _first_match(
        [
            r"步骤序号\s*(?:为|是|:|：)?\s*(\d+)",
            r"步骤\s*(?:为|是|:|：)?\s*(\d+)",
            r"第\s*(\d+)\s*步",
        ],
        q,
    )
    if step:
        try:
            entities["步骤序号"] = int(step)
        except ValueError:
            logging.debug("table_entity_filter ignored non-numeric step value: %s", step)

    fault_desc = _first_match(
        [
            # Prefixed: 故障描述为 "xxx"
            r"故障描述\s*(?:为|是|:|：)?\s*[\"']?(.+?)[\"']?(?=，|,|。|的|和|对应|怎么|如何|处理|维修|步骤|故障类别|$)",
            # After 出现: 出现 "xxx" 故障/时
            r"出现\s*[\"']?(.+?)[\"']?\s*(?:时|故障)",
            # Quoted text (2-80 chars) followed by 故障/怎么/如何
            r"[\"']([^\"'']{2,80})[\"']\s*(?:故障|怎么|如何|处理|维修)",
            # Text before 故障 keyword
            r"([^\n，,。？?]{2,80}?)故障\s*(?:怎么|如何|处理|维修)",
        ],
        q,
    )
    if fault_desc:
        cleaned = _strip_value(fault_desc.rstrip("？?？"))
        cleaned = _clean_fault_desc(cleaned)
        if len(cleaned) < 2 or cleaned in {"什么", "哪些", "多少", "谁", "哪里", "哪个", "什么？", "什么原因"}:
            cleaned = ""
        if cleaned:
            entities["故障描述"] = cleaned

    # Recover a standalone fault code from the question body and merge it as
    # 故障代码 (without overriding an explicitly extracted one). The 故障描述 patterns
    # above can swallow a fault code quoted alongside the description
    # (e.g. 出现“NQ_WC120_036，12213 NG”时 captures the whole quoted string), so this
    # recovers the code so it can filter on the 故障代码 field. Only the unambiguous
    # `<PREFIX>_WC<digits>_<digits>` form is used — bare N/P codes are part of fault
    # descriptions for some datasets and would mis-filter.
    if "故障代码" not in entities:
        fault_code = _first_match([_FAULT_CODE_WC_PATTERN], q)
        if fault_code:
            entities["故障代码"] = fault_code

    # Weak extraction: when no explicit entity prefix found, try pattern-based extraction
    # from the question body (fault codes, device codes, line numbers)
    if not entities:
        entities = _weak_extract(q)

    return entities


def _weak_extract(question: str) -> dict[str, Any]:
    entities: dict[str, Any] = {}

    # Fault code patterns: NQ_WC120_039, BD_WC200_044, BD_WC73OP27_8809, N709, P705, etc.
    fc = _first_match(
        [
            _FAULT_CODE_WC_PATTERN,
            _FAULT_CODE_BARE_PATTERN,
        ],
        question,
    )
    if fc:
        entities["故障代码"] = fc

    # Device code: V-SZ-*
    dc = _first_match(
        [r"(V-SZ-[\w-]+)"],
        question,
    )
    if dc:
        entities["设备编码"] = dc

    # Line: XXXX-lineN or lineN
    ln = _first_match(
        [r"([A-Za-z0-9-]+[Ll]ine\d+)"],
        question,
    )
    if ln:
        entities["产线"] = ln

    return entities


def _repair_mojibake(value: str) -> str:
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _norm_display(value: Any) -> str:
    value = _repair_mojibake(str(value or "").strip())
    return re.sub(r"[\s_]+", "", value).lower()


def _field_keys_for_column(column: str, field_map: dict[str, str]) -> list[str]:
    aliases = _COLUMN_ALIASES.get(column, (column,))
    alias_norms = {_norm_display(alias) for alias in aliases}
    keys = []
    for key, display in field_map.items():
        if _norm_display(display) in alias_norms:
            keys.append(str(key))
    return keys


def _filter_field_for_column(column: str, value: Any, field_map: dict[str, str]) -> str:
    keys = _field_keys_for_column(column, field_map)
    if not keys:
        return ""

    if column == "步骤序号" or isinstance(value, int):
        for key in keys:
            if key.endswith("_long"):
                return key

    for key in keys:
        if key.endswith("_raw"):
            return f"{key}.keyword"
    for key in keys:
        if key.endswith("_tks"):
            return f"{key[: -len('_tks')]}_raw.keyword"
    for key in keys:
        if key.endswith("_kwd"):
            return key
    return keys[0]


def _entities_to_filters(entities: dict[str, Any], field_map: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    filters: dict[str, Any] = {}
    mapped: dict[str, str] = {}
    for column in ("设备编码", "故障代码", "产线", "步骤序号", "故障描述"):
        if column not in entities:
            continue
        field = _filter_field_for_column(column, entities[column], field_map)
        if not field:
            continue
        filters[field] = entities[column]
        mapped[column] = field
    return filters, mapped


def _build_fallback_filters(filters: dict[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []

    def add(plan: dict[str, Any]) -> None:
        if plan and plan not in plans:
            plans.append(plan)

    add(dict(filters))

    # Fallback 1: drop step_number (multi-step faults might not have step 1 in separate chunk)
    no_step = {k: v for k, v in filters.items() if not k.endswith("_long")}
    add(no_step)

    # Fallback 2: truncate fault_desc values containing commas to just the code portion
    # e.g. "BD_WC200_044，螺丝安装角度NG" -> "BD_WC200_044"
    truncated = {}
    for k, v in no_step.items():
        if isinstance(v, str) and any(ch in v for ch in ("，", ",", "；")):
            truncated[k] = v.split("，")[0].split(",")[0].split("；")[0].strip()
        else:
            truncated[k] = v
    add(truncated)

    # Fallback 3: drop fault_desc entirely (rely on device_code + line filters alone)
    no_fault_desc = {k: v for k, v in no_step.items() if not k.startswith("gu_zhang_miao_shu")}
    add(no_fault_desc)

    return plans
