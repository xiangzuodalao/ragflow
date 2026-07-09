import { Spin } from '@/components/ui/spin';
import { cn } from '@/lib/utils';
import FileError from '@/pages/document-viewer/file-error';
import { useEffect, useState } from 'react';
import * as XLSX from 'xlsx';
import { useFetchDocument } from './hooks';

interface ExcelCsvPreviewerProps {
  className?: string;
  url: string;
}

type Row = (string | number)[];
type SheetData = { name: string; rows: Row[] };

// Renders an xlsx workbook as scrollable HTML tables using SheetJS directly.
// (@js-preview/excel 1.7.14 fails to parse some valid xlsx files - its bundled
// reader returns a workbook without `Sheets`; SheetJS 0.18.5 parses them fine.)
export const ExcelCsvPreviewer: React.FC<ExcelCsvPreviewerProps> = ({
  className,
  url,
}) => {
  const { fetchDocument } = useFetchDocument();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sheets, setSheets] = useState<SheetData[]>([]);
  const [active, setActive] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const ret = await fetchDocument(url);
        const wb = XLSX.read(ret.data, { type: 'array' });
        const data: SheetData[] = (wb.SheetNames || []).map((name) => ({
          name,
          rows: XLSX.utils.sheet_to_json<Row>(wb.Sheets[name], {
            header: 1,
            raw: false,
            defval: '',
          }) as Row[],
        }));
        if (!cancelled) {
          setSheets(data);
          setActive(0);
        }
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [url, fetchDocument]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spin />
      </div>
    );
  }
  if (error) return <FileError>{error}</FileError>;
  if (!sheets.length) return <FileError>文件为空</FileError>;

  const sheet = sheets[active] || sheets[0];

  return (
    <div className={cn('flex flex-col w-full h-full', className)}>
      {sheets.length > 1 && (
        <div className="flex gap-2 px-3 py-2 border-b border-border-normal overflow-x-auto shrink-0">
          {sheets.map((s, i) => (
            <button
              key={s.name}
              type="button"
              onClick={() => setActive(i)}
              className={cn(
                'px-2 py-0.5 rounded text-sm whitespace-nowrap',
                i === active
                  ? 'bg-primary text-primary-foreground'
                  : 'text-text-sub-title hover:bg-bg-card',
              )}
            >
              {s.name}
            </button>
          ))}
        </div>
      )}
      <div className="flex-1 overflow-auto bg-background-paper">
        <table className="border-collapse text-sm">
          <tbody>
            {sheet.rows.map((row, i) => (
              <tr
                key={i}
                className={i === 0 ? 'bg-bg-card font-medium sticky top-0' : ''}
              >
                {row.map((cell, j) => (
                  <td
                    key={j}
                    className="border border-border-normal px-2 py-1 whitespace-nowrap align-top"
                  >
                    {String(cell ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
