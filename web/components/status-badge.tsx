import { clsx } from "clsx";

const STYLE: Record<string, string> = {
  active:    "bg-success/15 text-success border-success/30",
  ready:     "bg-success/15 text-success border-success/30",
  deployed:  "bg-success/15 text-success border-success/30",
  completed: "bg-success/15 text-success border-success/30",
  running:   "bg-brand-500/15 text-brand-400 border-brand-500/30",
  training:  "bg-brand-500/15 text-brand-400 border-brand-500/30",
  pending:   "bg-warning/15 text-warning border-warning/30",
  trained:   "bg-teal-500/15 text-teal-400 border-teal-500/30",
  error:     "bg-danger/15 text-danger border-danger/30",
  idle:      "bg-gray-700/50 text-gray-400 border-gray-700",
  inactive:  "bg-gray-700/50 text-gray-400 border-gray-700",
  stopped:   "bg-gray-700/50 text-gray-400 border-gray-700",
  archived:  "bg-gray-700/50 text-gray-500 border-gray-700",
  created:   "bg-gray-700/50 text-gray-400 border-gray-700",
};

const DOT: Record<string, string> = {
  active: "bg-success", ready: "bg-success", deployed: "bg-success", completed: "bg-success",
  running: "bg-brand-400 animate-pulse", training: "bg-brand-400 animate-pulse",
  pending: "bg-warning",
  error: "bg-danger",
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const style = STYLE[status] ?? "bg-gray-700/50 text-gray-400 border-gray-700";
  const dot = DOT[status];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        style,
        className
      )}
    >
      {dot && <span className={clsx("h-1.5 w-1.5 rounded-full flex-shrink-0", dot)} />}
      {status}
    </span>
  );
}
