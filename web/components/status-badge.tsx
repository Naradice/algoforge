import { clsx } from "clsx";

const COLOR: Record<string, string> = {
  active: "bg-green-500/20 text-green-400",
  inactive: "bg-gray-700 text-gray-400",
  archived: "bg-gray-800 text-gray-500",
  running: "bg-blue-500/20 text-blue-400",
  pending: "bg-yellow-500/20 text-yellow-400",
  completed: "bg-green-500/20 text-green-400",
  error: "bg-red-500/20 text-red-400",
  stopped: "bg-gray-700 text-gray-400",
  deployed: "bg-green-500/20 text-green-400",
  training: "bg-blue-500/20 text-blue-400",
  trained: "bg-teal-500/20 text-teal-400",
  created: "bg-gray-700 text-gray-400",
  ready: "bg-green-500/20 text-green-400",
  idle: "bg-gray-700 text-gray-400",
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  return (
    <span className={clsx("rounded-full px-2 py-0.5 text-xs font-medium", COLOR[status] ?? "bg-gray-700 text-gray-400", className)}>
      {status}
    </span>
  );
}
