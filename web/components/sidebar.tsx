"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

const NAV = [
  { label: "Dashboard", href: "/dashboard" },
  { label: "Strategy", href: "/strategy" },
  { label: "ML Models", href: "/model" },
  { label: "Data", href: "/data" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-[220px] flex-shrink-0 flex-col border-r border-gray-800 bg-gray-950 p-4">
      <div className="mb-6 flex items-center gap-2">
        <span className="text-lg font-bold text-white">AlgoForge</span>
      </div>

      <nav className="flex flex-col gap-1">
        {NAV.map(({ label, href }) => (
          <Link
            key={href}
            href={href}
            className={clsx(
              "rounded px-3 py-2 text-sm transition-colors",
              pathname.startsWith(href)
                ? "bg-brand-500/10 text-brand-500 font-medium"
                : "text-gray-400 hover:bg-gray-800 hover:text-white"
            )}
          >
            {label}
          </Link>
        ))}
      </nav>

      <div className="mt-auto pt-4 border-t border-gray-800">
        <Link href="/settings" className="block rounded px-3 py-2 text-sm text-gray-400 hover:text-white hover:bg-gray-800">
          Settings
        </Link>
      </div>
    </aside>
  );
}
