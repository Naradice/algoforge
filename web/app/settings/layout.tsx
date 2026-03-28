"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const settingsTabs = [
  { href: "/settings/webhooks", label: "Webhooks" },
  { href: "/settings/api-keys", label: "API Keys" },
  { href: "/settings/brokers", label: "Brokers" },
];

export default function SettingsLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-white">Settings</h1>
      <div className="flex gap-1 border-b border-gray-700">
        {settingsTabs.map((tab) => (
          <Link
            key={tab.href}
            href={tab.href}
            className={`px-4 py-2 text-sm ${
              pathname === tab.href
                ? "border-b-2 border-brand-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            {tab.label}
          </Link>
        ))}
      </div>
      {children}
    </div>
  );
}
