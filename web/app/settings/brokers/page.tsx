"use client";

import { MT5Settings } from "@/components/mt5-settings";

export default function BrokersPage() {
  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-lg font-medium text-white">Broker Connections</h2>
        <p className="text-xs text-gray-400 mt-1">
          Configure real-time trading terminals and broker credentials for Live & Paper strategy runs.
        </p>
      </div>

      <MT5Settings />
    </div>
  );
}
