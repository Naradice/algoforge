"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";

interface BrokerClient {
  name: string;
  description: string;
  credentials_required: string[];
}

export default function BrokersPage() {
  const { data: clients } = useSWR<BrokerClient[]>("/api/v1/strategy-config/broker-clients", fetcher);

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-medium text-white">Broker Connections</h2>
      {!clients ? (
        <p className="text-gray-500 text-sm">Loading…</p>
      ) : (
        <div className="space-y-3">
          {clients.map((client) => (
            <div key={client.name} className="rounded border border-gray-700 bg-gray-900 p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-sm font-medium text-white capitalize">{client.name}</div>
                  <div className="text-xs text-gray-400 mt-1">{client.description}</div>
                  {client.credentials_required.length === 0 ? (
                    <span className="mt-2 inline-block rounded bg-green-900/40 px-2 py-0.5 text-xs text-green-300">
                      Connected (no credentials required)
                    </span>
                  ) : (
                    <span className="mt-2 inline-block rounded bg-gray-700 px-2 py-0.5 text-xs text-gray-400">
                      Not configured · requires: {client.credentials_required.join(", ")}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
