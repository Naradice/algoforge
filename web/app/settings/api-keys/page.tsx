"use client";

export default function ApiKeysPage() {
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-medium text-white">API Keys</h2>
      <div className="rounded border border-yellow-800 bg-yellow-900/20 p-4">
        <p className="text-sm text-yellow-300">
          API key management is available when authentication is enabled.
          Set <code className="text-yellow-200">ALGOFORGE_NO_AUTH=0</code> in your environment to enable auth.
        </p>
      </div>
    </div>
  );
}
