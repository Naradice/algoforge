"use client";

import { useEffect } from "react";

export default function ErrorPage({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-4">
      <p className="text-lg font-medium text-red-400">Something went wrong</p>
      <p className="text-sm text-gray-400">{error.message}</p>
      <button
        onClick={reset}
        className="rounded border border-gray-700 px-4 py-2 text-sm text-gray-300 hover:text-white"
      >
        Try again
      </button>
    </div>
  );
}
