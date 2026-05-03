/** @type {import('next').NextConfig} */
const watchpackPollingEnabled = ["1", "true", "yes"].includes(
  String(process.env.WATCHPACK_POLLING || "").toLowerCase()
);

const nextConfig = {
  output: "standalone",

  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/:path*`,
      },
    ];
  },

  webpack(config, { dev }) {
    if (dev) {
      // Keep large generated/vendor trees out of the dev watcher. Polling is opt-in
      // because recursive scans over Docker Desktop bind mounts can exhaust memory.
      config.watchOptions = {
        ...config.watchOptions,
        ignored: ["**/.git/**", "**/node_modules/**", "**/.next/**"],
        ...(watchpackPollingEnabled
          ? {
              poll: parseInt(process.env.WATCHPACK_POLLING_INTERVAL || "15000", 10),
              aggregateTimeout: 300,
            }
          : {}),
      };
    }
    return config;
  },
};

module.exports = nextConfig;
