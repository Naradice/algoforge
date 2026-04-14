/** @type {import('next').NextConfig} */
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
      // When WATCHPACK_POLLING is enabled (Docker + Windows volume mounts),
      // watchpack recursively scans every directory on each poll cycle.
      // Excluding node_modules and .next prevents ENOMEM on memory-constrained hosts.
      config.watchOptions = {
        ...config.watchOptions,
        ignored: ["**/.git/**", "**/node_modules/**", "**/.next/**"],
        poll: parseInt(process.env.WATCHPACK_POLLING_INTERVAL || "2000", 10),
        aggregateTimeout: 300,
      };
    }
    return config;
  },
};

module.exports = nextConfig;
