import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Uploads are large; the gateway owns storage, so the app never proxies image bytes.
  experimental: { typedRoutes: true },
};

export default config;
