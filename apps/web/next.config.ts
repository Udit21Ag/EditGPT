import type { NextConfig } from "next";

import { loadRepoEnv } from "./repo-env";

loadRepoEnv();

const config: NextConfig = {
  reactStrictMode: true,
  // Uploads are large; the gateway owns storage, so the app never proxies image bytes.
  typedRoutes: true,
  // Playwright drives the dev server over 127.0.0.1 while Next serves localhost, and
  // Next 15 warns that a future major will reject the cross-origin asset requests.
  allowedDevOrigins: ["127.0.0.1"],
};

export default config;
