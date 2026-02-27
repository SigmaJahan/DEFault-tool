import type { NextConfig } from "next";

// In production builds (next build --webpack), export to webapp/static/
// In dev (next dev --webpack), use default .next dir to avoid path conflicts
const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  ...(isProd ? { output: "export", distDir: "../static" } : {}),
  trailingSlash: true,
};

export default nextConfig;
