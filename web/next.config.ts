import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true,   // ⬅️ build won’t fail on lint errors
  },
};


export default nextConfig;
