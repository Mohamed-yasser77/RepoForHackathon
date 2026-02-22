import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Allow cross-origin API calls to localhost:8000 in Dev
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
        ],
      },
    ];
  },
};

export default nextConfig;
