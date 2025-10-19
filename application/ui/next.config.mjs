// next.config.mjs
/** @type {import('next').NextConfig} */
const nextConfig = {
    experimental: {
        serverActions: {
            bodySizeLimit: '20mb', // pick your limit
        },
    },
    async rewrites() {
        return [
            { source: '/media/:path*', destination: 'http://api:8000/media/:path*' }, // through Next
            // Optional: expose Django under /backend/* if you want
            // { source: '/backend/:path*', destination: 'http://api:8000/:path*' },
        ];
    },
    // allow your LAN origin during dev
    allowedDevOrigins: ['http://192.168.137.1:3000'],
    reactStrictMode: true,
    output: 'standalone',
};

export default nextConfig;