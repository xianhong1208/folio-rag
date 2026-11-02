import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The console is served by the FastAPI backend under /admin, so assets must resolve
// from /admin/assets/*. In dev, proxy the REST API to the running server.
const BACKEND = 'http://127.0.0.1:5039'

export default defineConfig({
  plugins: [react()],
  base: '/admin/',
  build: {
    outDir: '../static/console',
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
        },
      },
    },
  },
  server: {
    proxy: {
      // The backend's CSRF check wants Origin to match the host it sees; changeOrigin
      // rewrites Host to BACKEND, so present the same origin.
      '/api': { target: BACKEND, changeOrigin: true, headers: { origin: BACKEND } },
    },
  },
  // Vitest reads this same config. Pure-function suites run in Node (no jsdom),
  // so the test toolchain stays minimal.
  test: {
    environment: 'node',
    include: ['src/**/*.test.{js,jsx}'],
  },
})
