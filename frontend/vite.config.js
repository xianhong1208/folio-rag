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
      '/api': { target: BACKEND, changeOrigin: true },
    },
  },
})
