import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: 'app',
  publicDir: fileURLToPath(new URL('./data', import.meta.url)),
  plugins: [react()],
  server: { port: 5173 },
  build: {
    outDir: '../dist',
    emptyOutDir: true
  }
});
