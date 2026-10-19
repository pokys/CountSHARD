import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  base: './',
  plugins: [
    react(),
    VitePWA({
      registerType: 'prompt',
      injectRegister: false,
      includeAssets: ['icon.svg'],
      manifest: {
        name: 'CountIT – počítání objektů na fotografii',
        short_name: 'CountIT',
        description: 'Offline počítání trubek, tyčí a dalších objektů na fotografii. Snímky neopouštějí zařízení.',
        lang: 'cs',
        start_url: './',
        scope: './',
        display: 'standalone',
        background_color: '#0f172a',
        theme_color: '#0f172a',
        icons: [
          { src: 'icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' },
          { src: 'icon-maskable.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'maskable' },
        ],
      },
      workbox: {
        // The analysis worker embeds OpenCV.js (~13 MB uncompressed) and must be precached for offline use.
        maximumFileSizeToCacheInBytes: 40 * 1024 * 1024,
        globPatterns: ['**/*.{js,css,html,svg,png,webmanifest}'],
        cleanupOutdatedCaches: true,
        navigateFallback: 'index.html',
      },
    }),
  ],
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 16000 },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    testTimeout: 120000,
    hookTimeout: 120000,
  },
});
