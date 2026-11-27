/* Minimal streaming-safe COOP/COEP Service Worker for GitHub Pages
 * Enables SharedArrayBuffer & multi-threaded WASM SIMD without breaking 1GB streaming downloads. */
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

  self.addEventListener('fetch', (event) => {
    const req = event.request;
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    // Bypass Service Worker IPC streaming for large same-origin model binaries
    // (GGUF / ONNX). HTML/JS/WASM still get COOP/COEP so crossOriginIsolated stays true.
    const reqUrl = new URL(req.url);
    if (
      reqUrl.origin === self.location.origin &&
      (reqUrl.pathname.endsWith('.gguf') || reqUrl.pathname.endsWith('.onnx'))
    ) {
      return;
    }

    event.respondWith(
      fetch(req)
        .then((res) => {
          if (res.status === 0) return res;
          const headers = new Headers(res.headers);
          headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
          return new Response(res.body, {
            status: res.status,
            statusText: res.statusText,
            headers,
          });
        })
        .catch((err) => {
          throw err;
        }),
    );
  });
} else {
  (() => {
    const reloadedKey = 'coi_sw_reloaded_v1';
    if (window.crossOriginIsolated) {
      sessionStorage.removeItem(reloadedKey);
      return;
    }
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker
      .register(new URL('./coi-serviceworker.js', window.location.href).href)
      .then((reg) => {
        if (reg.active && !navigator.serviceWorker.controller) {
          if (!sessionStorage.getItem(reloadedKey)) {
            sessionStorage.setItem(reloadedKey, '1');
            window.location.reload();
          }
          return;
        }
        reg.addEventListener('updatefound', () => {
          const sw = reg.installing;
          if (!sw) return;
          sw.addEventListener('statechange', () => {
            if (sw.state === 'activated' && !window.crossOriginIsolated) {
              if (!sessionStorage.getItem(reloadedKey)) {
                sessionStorage.setItem(reloadedKey, '1');
                window.location.reload();
              }
            }
          });
        });
      })
      .catch(() => {
        // Ignore SW registration failure (e.g. restricted webview)
      });
  })();
}
