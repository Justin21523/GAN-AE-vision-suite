/// <reference types="vite/client" />

// Optional: declare custom env vars so TypeScript can provide type hints.
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
