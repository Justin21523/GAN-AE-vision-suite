/// <reference types="vite/client" />

// Optional: 宣告我們用到的自訂變數，讓 TS 有型別提示
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
