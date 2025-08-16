export const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

async function postJSON(url: string, body: any) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res;
}

export async function apiLoad(ckpt: string, device: string | null = null) {
  let res = await postJSON(`${API_BASE}/load`, { ckpt, device });
  if (res.status === 404) {
    // fallback: /api/gan/load
    res = await postJSON(`${API_BASE}/gan/load`, { ckpt, device });
  }
  if (!res.ok) throw new Error(`load failed: ${res.status}`);
  return res.json();
}

export async function apiGenerate(params: {
  n: number;
  nrow: number;
  seed: number;
  use_ema: boolean;
}) {
  const res = await postJSON(`${API_BASE}/gan/generate`, params);
  if (!res.ok) throw new Error(`generate failed: ${res.status}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
