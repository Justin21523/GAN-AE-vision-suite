/**
 * Frontend API helpers for the FastAPI backend.
 *
 * Notes:
 * - `API_BASE` defaults to `/api` so Vite's dev proxy can route requests.
 * - Some endpoints have historical aliases; we try a fallback path on 404.
 */
export const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

/** POST JSON helper that returns the raw `Response` for custom handling. */
async function postJSON(url: string, body: any) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res;
}

async function parseError(res: Response) {
  try {
    const data = await res.json();
    const msg = data?.error?.message || data?.detail || `HTTP ${res.status}`;
    return new Error(msg);
  } catch {
    return new Error(`HTTP ${res.status}`);
  }
}

/**
 * Ask the backend to load a checkpoint on a specific device.
 *
 * Backend endpoints may be either:
 * - `/api/load`
 * - `/api/gan/load` (fallback)
 */
export async function apiLoad(ckpt: string, device: string | null = null) {
  let res = await postJSON(`${API_BASE}/load`, { ckpt, device });
  if (res.status === 404) {
    // fallback: /api/gan/load
    res = await postJSON(`${API_BASE}/gan/load`, { ckpt, device });
  }
  if (!res.ok) throw await parseError(res);
  return res.json();
}

/**
 * Generate a sample grid image and return a blob URL suitable for `<img src=...>`.
 *
 * Backend endpoints may be either:
 * - `/api/gan/generate`
 * - `/api/generate` (fallback)
 */
export async function apiGenerate(params: {
  n: number;
  nrow: number;
  seed: number;
  use_ema: boolean;
}) {
  let res = await postJSON(`${API_BASE}/gan/generate`, params);
  if (res.status === 404) {
    res = await postJSON(`${API_BASE}/generate`, params);
  }
  if (!res.ok) throw await parseError(res);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export type JobType =
  | "train_gan"
  | "train_ae"
  | "sample_gan"
  | "eval_fid"
  | "eval_gan_pipeline"
  | "validate_data"
  | "prepare_data"
  | "prepare_demo"
  | "data_report";

export type JobInfo = {
  id: string;
  name: string;
  status: string;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  return_code?: number | null;
  log_path: string;
  job_dir: string;
  artifacts: string[];
  pid?: number | null;
  cmd: string[];
};

export async function apiJobsList(): Promise<JobInfo[]> {
  const res = await fetch(`${API_BASE}/jobs`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiJobsStart(type: JobType, args: any): Promise<JobInfo> {
  const res = await postJSON(`${API_BASE}/jobs/start`, { type, args });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiJobsGet(jobId: string): Promise<JobInfo> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiJobsLogs(jobId: string, tail: number = 200): Promise<string[]> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/logs?tail=${encodeURIComponent(String(tail))}`);
  if (!res.ok) throw await parseError(res);
  const body = await res.json();
  return body.lines || [];
}

export async function apiJobsCancel(jobId: string): Promise<void> {
  const res = await postJSON(`${API_BASE}/jobs/${jobId}/cancel`, {});
  if (!res.ok) throw await parseError(res);
}

export type FSListEntry = {
  name: string;
  path: string;
  type: "file" | "dir";
  size: number;
  mtime: number;
};

export type FSListResponse = {
  path: string;
  allowed_roots: string[];
  entries: FSListEntry[];
};

export async function apiFsList(path: string): Promise<FSListResponse> {
  const res = await fetch(`${API_BASE}/fs/list?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiFsRead(path: string, maxBytes: number = 200000): Promise<{ path: string; text: string; truncated: boolean; size: number }> {
  const res = await fetch(
    `${API_BASE}/fs/read?path=${encodeURIComponent(path)}&max_bytes=${encodeURIComponent(String(maxBytes))}`
  );
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export function apiFsFileUrl(path: string): string {
  return `${API_BASE}/fs/file?path=${encodeURIComponent(path)}`;
}

export async function apiFsWrite(path: string, text: string, overwrite: boolean = false): Promise<any> {
  const res = await postJSON(`${API_BASE}/fs/write`, { path, text, overwrite });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiFsMkdir(path: string): Promise<any> {
  const res = await postJSON(`${API_BASE}/fs/mkdir`, { path, parents: true, exist_ok: true });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiConfigValidate(kind: "auto" | "gan" | "ae", text: string): Promise<any> {
  const res = await postJSON(`${API_BASE}/config/validate`, { kind, text });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export type ConfigOverrideItem = { path: string; value: string; type?: "auto" | "string" | "int" | "float" | "bool" | "json" };

export async function apiConfigApplyOverrides(text: string, overrides: ConfigOverrideItem[]): Promise<{ ok: boolean; patched: string; applied: any[] }> {
  const res = await postJSON(`${API_BASE}/config/apply_overrides`, { text, overrides });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiConfigApplyOverlay(baseText: string, overlayText: string): Promise<{ ok: boolean; patched: string }> {
  const res = await postJSON(`${API_BASE}/config/apply_overlay`, { base_text: baseText, overlay_text: overlayText });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiConfigApplyOverlayPath(baseText: string, overlayPath: string): Promise<{ ok: boolean; patched: string }> {
  const res = await postJSON(`${API_BASE}/config/apply_overlay_path`, { base_text: baseText, overlay_path: overlayPath });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiConfigOverlaysList(dir: string = "./.ai_cache/configs/overrides"): Promise<any> {
  const res = await fetch(`${API_BASE}/config/overlays?dir=${encodeURIComponent(dir)}`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export type CapabilityJobField = {
  key: string;
  label: string;
  type: "string" | "number" | "boolean" | "path";
  required: boolean;
  default: any;
  placeholder?: string | null;
  help?: string | null;
  choices?: any[] | null;
};

export type CapabilityJob = {
  type: JobType | string;
  label: string;
  description: string;
  args: CapabilityJobField[];
};

export async function apiCapabilities(): Promise<{ jobs: CapabilityJob[] }> {
  const res = await fetch(`${API_BASE}/capabilities`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiRunsList(limit: number = 200): Promise<any> {
  const res = await fetch(`${API_BASE}/runs?limit=${encodeURIComponent(String(limit))}`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiRunDetail(runId: string, tailMetrics: number = 200): Promise<any> {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}?tail_metrics=${encodeURIComponent(String(tailMetrics))}`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiRunNotesGet(runId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}/notes`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiRunNotesSet(runId: string, tags: string[], note: string): Promise<any> {
  const res = await postJSON(`${API_BASE}/runs/${encodeURIComponent(runId)}/notes`, { tags, note });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiRunCloneConfig(runId: string, dest: string | null = null): Promise<any> {
  const url = dest
    ? `${API_BASE}/runs/${encodeURIComponent(runId)}/clone_config?dest=${encodeURIComponent(dest)}`
    : `${API_BASE}/runs/${encodeURIComponent(runId)}/clone_config`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export async function apiRunsCompare(run1: string, run2: string, format: "json" | "markdown" = "json", tailMetrics: number = 500): Promise<any> {
  const url = `${API_BASE}/runs/compare?run1=${encodeURIComponent(run1)}&run2=${encodeURIComponent(run2)}&format=${encodeURIComponent(
    format
  )}&tail_metrics=${encodeURIComponent(String(tailMetrics))}`;
  const res = await fetch(url);
  if (!res.ok) throw await parseError(res);
  if (format === "markdown") return res.text();
  return res.json();
}

export async function apiJobManifest(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/manifest`);
  if (!res.ok) throw await parseError(res);
  return res.json();
}
