import { useEffect, useMemo, useRef, useState } from "react";
import { apiJobManifest, apiJobsCancel, apiJobsGet, apiJobsLogs } from "../api";
import { FileBrowser } from "./FileBrowser";
import { apiFsFileUrl, apiFsList } from "../api";
import { API_BASE } from "../api";
import { MetricsChart } from "./MetricsChart";

function formatStatus(status) {
  if (!status) return "unknown";
  return status;
}

export function JobPanel({ jobId, onClose }) {
  const [job, setJob] = useState(null);
  const [lines, setLines] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [filesPath, setFilesPath] = useState(null);
  const [quick, setQuick] = useState([]);
  const [manifest, setManifest] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const esRef = useRef(null);

  const isDone = useMemo(() => {
    const st = job?.status;
    return st === "succeeded" || st === "failed" || st === "canceled";
  }, [job]);

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const j = await apiJobsGet(jobId);
      setJob(j);
      const logs = await apiJobsLogs(jobId, 400);
      setLines(logs);
      if (j?.status === "succeeded" || j?.status === "failed" || j?.status === "canceled") {
        try {
          const m = await apiJobManifest(jobId);
          setManifest(m);
        } catch {
          // ignore
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const computeQuickLinks = async (j) => {
    const artifacts = (j?.artifacts || []).slice();
    const roots = [];
    if (j?.job_dir) roots.push(j.job_dir);
    for (const a of artifacts) roots.push(a);

    const wanted = new Set([
      "report.json",
      "meta.json",
      "config_resolved.yaml",
      "metrics.jsonl",
      "samples_train.png",
      "samples_val.png",
      "samples_input.png",
      "samples_recon.png",
    ]);

    const out = [];
    for (const r of roots.slice(0, 4)) {
      try {
        const listing = await apiFsList(r);
        for (const e of listing.entries || []) {
          if (e.type === "file" && wanted.has(e.name)) {
            out.push({ label: e.name, path: e.path });
          }
        }
      } catch {
        // ignore
      }
    }
    setQuick(out.slice(0, 12));
  };

  useEffect(() => {
    // Prefer SSE stream; fall back to polling if it fails.
    setLines([]);
    setManifest(null);
    setMetrics([]);
    setError("");
    refresh();

    try {
      const es = new EventSource(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/events`);
      esRef.current = es;
      es.addEventListener("job", (ev) => {
        try {
          const j = JSON.parse(ev.data);
          setJob(j);
        } catch {
          // ignore
        }
      });
      es.addEventListener("log", (ev) => {
        try {
          const payload = JSON.parse(ev.data);
          const newLines = payload.lines || [];
          setLines((prev) => {
            const merged = prev.concat(newLines);
            return merged.slice(-2000);
          });
        } catch {
          // ignore
        }
      });
      es.addEventListener("metrics", (ev) => {
        try {
          const payload = JSON.parse(ev.data);
          const entries = payload.entries || [];
          setMetrics((prev) => {
            const merged = prev.concat(entries);
            return merged.slice(-500);
          });
        } catch {
          // ignore
        }
      });
      es.addEventListener("manifest", (ev) => {
        try {
          setManifest(JSON.parse(ev.data));
        } catch {
          // ignore
        }
      });
      es.addEventListener("done", () => {
        es.close();
      });
      es.onerror = () => {
        es.close();
      };
      return () => {
        es.close();
      };
    } catch {
      const t = setInterval(() => refresh(), 1500);
      return () => clearInterval(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  useEffect(() => {
    if (!job) return;
    computeQuickLinks(job);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, job?.status, job?.finished_at]);

  const onCancel = async () => {
    try {
      await apiJobsCancel(jobId);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="Card">
      <div className="CardHeader">
        <div className="Row" style={{ justifyContent: "space-between" }}>
          <div>
            <h2>Job</h2>
            <p className="Status">
              <span className="Pill">
                {job?.name || jobId} · {formatStatus(job?.status)}
                {job?.pid ? ` · pid=${job.pid}` : ""}
              </span>
            </p>
          </div>
          <div className="Row">
            {!isDone ? (
              <button className="ButtonGhost" onClick={onCancel} disabled={loading}>
                Cancel
              </button>
            ) : null}
            <button
              className="ButtonGhost"
              onClick={() => {
                setFilesPath(job?.job_dir || job?.log_path || ".");
                setShowFiles((v) => !v);
              }}
            >
              {showFiles ? "Hide Files" : "Browse Files"}
            </button>
            <button className="ButtonGhost" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>

      <div className="CardBody">
        {error ? <div className="Status">Error: {error}</div> : null}
        {job?.cmd?.length ? (
          <div className="Code" style={{ marginBottom: 10 }}>
            $ {job.cmd.join(" ")}
          </div>
        ) : null}
        {quick.length ? (
          <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
            <span className="Status">Quick open:</span>
            {quick.map((q) => (
              <a key={q.path} className="ButtonGhost" href={apiFsFileUrl(q.path)} target="_blank" rel="noreferrer">
                {q.label}
              </a>
            ))}
          </div>
        ) : null}
        {job?.artifacts?.length ? (
          <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
            <span className="Status">Artifacts:</span>
            {job.artifacts.map((p) => (
              <button
                key={p}
                className="ButtonGhost"
                onClick={() => {
                  setFilesPath(p);
                  setShowFiles(true);
                }}
              >
                Open
              </button>
            ))}
            {job.job_dir ? (
              <button
                className="ButtonGhost"
                onClick={() => {
                  setFilesPath(job.job_dir);
                  setShowFiles(true);
                }}
              >
                Job dir
              </button>
            ) : null}
          </div>
        ) : null}
        <textarea className="Textarea" readOnly value={lines.join("\n")} />

        {metrics.length ? (
          <div style={{ marginTop: 14 }}>
            <div className="Row" style={{ justifyContent: "space-between", marginBottom: 8, flexWrap: "wrap" }}>
              <div className="Status">Live metrics (tail)</div>
              <span className="Pill">{metrics.length} events</span>
            </div>
            <div style={{ marginBottom: 10 }}>
              <MetricsChart
                series={[
                  {
                    label: "loss",
                    color: "#4f46e5",
                    points: metrics
                      .map((m) => {
                        const x = m.step ?? m.epoch ?? null;
                        const y =
                          typeof m.g_loss === "number"
                            ? m.g_loss
                            : typeof m.train_loss === "number"
                              ? m.train_loss
                              : typeof m.val_loss === "number"
                                ? m.val_loss
                                : null;
                        if (x == null || y == null) return null;
                        return { x: Number(x), y: Number(y) };
                      })
                      .filter(Boolean),
                  },
                ]}
              />
            </div>
            <textarea className="Textarea" readOnly value={metrics.slice(-30).map((m) => JSON.stringify(m)).join("\n")} />
          </div>
        ) : null}

        {manifest?.artifacts?.length ? (
          <div style={{ marginTop: 14 }}>
            <div className="Row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
              <div className="Status">Manifest artifacts</div>
              <button
                className="ButtonGhost"
                onClick={() => {
                  setFilesPath(job?.job_dir || ".");
                  setShowFiles(true);
                }}
              >
                Browse all
              </button>
            </div>
            <div className="JobList">
              {manifest.artifacts.map((a, idx) => (
                <div key={`${a.path}-${idx}`} className="JobItem">
                  <div className="JobItemHeader">
                    <div>
                      <div className="JobName">{a.display_name || a.path}</div>
                      <div className="JobMeta">{a.type}</div>
                    </div>
                    <div className="Row">
                      <a className="ButtonGhost" href={apiFsFileUrl(a.path)} target="_blank" rel="noreferrer">
                        Open
                      </a>
                      <button
                        className="ButtonGhost"
                        onClick={() => {
                          setFilesPath(a.path);
                          setShowFiles(true);
                        }}
                      >
                        Browse
                      </button>
                    </div>
                  </div>
                  {a.type === "image" ? (
                    <div style={{ marginTop: 8 }}>
                      <img className="PreviewImage" src={apiFsFileUrl(a.path)} alt={a.display_name || a.path} />
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {showFiles ? (
          <div style={{ marginTop: 14 }}>
            <FileBrowser initialPath={filesPath || "."} title="Browse Artifacts" />
          </div>
        ) : null}
      </div>
    </div>
  );
}
