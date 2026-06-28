import { useEffect, useMemo, useState } from "react";
import { apiRunCloneConfig, apiRunDetail, apiRunNotesSet, apiRunsCompare, apiRunsList } from "../api";
import { FileBrowser } from "../components/FileBrowser";
import { MetricsChart } from "../components/MetricsChart";

export function RunsPage() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [selected2, setSelected2] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detail2, setDetail2] = useState(null);
  const [error, setError] = useState("");
  const [metricKey, setMetricKey] = useState("train_loss");
  const [tagsText, setTagsText] = useState("");
  const [noteText, setNoteText] = useState("");
  const [noteStatus, setNoteStatus] = useState("");
  const [cloneStatus, setCloneStatus] = useState("");
  const [compareStatus, setCompareStatus] = useState("");
  const [browsePath, setBrowsePath] = useState(null);

  const refresh = async () => {
    setError("");
    try {
      const r = await apiRunsList(200);
      setRuns(r.runs || []);
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const parseTags = (txt) =>
    String(txt || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);

  const openRun = async (run) => {
    setSelected(run);
    setDetail(null);
    setError("");
    setNoteStatus("");
    setCloneStatus("");
    setCompareStatus("");
    setBrowsePath(null);
    try {
      const d = await apiRunDetail(run.id, 200);
      setDetail(d);
      const n = d?.notes || run?.notes || { tags: [], note: "" };
      setTagsText((n.tags || []).join(", "));
      setNoteText(n.note || "");
    } catch (e) {
      setError(String(e));
    }
  };

  const openRun2 = async (run) => {
    setSelected2(run);
    setDetail2(null);
    setError("");
    try {
      const d = await apiRunDetail(run.id, 200);
      setDetail2(d);
    } catch (e) {
      setError(String(e));
    }
  };

  const saveNotes = async () => {
    if (!selected) return;
    setError("");
    setNoteStatus("Saving…");
    try {
      const out = await apiRunNotesSet(selected.id, parseTags(tagsText), String(noteText || ""));
      setNoteStatus("Saved");
      setDetail((d) => ({ ...(d || {}), notes: out }));
      setRuns((arr) => (arr || []).map((r) => (r.id === selected.id ? { ...r, notes: out } : r)));
    } catch (e) {
      setNoteStatus("");
      setError(String(e));
    }
  };

  const cloneConfig = async () => {
    if (!selected) return;
    setError("");
    setCloneStatus("Cloning…");
    try {
      const out = await apiRunCloneConfig(selected.id);
      setCloneStatus(out?.path ? `Cloned to ${out.path}` : "Cloned");
      const p = String(out?.path || "");
      if (p) {
        // FileBrowser prefers directories; jump to the parent.
        const parent = p.replace(/\\/g, "/").split("/").slice(0, -1).join("/") || ".";
        setBrowsePath(parent);
      }
    } catch (e) {
      setCloneStatus("");
      setError(String(e));
    }
  };

  const exportCompare = async (fmt) => {
    if (!selected || !selected2) return;
    setError("");
    setCompareStatus("Exporting…");
    try {
      const out = await apiRunsCompare(selected.id, selected2.id, fmt === "md" ? "markdown" : "json", 500);
      const isMd = fmt === "md";
      const content = isMd ? String(out || "") : JSON.stringify(out, null, 2);
      const blob = new Blob([content], { type: isMd ? "text/markdown" : "application/json" });
      const url = URL.createObjectURL(blob);
      const safe = (s) => String(s || "").replace(/[^\w.-]+/g, "_").slice(0, 120);
      const name = `compare_${safe(selected.id)}_vs_${safe(selected2.id)}.${isMd ? "md" : "json"}`;
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setCompareStatus("Downloaded");
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setCompareStatus("");
      setError(String(e));
    }
  };

  const availableKeys = useMemo(() => {
    const keys = new Set();
    for (const x of detail?.metrics_tail || []) {
      Object.keys(x).forEach((k) => keys.add(k));
    }
    for (const x of detail2?.metrics_tail || []) {
      Object.keys(x).forEach((k) => keys.add(k));
    }
    // prefer numeric-ish keys
    const arr = Array.from(keys).filter((k) => !["split", "event"].includes(k));
    arr.sort();
    return arr;
  }, [detail?.metrics_tail, detail2?.metrics_tail]);

  const chartSeries = useMemo(() => {
    const build = (d, label, color) => {
      const pts = [];
      for (const x of d?.metrics_tail || []) {
        const step = x.step ?? x.epoch ?? null;
        const y = x[metricKey];
        if (step == null) continue;
        if (typeof y !== "number") continue;
        pts.push({ x: Number(step), y: Number(y) });
      }
      return { label, color, points: pts };
    };
    const s = [];
    if (detail) s.push(build(detail, selected?.id || "run1", "#4f46e5"));
    if (detail2) s.push(build(detail2, selected2?.id || "run2", "#14b8a6"));
    return s;
  }, [detail, detail2, metricKey, selected?.id, selected2?.id]);

  return (
    <div className="Grid">
      <div className="Card">
        <div className="CardHeader">
          <h2>Runs</h2>
          <p>Scans `./logs/**/meta.json` and shows recent runs.</p>
        </div>
        <div className="CardBody">
          {error ? <div className="Status">Error: {error}</div> : null}
          <div className="Row" style={{ marginBottom: 10 }}>
            <button className="ButtonGhost" onClick={refresh}>
              Refresh
            </button>
          </div>
          <div className="JobList">
            {(runs || []).map((r) => (
              <div key={r.id} className="JobItem">
                <div className="JobItemHeader">
                  <div>
                    <div className="JobName">{r.id}</div>
                    <div className="JobMeta">
                      {r.script || "unknown"} · {r.created_at || "unknown"}
                    </div>
                    {r?.notes?.tags?.length ? (
                      <div className="Row" style={{ gap: 6, marginTop: 6, flexWrap: "wrap" }}>
                        {r.notes.tags.slice(0, 6).map((t) => (
                          <span key={t} className="Pill">
                            {t}
                          </span>
                        ))}
                        {r.notes.tags.length > 6 ? <span className="Pill">…</span> : null}
                      </div>
                    ) : null}
                  </div>
                  <div className="Row">
                    <button className="ButtonGhost" onClick={() => openRun(r)}>
                      Open
                    </button>
                    <button className="ButtonGhost" onClick={() => openRun2(r)}>
                      Compare
                    </button>
                  </div>
                </div>
              </div>
            ))}
            {!runs?.length ? <div className="Status">No runs found yet.</div> : null}
          </div>
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Details</h2>
          <p>{selected ? selected.id : "Select a run"}</p>
        </div>
        <div className="CardBody">
          <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
            <div className="Field" style={{ minWidth: 220, flex: 0 }}>
              <label>Metric key</label>
              <select className="Select" value={metricKey} onChange={(e) => setMetricKey(e.target.value)}>
                {availableKeys.length ? null : <option value={metricKey}>{metricKey}</option>}
                {availableKeys.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </div>
            <span className="Pill">{selected2 ? `compare: ${selected2.id}` : "compare: (none)"}</span>
            <div style={{ flex: 1 }} />
            <button className="Button" disabled={!selected} onClick={saveNotes}>
              Save Notes
            </button>
            <button className="ButtonGhost" disabled={!selected} onClick={cloneConfig}>
              Clone config
            </button>
            <button className="ButtonGhost" disabled={!selected || !selected2} onClick={() => exportCompare("json")}>
              Export compare (JSON)
            </button>
            <button className="ButtonGhost" disabled={!selected || !selected2} onClick={() => exportCompare("md")}>
              Export compare (MD)
            </button>
          </div>

          {noteStatus ? <div className="Status">{noteStatus}</div> : null}
          {cloneStatus ? <div className="Status">{cloneStatus}</div> : null}
          {compareStatus ? <div className="Status">{compareStatus}</div> : null}

          {selected ? (
            <div style={{ marginBottom: 12 }}>
              <div className="Row" style={{ gap: 10, flexWrap: "wrap" }}>
                <div className="Field" style={{ minWidth: 320, flex: 1 }}>
                  <label>Tags (comma separated)</label>
                  <input className="Input" value={tagsText} onChange={(e) => setTagsText(e.target.value)} placeholder="baseline, wgangp, fp16" />
                </div>
              </div>
              <div className="Field" style={{ marginTop: 8 }}>
                <label>Note</label>
                <textarea className="Textarea" value={noteText} onChange={(e) => setNoteText(e.target.value)} placeholder="What changed? Why this run matters?" />
              </div>
            </div>
          ) : null}

          {chartSeries.some((s) => (s.points || []).length) ? (
            <div style={{ marginBottom: 10 }}>
              <MetricsChart series={chartSeries} />
            </div>
          ) : (
            <div className="Status">No numeric series for this key in metrics tail.</div>
          )}

          {detail?.meta ? (
            <div className="Code">{JSON.stringify(detail.meta, null, 2)}</div>
          ) : (
            <div className="Status">No meta loaded.</div>
          )}
          {detail?.metrics_tail?.length ? (
            <div style={{ marginTop: 10 }}>
              <div className="Status">metrics.jsonl (tail)</div>
              <textarea className="Textarea" readOnly value={detail.metrics_tail.map((x) => JSON.stringify(x)).join("\n")} />
            </div>
          ) : null}
          {selected ? (
            <div style={{ marginTop: 14 }}>
              <FileBrowser initialPath={selected.id} title="Browse Run Files" />
            </div>
          ) : null}
          {browsePath ? (
            <div style={{ marginTop: 14 }}>
              <FileBrowser initialPath={browsePath} title="Browse Cloned Config" />
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
