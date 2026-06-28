import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFsFileUrl, apiFsList, apiFsRead } from "../api";

function humanSize(n) {
  const x = Number(n || 0);
  if (x < 1024) return `${x} B`;
  if (x < 1024 * 1024) return `${(x / 1024).toFixed(1)} KB`;
  if (x < 1024 * 1024 * 1024) return `${(x / (1024 * 1024)).toFixed(1)} MB`;
  return `${(x / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function isImage(name) {
  const s = name.toLowerCase();
  return s.endsWith(".png") || s.endsWith(".jpg") || s.endsWith(".jpeg") || s.endsWith(".webp") || s.endsWith(".bmp");
}

function isText(name) {
  const s = name.toLowerCase();
  return (
    s.endsWith(".txt") ||
    s.endsWith(".log") ||
    s.endsWith(".json") ||
    s.endsWith(".jsonl") ||
    s.endsWith(".yaml") ||
    s.endsWith(".yml") ||
    s.endsWith(".md")
  );
}

function getRecent() {
  try {
    const raw = localStorage.getItem("fs_recent");
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function pushRecent(p) {
  try {
    const prev = getRecent();
    const next = [p, ...prev.filter((x) => x !== p)].slice(0, 10);
    localStorage.setItem("fs_recent", JSON.stringify(next));
  } catch {
    // ignore
  }
}

export function FileBrowser({ initialPath = ".", title = "Files" }) {
  const [path, setPath] = useState(initialPath);
  const [list, setList] = useState(null);
  const [selected, setSelected] = useState(null);
  const [textPreview, setTextPreview] = useState(null);
  const [error, setError] = useState("");
  const [recent, setRecent] = useState(getRecent());

  const breadcrumbs = useMemo(() => {
    const p = (list?.path || path || "").replace(/\\/g, "/");
    const parts = p.split("/").filter(Boolean);
    const crumbs = [];
    let cur = p.startsWith("/") ? "/" : "";
    for (const part of parts) {
      cur = cur === "/" ? `/${part}` : cur ? `${cur}/${part}` : part;
      crumbs.push({ label: part, path: cur });
    }
    return crumbs;
  }, [list?.path, path]);

  const refresh = useCallback(async (p) => {
    setError("");
    setSelected(null);
    setTextPreview(null);
    try {
      const r = await apiFsList(p);
      setList(r);
      setPath(r.path);
      pushRecent(r.path);
      setRecent(getRecent());
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh(initialPath);
  }, [initialPath, refresh]);

  const openEntry = async (e) => {
    setError("");
    setSelected(e);
    setTextPreview(null);
    if (e.type === "dir") {
      await refresh(e.path);
      return;
    }
    if (isText(e.name)) {
      try {
        const r = await apiFsRead(e.path, 250000);
        setTextPreview(r);
      } catch (err) {
        setError(String(err));
      }
    }
  };

  return (
    <div className="Card">
      <div className="CardHeader">
        <h2>{title}</h2>
        <p>
          <span className="Pill">{list?.path || path}</span>
        </p>
      </div>
      <div className="CardBody">
        {error ? <div className="Status">Error: {error}</div> : null}

        <div className="Row" style={{ marginBottom: 10 }}>
          <input className="Input" value={path} onChange={(ev) => setPath(ev.target.value)} />
          <button className="ButtonGhost" onClick={() => refresh(path)}>
            Go
          </button>
        </div>

        <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
          <span className="Status">Shortcuts:</span>
          {["./outputs", "./logs", "./data", "./.ai_cache", "./.ai_cache/jobs"].map((p) => (
            <button key={p} className="ButtonGhost" onClick={() => refresh(p)}>
              {p}
            </button>
          ))}
        </div>

        {recent.length ? (
          <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
            <span className="Status">Recent:</span>
            {recent.slice(0, 6).map((p) => (
              <button key={p} className="ButtonGhost" onClick={() => refresh(p)}>
                {p}
              </button>
            ))}
          </div>
        ) : null}

        <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
          <span className="Status">Roots:</span>
          {(list?.allowed_roots || []).map((r) => (
            <button key={r} className="ButtonGhost" onClick={() => refresh(r)}>
              {r}
            </button>
          ))}
        </div>

        <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
          <span className="Status">Path:</span>
          <button className="ButtonGhost" onClick={() => refresh(list?.allowed_roots?.[0] || ".")}>
            repo
          </button>
          {breadcrumbs.map((c) => (
            <button key={c.path} className="ButtonGhost" onClick={() => refresh(c.path)}>
              {c.label}
            </button>
          ))}
        </div>

        <div className="Grid">
          <div className="Card" style={{ boxShadow: "none" }}>
            <div className="CardHeader">
              <h2>Entries</h2>
              <p>{list?.entries?.length || 0} items</p>
            </div>
            <div className="CardBody">
              <div className="JobList">
                {(list?.entries || []).map((e) => (
                  <div key={e.path} className="JobItem" style={{ cursor: "pointer" }} onClick={() => openEntry(e)}>
                    <div className="JobItemHeader">
                      <div>
                        <div className="JobName">
                          {e.type === "dir" ? "[dir]" : "[file]"} {e.name}
                        </div>
                        <div className="JobMeta">
                          {humanSize(e.size)} · {new Date(e.mtime * 1000).toLocaleString()}
                        </div>
                      </div>
                      {e.type === "file" ? (
                        <a className="ButtonGhost" href={apiFsFileUrl(e.path)} target="_blank" rel="noreferrer">
                          Download
                        </a>
                      ) : null}
                    </div>
                  </div>
                ))}
                {!list?.entries?.length ? <div className="Status">Empty directory.</div> : null}
              </div>
            </div>
          </div>

          <div className="Card" style={{ boxShadow: "none" }}>
            <div className="CardHeader">
              <h2>Preview</h2>
              <p>{selected ? selected.name : "Select a file to preview"}</p>
            </div>
            <div className="CardBody">
              {!selected ? <div className="Status">No selection.</div> : null}
              {selected && selected.type === "file" && isImage(selected.name) ? (
                <img className="PreviewImage" src={apiFsFileUrl(selected.path)} alt={selected.name} />
              ) : null}
              {selected && selected.type === "file" && textPreview ? (
                <textarea
                  className="Textarea"
                  readOnly
                  value={textPreview.text + (textPreview.truncated ? "\n\n[truncated]" : "")}
                />
              ) : null}
              {selected && selected.type === "file" && !isImage(selected.name) && !isText(selected.name) ? (
                <div className="Status">Preview not available for this file type.</div>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
