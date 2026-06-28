import { useEffect, useMemo, useState } from "react";
import {
  apiConfigApplyOverlayPath,
  apiConfigApplyOverrides,
  apiConfigOverlaysList,
  apiConfigValidate,
  apiFsList,
  apiFsMkdir,
  apiFsRead,
  apiFsWrite,
  apiJobsStart,
} from "../api";
import { FileBrowser } from "../components/FileBrowser";

function isYaml(name) {
  const s = String(name || "").toLowerCase();
  return s.endsWith(".yaml") || s.endsWith(".yml");
}

export function ConfigEditorPage({ onJob, caps }) {
  const [dir, setDir] = useState("./configs");
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState("");
  const [text, setText] = useState("");
  const [status, setStatus] = useState("");
  const [outPath, setOutPath] = useState("./.ai_cache/configs/local.yaml");
  const [showBrowser, setShowBrowser] = useState(false);
  const [kind, setKind] = useState("auto");
  const [validateResult, setValidateResult] = useState(null);
  const [ovDevice, setOvDevice] = useState("");
  const [ovSeed, setOvSeed] = useState("");
  const [ovBatchSize, setOvBatchSize] = useState("");
  const [ovImgSize, setOvImgSize] = useState("");
  const [ovEpochs, setOvEpochs] = useState("");
  const [ovLr, setOvLr] = useState("");
  const [ovLrG, setOvLrG] = useState("");
  const [ovLrD, setOvLrD] = useState("");
  const [ovNCritic, setOvNCritic] = useState("");
  const [ovLambdaGp, setOvLambdaGp] = useState("");

  const trainGanSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "train_gan"), [caps]);
  const trainAeSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "train_ae"), [caps]);

  const [trainDevice, setTrainDevice] = useState("cuda");
  const [trainRunName, setTrainRunName] = useState("exp01");
  const [trainEpochs, setTrainEpochs] = useState("");
  const [reportOut, setReportOut] = useState("./outputs/data_report");
  const [overlayDir, setOverlayDir] = useState("./.ai_cache/configs/overrides");
  const [overlayName, setOverlayName] = useState("exp_override");
  const [overlayList, setOverlayList] = useState([]);
  const [selectedOverlay, setSelectedOverlay] = useState("");

  const refresh = async () => {
    setStatus("");
    try {
      const r = await apiFsList(dir);
      const ys = (r.entries || []).filter((e) => e.type === "file" && isYaml(e.name));
      setFiles(ys);
      if (!selected && ys.length) setSelected(ys[0].path);
    } catch (e) {
      setStatus(String(e));
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dir]);

  useEffect(() => {
    (async () => {
      if (!selected) return;
      setStatus("Loading...");
      try {
        const r = await apiFsRead(selected, 600000);
        setText(r.text);
        setStatus("Loaded.");
      } catch (e) {
        setStatus(String(e));
      }
    })();
  }, [selected]);

  const saveAs = async () => {
    setStatus("Saving...");
    try {
      await apiFsMkdir("./.ai_cache/configs");
      await apiFsWrite(outPath, text, true);
      setStatus(`Saved: ${outPath}`);
      return true;
    } catch (e) {
      setStatus(String(e));
      return false;
    }
  };

  const validate = async () => {
    setStatus("Validating...");
    setValidateResult(null);
    try {
      const r = await apiConfigValidate(kind, text);
      setValidateResult(r);
      setStatus(r.warnings?.length ? `OK (warnings: ${r.warnings.length})` : "OK");
      return r;
    } catch (e) {
      setStatus(String(e));
      return null;
    }
  };

  const applyOverrides = async () => {
    setStatus("Applying overrides...");
    try {
      const overrides = [];
      const pushInt = (path, v) => {
        if (!v) return;
        overrides.push({ path, value: String(v), type: "int" });
      };
      const pushFloat = (path, v) => {
        if (!v) return;
        overrides.push({ path, value: String(v), type: "float" });
      };
      const pushString = (path, v) => {
        if (!v) return;
        overrides.push({ path, value: String(v), type: "string" });
      };

      pushString("device", ovDevice);
      pushInt("seed", ovSeed);
      pushInt("data.batch_size", ovBatchSize);
      pushInt("data.img_size", ovImgSize);
      pushInt("data.image_size", ovImgSize);

      // AE-style
      pushInt("training.epochs", ovEpochs);
      pushFloat("training.lr", ovLr);

      // GAN-style
      pushInt("training.epochs", ovEpochs);
      pushFloat("training.lr_g", ovLrG);
      pushFloat("training.lr_d", ovLrD);
      pushInt("training.n_critic", ovNCritic);
      pushFloat("training.lambda_gp", ovLambdaGp);

      if (!overrides.length) {
        setStatus("No overrides set.");
        return;
      }
      const r = await apiConfigApplyOverrides(text, overrides);
      setText(r.patched);
      setStatus(`Applied ${r.applied.length} overrides.`);
      // Re-validate to refresh kind/warnings
      const vr = await apiConfigValidate("auto", r.patched);
      setValidateResult(vr);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const refreshOverlays = async () => {
    try {
      const r = await apiConfigOverlaysList(overlayDir);
      setOverlayList(r.overlays || []);
      if (!selectedOverlay && (r.overlays || []).length) {
        setSelectedOverlay(r.overlays[0].path);
      }
    } catch (e) {
      setStatus(String(e));
    }
  };

  useEffect(() => {
    refreshOverlays();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlayDir]);

  const buildOverlayObject = () => {
    const o = {};
    const set = (path, value) => {
      const parts = String(path).split(".").filter(Boolean);
      let cur = o;
      for (const p of parts.slice(0, -1)) {
        if (!cur[p] || typeof cur[p] !== "object") cur[p] = {};
        cur = cur[p];
      }
      cur[parts[parts.length - 1]] = value;
    };
    if (ovDevice) set("device", String(ovDevice));
    if (ovSeed) set("seed", Number(ovSeed));
    if (ovBatchSize) set("data.batch_size", Number(ovBatchSize));
    if (ovImgSize) set("data.img_size", Number(ovImgSize));
    if (ovEpochs) set("training.epochs", Number(ovEpochs));
    if (ovLr) set("training.lr", String(ovLr));
    if (ovLrG) set("training.lr_g", String(ovLrG));
    if (ovLrD) set("training.lr_d", String(ovLrD));
    if (ovNCritic) set("training.n_critic", Number(ovNCritic));
    if (ovLambdaGp) set("training.lambda_gp", String(ovLambdaGp));
    return o;
  };

  const saveOverlay = async () => {
    setStatus("Saving overlay...");
    try {
      await apiFsMkdir(overlayDir);
      const obj = buildOverlayObject();
      const out = `${overlayDir}/${overlayName}.yaml`;
      await apiFsWrite(out, JSON.stringify(obj, null, 2) + "\n", true);
      setStatus(`Overlay saved: ${out}`);
      await refreshOverlays();
      setSelectedOverlay(out);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const applyOverlay = async () => {
    if (!selectedOverlay) {
      setStatus("No overlay selected.");
      return;
    }
    setStatus("Applying overlay...");
    try {
      const r = await apiConfigApplyOverlayPath(text, selectedOverlay);
      setText(r.patched);
      const vr = await apiConfigValidate("auto", r.patched);
      setValidateResult(vr);
      setStatus(`Overlay applied: ${selectedOverlay}`);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const saveAndRun = async (type) => {
    const ok = await saveAs();
    if (!ok) {
      return;
    }
    setStatus(`Starting ${type}...`);
    try {
      const args = {
        config: outPath,
        device: trainDevice,
        run_name: trainRunName || undefined,
      };
      if (type === "train_ae" && trainEpochs) {
        args.epochs = parseInt(trainEpochs);
      }
      const job = await apiJobsStart(type, args);
      setStatus(`Started: ${job.id}`);
      onJob(job.id);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const saveAndReport = async () => {
    const ok = await saveAs();
    if (!ok) return;
    setStatus("Starting data_report...");
    try {
      const job = await apiJobsStart("data_report", { config: outPath, out: reportOut });
      setStatus(`Started: ${job.id}`);
      onJob(job.id);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const saveAndTrainAuto = async () => {
    const ok = await saveAs();
    if (!ok) return;
    const r = await apiConfigValidate("auto", text).catch((e) => {
      setStatus(String(e));
      return null;
    });
    if (!r) return;
    setValidateResult(r);
    const target = r.kind === "gan" ? "train_gan" : "train_ae";
    setStatus(`Starting ${target}...`);
    try {
      const args = {
        config: outPath,
        device: trainDevice,
        run_name: trainRunName || undefined,
      };
      if (target === "train_ae" && trainEpochs) {
        args.epochs = parseInt(trainEpochs);
      }
      const job = await apiJobsStart(target, args);
      setStatus(`Started: ${job.id}`);
      onJob(job.id);
    } catch (e) {
      setStatus(String(e));
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="Card">
        <div className="CardHeader">
          <h2>Config Editor</h2>
          <p>Edit YAML configs and save to `./.ai_cache/configs/` for local experiments.</p>
        </div>
        <div className="CardBody">
          <div className="Row">
            <div className="Field">
              <label>Directory</label>
              <input className="Input" value={dir} onChange={(e) => setDir(e.target.value)} />
            </div>
            <button className="ButtonGhost" onClick={refresh}>
              Refresh
            </button>
            <button className="ButtonGhost" onClick={() => setShowBrowser((v) => !v)}>
              {showBrowser ? "Hide Browser" : "Browse Files"}
            </button>
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            <div className="Field">
              <label>YAML file</label>
              <select className="Select" value={selected} onChange={(e) => setSelected(e.target.value)}>
                {files.map((f) => (
                  <option key={f.path} value={f.path}>
                    {f.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="Field">
              <label>Save as</label>
              <input className="Input" value={outPath} onChange={(e) => setOutPath(e.target.value)} />
            </div>
            <div className="Field" style={{ minWidth: 180, flex: 0 }}>
              <label>Validate as</label>
              <select className="Select" value={kind} onChange={(e) => setKind(e.target.value)}>
                <option value="auto">auto</option>
                <option value="gan">gan</option>
                <option value="ae">ae</option>
              </select>
            </div>
          </div>

          <div style={{ marginTop: 10 }}>
            <textarea className="Textarea" value={text} onChange={(e) => setText(e.target.value)} />
          </div>

          <div className="Row" style={{ marginTop: 10, flexWrap: "wrap" }}>
            <button className="ButtonGhost" onClick={validate}>
              Validate
            </button>
            <button className="Button" onClick={saveAs}>
              Save (overwrite)
            </button>
            <button className="ButtonSecondary" onClick={saveAndReport}>
              Save & Run Data Report
            </button>
            <button className="Button" onClick={saveAndTrainAuto}>
              Save & Train (auto)
            </button>
            <button className="Button" onClick={() => saveAndRun("train_gan")} disabled={!trainGanSpec}>
              Save & Train GAN
            </button>
            <button className="Button" onClick={() => saveAndRun("train_ae")} disabled={!trainAeSpec}>
              Save & Train AE/VAE
            </button>
            <span className="Status">{status}</span>
          </div>

          {validateResult ? (
            <div style={{ marginTop: 10 }}>
              <div className="Row" style={{ flexWrap: "wrap" }}>
                <span className="Pill">kind={validateResult.kind}</span>
                <span className="Pill">keys={validateResult.top_level_keys.join(", ")}</span>
                {(validateResult.warnings || []).map((w) => (
                  <span key={w} className="Pill">
                    warn: {w}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>One-Click Training Settings</h2>
          <p>Used by “Save & Train …” buttons.</p>
        </div>
        <div className="CardBody">
          <div className="Row">
            <div className="Field">
              <label>Device</label>
              <select className="Select" value={trainDevice} onChange={(e) => setTrainDevice(e.target.value)}>
                <option value="cuda">cuda</option>
                <option value="cuda:0">cuda:0</option>
                <option value="cpu">cpu</option>
              </select>
            </div>
            <div className="Field">
              <label>Run name</label>
              <input className="Input" value={trainRunName} onChange={(e) => setTrainRunName(e.target.value)} />
            </div>
            <div className="Field">
              <label>AE epochs override (optional)</label>
              <input className="Input" value={trainEpochs} onChange={(e) => setTrainEpochs(e.target.value)} placeholder="e.g. 10" />
            </div>
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            <div className="Field">
              <label>Data report out dir</label>
              <input className="Input" value={reportOut} onChange={(e) => setReportOut(e.target.value)} />
            </div>
          </div>
        </div>
      </div>

      {showBrowser ? <FileBrowser initialPath="configs" title="Browse Configs" /> : null}

      <div className="Card">
        <div className="CardHeader">
          <h2>Overrides (no hand-edit)</h2>
          <p>Fill common knobs and click “Apply overrides to editor”. You can also save them as a reusable overlay.</p>
        </div>
        <div className="CardBody">
          <div className="Row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
            <div className="Field">
              <label>Config kind</label>
              <span className={`Pill ${validateResult?.kind ? "PillGood" : ""}`}>
                {validateResult?.kind || "unknown"}
                {validateResult?.model_type ? ` · model=${validateResult.model_type}` : ""}
              </span>
            </div>
            {validateResult?.is_wgan_gp ? <span className="Pill PillGood">WGAN-GP</span> : null}
          </div>

          <div className="Row">
            <div className="Field">
              <label>device (top-level)</label>
              <input className="Input" value={ovDevice} onChange={(e) => setOvDevice(e.target.value)} placeholder="cuda / cpu" />
            </div>
            <div className="Field">
              <label>seed</label>
              <input className="Input" value={ovSeed} onChange={(e) => setOvSeed(e.target.value)} placeholder="42" />
            </div>
            <div className="Field">
              <label>data.batch_size</label>
              <input className="Input" value={ovBatchSize} onChange={(e) => setOvBatchSize(e.target.value)} placeholder="64" />
            </div>
            <div className="Field">
              <label>data.img_size</label>
              <input className="Input" value={ovImgSize} onChange={(e) => setOvImgSize(e.target.value)} placeholder="128" />
            </div>
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            <div className="Field">
              <label>training.epochs</label>
              <input className="Input" value={ovEpochs} onChange={(e) => setOvEpochs(e.target.value)} placeholder="5" />
            </div>
            {validateResult?.kind !== "gan" ? (
              <div className="Field">
              <label>training.lr (AE/VAE)</label>
              <input className="Input" value={ovLr} onChange={(e) => setOvLr(e.target.value)} placeholder="1e-3" />
              </div>
            ) : null}
            {validateResult?.kind === "gan" ? (
              <div className="Field">
              <label>training.lr_g (GAN)</label>
              <input className="Input" value={ovLrG} onChange={(e) => setOvLrG(e.target.value)} placeholder="1e-4" />
              </div>
            ) : null}
            {validateResult?.kind === "gan" ? (
              <div className="Field">
              <label>training.lr_d (GAN)</label>
              <input className="Input" value={ovLrD} onChange={(e) => setOvLrD(e.target.value)} placeholder="1e-4" />
              </div>
            ) : null}
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            {validateResult?.is_wgan_gp ? (
              <>
                <div className="Field">
                  <label>training.n_critic (WGAN-GP)</label>
                  <input className="Input" value={ovNCritic} onChange={(e) => setOvNCritic(e.target.value)} placeholder="5" />
                </div>
                <div className="Field">
                  <label>training.lambda_gp (WGAN-GP)</label>
                  <input className="Input" value={ovLambdaGp} onChange={(e) => setOvLambdaGp(e.target.value)} placeholder="10.0" />
                </div>
              </>
            ) : null}
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            <button className="ButtonGhost" onClick={applyOverrides}>
              Apply overrides to editor
            </button>
          </div>

          <div className="Row" style={{ marginTop: 14 }}>
            <div className="Field">
              <label>Overlay dir</label>
              <input className="Input" value={overlayDir} onChange={(e) => setOverlayDir(e.target.value)} />
            </div>
            <div className="Field">
              <label>Overlay name</label>
              <input className="Input" value={overlayName} onChange={(e) => setOverlayName(e.target.value)} />
            </div>
            <button className="Button" onClick={saveOverlay}>
              Save overlay
            </button>
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            <div className="Field">
              <label>Saved overlays</label>
              <select className="Select" value={selectedOverlay} onChange={(e) => setSelectedOverlay(e.target.value)}>
                {(overlayList || []).map((o) => (
                  <option key={o.path} value={o.path}>
                    {o.name}
                  </option>
                ))}
              </select>
            </div>
            <button className="ButtonGhost" onClick={refreshOverlays}>
              Refresh
            </button>
            <button className="ButtonSecondary" onClick={applyOverlay} disabled={!selectedOverlay}>
              Apply overlay
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
