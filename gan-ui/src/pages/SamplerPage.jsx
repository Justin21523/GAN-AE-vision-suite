import { useState } from "react";
import { apiGenerate, apiLoad } from "../api";
import { Field } from "../components/Field";

export function SamplerPage() {
  const [ckpt, setCkpt] = useState("logs/stage3_wgangp/ckpt_epoch1.pt");
  const [device, setDevice] = useState("cuda");
  const [n, setN] = useState(64);
  const [nrow, setNrow] = useState(8);
  const [seed, setSeed] = useState(42);
  const [useEma, setUseEma] = useState(false);
  const [imgUrl, setImgUrl] = useState(null);
  const [status, setStatus] = useState("");

  const onLoad = async () => {
    setStatus("Loading checkpoint...");
    try {
      const info = await apiLoad(ckpt, device);
      setStatus(
        `Loaded: device=${info.device}, img_size=${info.img_size}, has_ema_shadow=${String(info.has_ema_shadow)}`
      );
    } catch (e) {
      setStatus(String(e));
    }
  };

  const onGenerate = async () => {
    setStatus("Generating...");
    try {
      const url = await apiGenerate({ n, nrow, seed, use_ema: useEma });
      setImgUrl(url);
      setStatus("Done.");
    } catch (e) {
      setStatus(String(e));
    }
  };

  return (
    <div className="Grid">
      <div className="Card">
        <div className="CardHeader">
          <h2>GAN Sampler (API)</h2>
          <p>Load a checkpoint server-side, then generate grids on demand.</p>
        </div>
        <div className="CardBody">
          <div className="Row">
            <Field label="Checkpoint path">
              <input className="Input" value={ckpt} onChange={(e) => setCkpt(e.target.value)} />
            </Field>
            <Field label="Device">
              <select className="Select" value={device} onChange={(e) => setDevice(e.target.value)}>
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
                <option value="cuda:0">cuda:0</option>
              </select>
            </Field>
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            <button className="Button" onClick={onLoad}>
              Load
            </button>
            <span className="Status">{status}</span>
          </div>

          <div className="Row" style={{ marginTop: 14 }}>
            <Field label="n">
              <input className="Input" type="number" value={n} onChange={(e) => setN(parseInt(e.target.value || "0"))} />
            </Field>
            <Field label="nrow">
              <input
                className="Input"
                type="number"
                value={nrow}
                onChange={(e) => setNrow(parseInt(e.target.value || "0"))}
              />
            </Field>
            <Field label="seed">
              <input
                className="Input"
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value || "0"))}
              />
            </Field>
            <div className="Field" style={{ minWidth: 140, flex: 0 }}>
              <label>EMA</label>
              <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input type="checkbox" checked={useEma} onChange={(e) => setUseEma(e.target.checked)} />
                Use EMA (if available)
              </label>
            </div>
          </div>

          <div className="Row" style={{ marginTop: 10 }}>
            <button className="ButtonSecondary" onClick={onGenerate}>
              Generate
            </button>
            <span className="Status">Tip: use `Sample GAN (CLI)` if you want a file written to disk.</span>
          </div>
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Preview</h2>
          <p>Generated PNG grid.</p>
        </div>
        <div className="CardBody">
          {imgUrl ? <img className="PreviewImage" src={imgUrl} alt="grid" /> : <div className="Status">No image yet.</div>}
        </div>
      </div>
    </div>
  );
}
