import { useState } from "react";
import { apiLoad, apiGenerate } from "./api";

export default function App() {
  const [ckpt, setCkpt] = useState("logs/stage3_wgangp/ckpt_epoch1.pt");
  const [n, setN] = useState(64);
  const [nrow, setNrow] = useState(8);
  const [seed, setSeed] = useState(42);
  const [useEma, setUseEma] = useState(false);
  const [imgUrl, setImgUrl] = useState(null);
  const [status, setStatus] = useState("");

  const onLoad = async () => {
    setStatus("Loading...");
    try {
      const info = await apiLoad(ckpt, "cuda");
      setStatus(`Loaded on ${info.device}, img_size=${info.img_size}`);
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
    <div style={{ maxWidth: 960, margin: "24px auto", fontFamily: "Inter,system-ui,sans-serif" }}>
      <h2>GAN Sampler (FastAPI ↔ React)</h2>
      <div style={{ display: "flex", gap: 8 }}>
        <input style={{ flex: 1 }} value={ckpt} onChange={(e) => setCkpt(e.target.value)} />
        <button onClick={onLoad}>Load</button>
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 12, alignItems: "center" }}>
        <label>n</label>
        <input type="number" value={n} onChange={(e) => setN(parseInt(e.target.value || "0"))} />
        <label>nrow</label>
        <input type="number" value={nrow} onChange={(e) => setNrow(parseInt(e.target.value || "0"))} />
        <label>seed</label>
        <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value || "0"))} />
        <label>
          <input type="checkbox" checked={useEma} onChange={(e) => setUseEma(e.target.checked)} />
          Use EMA
        </label>
        <button onClick={onGenerate}>Generate</button>
      </div>

      <p style={{ color: "#666" }}>{status}</p>

      <div style={{ marginTop: 16 }}>
        {imgUrl ? <img src={imgUrl} alt="grid" style={{ maxWidth: "100%", borderRadius: 8 }} /> : <em>No image yet</em>}
      </div>
    </div>
  );
}
