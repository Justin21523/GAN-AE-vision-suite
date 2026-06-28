import { useMemo, useState } from "react";
import { apiJobsStart } from "../api";
import { DynamicJobForm } from "../components/DynamicJobForm";

export function EvalPage({ onJob, caps }) {
  const [status, setStatus] = useState("");
  const sampleSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "sample_gan"), [caps]);
  const evalSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "eval_fid"), [caps]);
  const evalRunSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "eval_gan_pipeline"), [caps]);

  const start = async (type, args) => {
    setStatus("Starting...");
    try {
      const job = await apiJobsStart(type, args);
      setStatus(`Started: ${job.id}`);
      onJob(job.id);
    } catch (e) {
      setStatus(String(e));
    }
  };

  return (
    <div className="Grid">
      <div className="Card">
        <div className="CardHeader">
          <h2>Sample GAN (CLI)</h2>
          <p>Runs `python -m src.scripts.sample_gan` to write a grid image to disk.</p>
        </div>
        <div className="CardBody">
          {sampleSpec ? (
            <DynamicJobForm job={sampleSpec} onSubmit={(args) => start("sample_gan", args)} submitLabel="Run sampling" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Eval FID/KID (CLI)</h2>
          <p>Runs `python -m src.scripts.eval_fid` against a directory of generated images.</p>
        </div>
        <div className="CardBody">
          {evalSpec ? (
            <DynamicJobForm job={evalSpec} onSubmit={(args) => start("eval_fid", args)} submitLabel="Run eval" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Eval Run (Generate → FID/KID)</h2>
          <p>Evaluates an existing run directory and appends scores to its metrics.jsonl.</p>
        </div>
        <div className="CardBody">
          {evalRunSpec ? (
            <DynamicJobForm job={evalRunSpec} onSubmit={(args) => start("eval_gan_pipeline", args)} submitLabel="Run evaluation" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
        </div>
      </div>

      {status ? <div className="Status">{status}</div> : null}
    </div>
  );
}
