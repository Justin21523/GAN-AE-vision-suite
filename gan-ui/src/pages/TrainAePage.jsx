import { useMemo, useState } from "react";
import { apiJobsStart } from "../api";
import { DynamicJobForm } from "../components/DynamicJobForm";

export function TrainAePage({ onJob, caps }) {
  const [status, setStatus] = useState("");

  const spec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "train_ae"), [caps]);

  const onStart = async (args) => {
    setStatus("Starting job...");
    try {
      const job = await apiJobsStart("train_ae", args);
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
          <h2>Train AE/VAE (CLI)</h2>
          <p>Runs `python -m src.scripts.train_ae` as a background job.</p>
        </div>
        <div className="CardBody">
          {spec ? (
            <DynamicJobForm job={spec} onSubmit={onStart} submitLabel="Start training" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
          <div className="Status" style={{ marginTop: 8 }}>
            {status}
          </div>
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Notes</h2>
          <p>AE/VAE outputs include recon grids under `save.out_dir` plus checkpoints under `training.save_dir`.</p>
        </div>
        <div className="CardBody">
          <div className="Status">
            Tip: run a `Data Report` first to ensure normalization and splits look correct.
          </div>
        </div>
      </div>
    </div>
  );
}
