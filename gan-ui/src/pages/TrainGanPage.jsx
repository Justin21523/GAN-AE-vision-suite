import { useMemo, useState } from "react";
import { apiJobsStart } from "../api";
import { DynamicJobForm } from "../components/DynamicJobForm";

export function TrainGanPage({ onJob, caps }) {
  const [status, setStatus] = useState("");

  const spec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "train_gan"), [caps]);

  const onStart = async (args) => {
    setStatus("Starting job...");
    try {
      const job = await apiJobsStart("train_gan", args);
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
          <h2>Train GAN (CLI)</h2>
          <p>Runs `python -m src.scripts.train_gan` as a background job.</p>
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
          <h2>Outputs</h2>
          <p>Look under `training.logdir` for `metrics.jsonl`, `meta.json`, samples and checkpoints.</p>
        </div>
        <div className="CardBody">
          <div className="Status">
            Tip: open the Jobs panel to see live logs and the exact command line.
          </div>
        </div>
      </div>
    </div>
  );
}
