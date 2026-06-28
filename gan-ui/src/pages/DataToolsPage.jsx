import { useMemo, useState } from "react";
import { apiJobsStart } from "../api";
import { DynamicJobForm } from "../components/DynamicJobForm";

export function DataToolsPage({ onJob, caps }) {
  const [status, setStatus] = useState("");

  const reportSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "data_report"), [caps]);
  const validateSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "validate_data"), [caps]);
  const prepareSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "prepare_data"), [caps]);
  const demoSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "prepare_demo"), [caps]);

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
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="Card">
        <div className="CardHeader">
          <h2>Data Report</h2>
          <p>Runs `python -m src.scripts.data_report` and writes `report.json` + sample grids.</p>
        </div>
        <div className="CardBody">
          {reportSpec ? (
            <DynamicJobForm job={reportSpec} onSubmit={(args) => start("data_report", args)} submitLabel="Run report" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
        </div>
      </div>

      <div className="Grid">
        <div className="Card">
          <div className="CardHeader">
            <h2>Prepare Data (Check)</h2>
            <p>Validates local dataset folder layout (no downloads).</p>
          </div>
          <div className="CardBody">
          {prepareSpec ? (
            <DynamicJobForm job={prepareSpec} onSubmit={(args) => start("prepare_data", args)} submitLabel="Check" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
          </div>
        </div>

        <div className="Card">
          <div className="CardHeader">
            <h2>Create Demo ImageFolder</h2>
            <p>Creates a tiny random dataset for fast smoke testing.</p>
          </div>
          <div className="CardBody">
          {demoSpec ? (
            <DynamicJobForm job={demoSpec} onSubmit={(args) => start("prepare_demo", args)} submitLabel="Create" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
          </div>
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Validate Data</h2>
          <p>Runs `python -m src.validate_data` and writes sample grids + PSNR/SSIM check.</p>
        </div>
        <div className="CardBody">
          {validateSpec ? (
            <DynamicJobForm job={validateSpec} onSubmit={(args) => start("validate_data", args)} submitLabel="Run validation" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
        </div>
      </div>

      {status ? <div className="Status">{status}</div> : null}
    </div>
  );
}
