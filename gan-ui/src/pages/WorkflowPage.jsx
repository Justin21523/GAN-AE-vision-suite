import { useMemo, useState } from "react";
import { apiFsFileUrl, apiFsRead, apiJobsStart } from "../api";
import { DynamicJobForm } from "../components/DynamicJobForm";

function pillClass(ok, warn) {
  if (ok) return "Pill PillGood";
  if (warn) return "Pill PillWarn";
  return "Pill PillBad";
}

export function WorkflowPage({ caps, onJob }) {
  const reportSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "data_report"), [caps]);
  const trainSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "train_gan"), [caps]);
  const trainAeSpec = useMemo(() => (caps?.jobs || []).find((j) => j.type === "train_ae"), [caps]);

  const [status, setStatus] = useState("");
  const [reportArgs, setReportArgs] = useState({ config: "configs/dataset_celeba.yaml", out: "./outputs/data_report" });
  const [reportSummary, setReportSummary] = useState(null);
  const [mode, setMode] = useState("gan"); // gan|ae

  const runReport = async (args) => {
    setStatus("Starting data report...");
    setReportArgs(args);
    setReportSummary(null);
    try {
      const job = await apiJobsStart("data_report", args);
      setStatus(`Data report started: ${job.id}`);
      onJob(job.id);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const loadReportSummary = async () => {
    const outDir = reportArgs.out || "./outputs/data_report";
    setStatus("Loading report.json...");
    try {
      const r = await apiFsRead(`${outDir}/report.json`, 400000);
      const obj = JSON.parse(r.text);
      setReportSummary(obj);
      setStatus("Report loaded.");
    } catch (e) {
      setStatus(String(e));
    }
  };

  const runTrain = async (args) => {
    setStatus("Starting training...");
    try {
      const job = await apiJobsStart(mode === "gan" ? "train_gan" : "train_ae", args);
      setStatus(`Training started: ${job.id}`);
      onJob(job.id);
    } catch (e) {
      setStatus(String(e));
    }
  };

  const outDir = reportArgs.out || "./outputs/data_report";
  const disk = reportSummary?.disk_scan;
  const trainStats = reportSummary?.train_sample?.tensor_stats;
  const badFiles = disk?.bad_files ?? 0;
  const mean0 = trainStats?.mean?.[0];
  const std0 = trainStats?.std?.[0];
  const meanOk = typeof mean0 === "number" ? Math.abs(mean0) < 0.2 : false;
  const stdOk = typeof std0 === "number" ? std0 > 0.2 && std0 < 1.2 : false;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="Card">
        <div className="CardHeader">
          <h2>Workflow: Data → Train (GAN)</h2>
          <p>Recommended local flow: run Data Report first, verify normalization/splits, then start training.</p>
        </div>
        <div className="CardBody">
          <div className="Row" style={{ flexWrap: "wrap" }}>
            <span className="Status">Mode:</span>
            <button className="ButtonGhost" data-active={mode === "gan"} onClick={() => setMode("gan")}>
              GAN
            </button>
            <button className="ButtonGhost" data-active={mode === "ae"} onClick={() => setMode("ae")}>
              AE/VAE
            </button>
          </div>
          {status ? <div className="Status">{status}</div> : null}
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Step 1 — Data Report</h2>
          <p>Generates `report.json` + sample grids under the output directory.</p>
        </div>
        <div className="CardBody">
          {reportSpec ? (
            <DynamicJobForm
              job={reportSpec}
              initial={reportArgs}
              submitLabel="Run Data Report"
              onSubmit={runReport}
              extraButtons={
                <button className="ButtonGhost" onClick={loadReportSummary}>
                  Load report.json
                </button>
              }
            />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}

          <div className="Row" style={{ marginTop: 10, flexWrap: "wrap" }}>
            <a className="ButtonGhost" href={apiFsFileUrl(`${outDir}/report.json`)} target="_blank" rel="noreferrer">
              Open report.json
            </a>
            <a className="ButtonGhost" href={apiFsFileUrl(`${outDir}/samples_train.png`)} target="_blank" rel="noreferrer">
              samples_train.png
            </a>
            <a className="ButtonGhost" href={apiFsFileUrl(`${outDir}/samples_val.png`)} target="_blank" rel="noreferrer">
              samples_val.png
            </a>
          </div>

          {reportSummary ? (
            <div style={{ marginTop: 14 }}>
              <div className="Row" style={{ gap: 10, flexWrap: "wrap" }}>
                {disk ? (
                  <span className={pillClass(badFiles === 0, badFiles > 0 && badFiles < 5)}>
                    disk_scan: ok={disk.ok_files} bad={disk.bad_files} total={disk.total_files}
                  </span>
                ) : (
                  <span className="Pill PillWarn">disk_scan: n/a</span>
                )}
                {trainStats?.mean ? (
                  <span className={pillClass(meanOk, typeof mean0 === "number" && Math.abs(mean0) < 0.5)}>
                    mean={trainStats.mean.join(", ")}
                  </span>
                ) : null}
                {trainStats?.std ? (
                  <span className={pillClass(stdOk, typeof std0 === "number" && std0 > 0.1)}>
                    std={trainStats.std.join(", ")}
                  </span>
                ) : null}
              </div>
              <div style={{ marginTop: 10 }}>
                <img className="PreviewImage" src={apiFsFileUrl(`${outDir}/samples_train.png`)} alt="samples_train" />
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>Step 2 — Train</h2>
          <p>Starts training with a run name so outputs are isolated under the configured log directory.</p>
        </div>
        <div className="CardBody">
          {mode === "gan" ? (
            trainSpec ? (
              <DynamicJobForm job={trainSpec} onSubmit={runTrain} submitLabel="Start GAN Training" />
            ) : (
              <div className="Status">Loading capabilities...</div>
            )
          ) : trainAeSpec ? (
            <DynamicJobForm job={trainAeSpec} onSubmit={(args) => runTrain(args)} submitLabel="Start AE/VAE Training" />
          ) : (
            <div className="Status">Loading capabilities...</div>
          )}
        </div>
      </div>
    </div>
  );
}
