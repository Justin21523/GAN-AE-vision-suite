import { useMemo, useState } from "react";
import { apiJobsStart } from "../api";
import { DynamicJobForm } from "../components/DynamicJobForm";

function groupLabel(jobType) {
  if (jobType.startsWith("train_")) return "Training";
  if (jobType.includes("data") || jobType.startsWith("prepare")) return "Data";
  if (jobType.startsWith("eval") || jobType.startsWith("sample")) return "Eval/Sampling";
  return "Other";
}

export function CommandCenterPage({ caps, onJob }) {
  const jobs = useMemo(() => caps?.jobs || [], [caps]);
  const groups = useMemo(() => {
    const m = new Map();
    for (const j of jobs) {
      const g = groupLabel(j.type);
      if (!m.has(g)) m.set(g, []);
      m.get(g).push(j);
    }
    for (const [k, arr] of m.entries()) {
      arr.sort((a, b) => String(a.label).localeCompare(String(b.label)));
      m.set(k, arr);
    }
    return Array.from(m.entries());
  }, [jobs]);

  const [selectedType, setSelectedType] = useState(jobs[0]?.type || "");
  const [status, setStatus] = useState("");

  const selected = useMemo(() => jobs.find((j) => j.type === selectedType), [jobs, selectedType]);

  const run = async (args) => {
    if (!selected) return;
    setStatus("Starting...");
    try {
      const job = await apiJobsStart(selected.type, args);
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
          <h2>Command Center</h2>
          <p>All backend functions exposed as UI forms (from `/api/capabilities`).</p>
        </div>
        <div className="CardBody">
          <div className="Status" style={{ marginBottom: 10 }}>
            Select a command, fill parameters, then run. Logs/artifacts show in the Jobs panel.
          </div>

          {groups.map(([g, arr]) => (
            <div key={g} style={{ marginBottom: 12 }}>
              <div className="Status" style={{ fontWeight: 700, marginBottom: 6 }}>
                {g}
              </div>
              <div className="Row">
                {arr.map((j) => (
                  <button key={j.type} className="ButtonGhost" onClick={() => setSelectedType(j.type)} data-active={j.type === selectedType}>
                    {j.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="Card">
        <div className="CardHeader">
          <h2>{selected ? selected.label : "Select a command"}</h2>
          <p>{selected ? selected.description : ""}</p>
        </div>
        <div className="CardBody">
          {selected ? (
            <DynamicJobForm job={selected} onSubmit={run} submitLabel="Run" />
          ) : (
            <div className="Status">No command selected.</div>
          )}
          {status ? <div className="Status" style={{ marginTop: 10 }}>{status}</div> : null}
        </div>
      </div>
    </div>
  );
}
