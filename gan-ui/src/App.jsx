import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import { apiCapabilities, apiJobsList } from "./api";
import { FileBrowser } from "./components/FileBrowser";
import { JobPanel } from "./components/JobPanel";
import { DataToolsPage } from "./pages/DataToolsPage";
import { EvalPage } from "./pages/EvalPage";
import { SamplerPage } from "./pages/SamplerPage";
import { TrainAePage } from "./pages/TrainAePage";
import { TrainGanPage } from "./pages/TrainGanPage";
import { RunsPage } from "./pages/RunsPage";
import { WorkflowPage } from "./pages/WorkflowPage";
import { CommandCenterPage } from "./pages/CommandCenterPage";
import { ConfigEditorPage } from "./pages/ConfigEditorPage";
import { OverviewPage } from "./pages/OverviewPage";

export default function App() {
  const tabs = useMemo(
    () => [
      { key: "overview", label: "Overview" },
      { key: "sampler", label: "Sampler" },
      { key: "workflow", label: "Workflow" },
      { key: "commands", label: "Commands" },
      { key: "train_gan", label: "Train GAN" },
      { key: "train_ae", label: "Train AE/VAE" },
      { key: "data", label: "Data Tools" },
      { key: "eval", label: "Eval" },
      { key: "runs", label: "Runs" },
      { key: "configs", label: "Configs" },
      { key: "files", label: "Files" },
      { key: "jobs", label: "Jobs" },
    ],
    []
  );

  const [tab, setTab] = useState("overview");
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [jobsError, setJobsError] = useState("");
  const [caps, setCaps] = useState(null);
  const [capsError, setCapsError] = useState("");

  const refreshJobs = useCallback(async () => {
    setJobsError("");
    try {
      const list = await apiJobsList();
      setJobs(list.slice().reverse());
    } catch (e) {
      setJobsError(String(e));
    }
  }, []);

  useEffect(() => {
    refreshJobs();
    const t = setInterval(() => refreshJobs(), 2000);
    return () => clearInterval(t);
  }, [refreshJobs]);

  useEffect(() => {
    (async () => {
      setCapsError("");
      try {
        const c = await apiCapabilities();
        setCaps(c);
      } catch (e) {
        setCapsError(String(e));
      }
    })();
  }, []);

  return (
    <div className="AppShell">
      <div className="TopBar">
        <div className="TopBarInner">
          <div className="Brand">
            <div className="BrandMark" />
            <div className="BrandTitle">
              <strong>GAN-AE Vision Suite</strong>
              <span>Local UI for CLI workflows</span>
            </div>
          </div>
          <div className="Nav">
            {tabs.map((t) => (
              <button key={t.key} data-active={tab === t.key} onClick={() => setTab(t.key)}>
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="Container">
        {capsError && tab !== "overview" ? <div className="Status">Capabilities error: {capsError}</div> : null}
        {tab === "overview" ? <OverviewPage apiOnline={!capsError && Boolean(caps)} /> : null}
        {tab === "sampler" ? <SamplerPage /> : null}
        {tab === "workflow" ? <WorkflowPage caps={caps} onJob={(id) => setActiveJobId(id)} /> : null}
        {tab === "commands" ? <CommandCenterPage caps={caps} onJob={(id) => setActiveJobId(id)} /> : null}
        {tab === "train_gan" ? <TrainGanPage onJob={(id) => setActiveJobId(id)} caps={caps} /> : null}
        {tab === "train_ae" ? <TrainAePage onJob={(id) => setActiveJobId(id)} caps={caps} /> : null}
        {tab === "data" ? <DataToolsPage onJob={(id) => setActiveJobId(id)} caps={caps} /> : null}
        {tab === "eval" ? <EvalPage onJob={(id) => setActiveJobId(id)} caps={caps} /> : null}
        {tab === "runs" ? <RunsPage /> : null}
        {tab === "configs" ? <ConfigEditorPage onJob={(id) => setActiveJobId(id)} caps={caps} /> : null}
        {tab === "files" ? <FileBrowser initialPath="." title="File Browser" /> : null}

        {tab === "jobs" ? (
          <div className="Grid">
            <div className="Card">
              <div className="CardHeader">
                <h2>Jobs</h2>
                <p>Background tasks started from the UI.</p>
              </div>
              <div className="CardBody">
                {jobsError ? <div className="Status">Error: {jobsError}</div> : null}
                <div className="JobList">
                  {jobs.length ? (
                    jobs.map((j) => (
                      <div key={j.id} className="JobItem">
                        <div className="JobItemHeader">
                          <div>
                            <div className="JobName">{j.name}</div>
                            <div className="JobMeta">
                              {j.id} · {j.status}
                              {j.return_code != null ? ` · rc=${j.return_code}` : ""}
                            </div>
                          </div>
                          <div className="Row">
                            {j.artifacts?.length ? (
                              <button
                                className="ButtonGhost"
                                onClick={() => {
                                  setActiveJobId(j.id);
                                }}
                              >
                                Artifacts
                              </button>
                            ) : null}
                            <button className="ButtonGhost" onClick={() => setActiveJobId(j.id)}>
                              View
                            </button>
                          </div>
                        </div>
                        <div className="Code">$ {j.cmd.join(" ")}</div>
                      </div>
                    ))
                  ) : (
                    <div className="Status">No jobs yet. Start one from the other tabs.</div>
                  )}
                </div>
              </div>
            </div>

            {activeJobId ? <JobPanel jobId={activeJobId} onClose={() => setActiveJobId(null)} /> : null}
          </div>
        ) : null}

        {tab !== "jobs" && activeJobId ? (
          <div style={{ marginTop: 16 }}>
            <JobPanel jobId={activeJobId} onClose={() => setActiveJobId(null)} />
          </div>
        ) : null}
      </div>
    </div>
  );
}
