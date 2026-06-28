import { architectureSteps, demoMetrics, demoRuns, demoTiles } from "../demoData";
import { MetricsChart } from "../components/MetricsChart";

function SampleGrid() {
  return (
    <div className="DemoSampleGrid" aria-label="Synthetic GAN sample grid">
      {demoTiles.map((tile) => (
        <div key={tile.id} className="DemoSampleTile" style={{ background: tile.background }}>
          <span />
        </div>
      ))}
    </div>
  );
}

export function OverviewPage({ apiOnline }) {
  return (
    <div className="OverviewPage">
      <section className="HeroPanel">
        <div className="HeroCopy">
          <div className="Kicker">Portfolio Demo Platform</div>
          <h1>GAN-AE Vision Suite</h1>
          <p>
            A local-first computer vision lab for dataset quality checks, AE/VAE reconstruction, GAN training,
            checkpoint sampling, metrics tracking and artifact review.
          </p>
          <div className="Row">
            <span className={apiOnline ? "Pill PillGood" : "Pill PillWarn"}>
              API {apiOnline ? "connected" : "demo mode"}
            </span>
            <span className="Pill">PyTorch</span>
            <span className="Pill">FastAPI</span>
            <span className="Pill">React/Vite</span>
          </div>
        </div>
        <div className="HeroVisual">
          <SampleGrid />
        </div>
      </section>

      <section className="DashboardBand">
        <div className="MetricCard">
          <span>Jobs exposed</span>
          <strong>9</strong>
          <small>training, reports, sampling, eval</small>
        </div>
        <div className="MetricCard">
          <span>Smoke tests</span>
          <strong>29</strong>
          <small>CPU-friendly pytest coverage</small>
        </div>
        <div className="MetricCard">
          <span>Artifacts</span>
          <strong>JSONL + PNG</strong>
          <small>metrics, configs, grids, checkpoints</small>
        </div>
        <div className="MetricCard">
          <span>Deploy mode</span>
          <strong>Static + local API</strong>
          <small>safe public demo, full local workflow</small>
        </div>
      </section>

      <div className="OverviewGrid">
        <div className="Card">
          <div className="CardHeader">
            <h2>Demo Scenario</h2>
            <p>What an interviewer can understand in two minutes.</p>
          </div>
          <div className="CardBody">
            <div className="Timeline">
              {["Inspect dataset", "Train smoke run", "Sample checkpoint", "Compare metrics"].map((label, idx) => (
                <div key={label} className="TimelineStep">
                  <span>{idx + 1}</span>
                  <strong>{label}</strong>
                </div>
              ))}
            </div>
            <p className="BodyText">
              The static demo shows the intended platform state. The local fullstack mode runs the actual FastAPI
              job runner, writes artifacts, and streams logs/metrics into the UI.
            </p>
          </div>
        </div>

        <div className="Card">
          <div className="CardHeader">
            <h2>Training Signal</h2>
            <p>Mocked from the lightweight demo run shape.</p>
          </div>
          <div className="CardBody">
            <MetricsChart series={[{ label: "loss", color: "#4f46e5", points: demoMetrics }]} />
          </div>
        </div>

        <div className="Card">
          <div className="CardHeader">
            <h2>Run Comparison</h2>
            <p>Representative states for screenshots and walkthroughs.</p>
          </div>
          <div className="CardBody">
            <div className="JobList">
              {demoRuns.map((run) => (
                <div key={run.name} className="JobItem">
                  <div className="JobItemHeader">
                    <div>
                      <div className="JobName">{run.name}</div>
                      <div className="JobMeta">FID {run.fid} · KID {run.kid}</div>
                    </div>
                    <span className="Pill">{run.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="Card">
          <div className="CardHeader">
            <h2>Architecture</h2>
            <p>End-to-end ML engineering surface.</p>
          </div>
          <div className="CardBody">
            <div className="ArchitectureList">
              {architectureSteps.map((step) => (
                <div key={step}>{step}</div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
