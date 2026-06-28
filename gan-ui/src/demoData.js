export const demoMetrics = [
  { x: 1, y: 0.92 },
  { x: 2, y: 0.71 },
  { x: 3, y: 0.54 },
  { x: 4, y: 0.43 },
  { x: 5, y: 0.36 },
  { x: 6, y: 0.31 },
];

export const demoRuns = [
  { name: "wgangp-anime64-smoke", fid: "78.4", kid: "0.031", status: "baseline" },
  { name: "sngan-resnet-128", fid: "42.7", kid: "0.014", status: "best visual" },
  { name: "conv-ae-recon", fid: "-", kid: "-", status: "reconstruction" },
];

export const demoTiles = Array.from({ length: 16 }, (_, i) => {
  const hue = (i * 29 + 178) % 360;
  return {
    id: i,
    background: `linear-gradient(135deg, hsl(${hue} 72% 62%), hsl(${(hue + 54) % 360} 78% 45%))`,
  };
});

export const architectureSteps = [
  "Config-driven datasets and transforms",
  "PyTorch AE/VAE and GAN training loops",
  "FastAPI job runner and artifact APIs",
  "React dashboard for runs, metrics, files and sampling",
];
