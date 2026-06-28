import { useMemo } from "react";

function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v == null || Number.isNaN(v)) continue;
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

function toPoints(xs, ys, w, h, pad = 12) {
  const [xmin, xmax] = extent(xs);
  const [ymin, ymax] = extent(ys);
  const sx = (v) => pad + ((v - xmin) / (xmax - xmin)) * (w - pad * 2);
  const sy = (v) => h - pad - ((v - ymin) / (ymax - ymin)) * (h - pad * 2);
  const pts = [];
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    const y = ys[i];
    if (x == null || y == null) continue;
    pts.push([sx(x), sy(y)]);
  }
  return { pts, xmin, xmax, ymin, ymax };
}

export function MetricsChart({ series = [], width = 520, height = 220 }) {
  const rendered = useMemo(() => {
    const out = [];
    for (const s of series) {
      const xs = s.points.map((p) => p.x);
      const ys = s.points.map((p) => p.y);
      const { pts, ymin, ymax } = toPoints(xs, ys, width, height);
      out.push({ ...s, pts, ymin, ymax });
    }
    return out;
  }, [series, width, height]);

  return (
    <svg width={width} height={height} style={{ width: "100%", borderRadius: 12, border: "1px solid rgba(15,23,42,0.10)", background: "white" }}>
      <rect x="0" y="0" width={width} height={height} fill="transparent" />
      {/* grid */}
      {[0.25, 0.5, 0.75].map((t) => (
        <line key={t} x1="12" x2={width - 12} y1={12 + t * (height - 24)} y2={12 + t * (height - 24)} stroke="rgba(15,23,42,0.06)" />
      ))}
      {rendered.map((s, idx) => (
        <g key={idx}>
          <polyline
            fill="none"
            stroke={s.color || (idx === 0 ? "#4f46e5" : "#14b8a6")}
            strokeWidth="2"
            points={s.pts.map(([x, y]) => `${x},${y}`).join(" ")}
          />
        </g>
      ))}
    </svg>
  );
}

