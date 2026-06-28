#!/usr/bin/env bash
#
# Build the committed GitHub Pages site under ./docs.
#
# The site contains:
# - public landing page at /GAN-AE-vision-suite/
# - static React demo at /GAN-AE-vision-suite/app/
# - screenshots/video under /assets/
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME="${CHROME:-}"

if [ -z "${CHROME}" ]; then
  CHROME="$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)"
fi

cd "${ROOT}"

python -m src.scripts.make_demo_assets --out /tmp/gan-ae-pages-demo-assets

mkdir -p docs/assets docs/app
cp /tmp/gan-ae-pages-demo-assets/gan_samples.png docs/assets/gan-samples.png
cp /tmp/gan-ae-pages-demo-assets/ae_recon_grid.png docs/assets/ae-recon-grid.png

(cd gan-ui && npm ci && npm run build)
rm -rf docs/app
mkdir -p docs/app
cp -R gan-ui/dist/. docs/app/

if [ -n "${CHROME}" ]; then
  python -m http.server 8765 --directory docs >/tmp/gan-ae-pages-http.log 2>&1 &
  SERVER_PID="$!"
  cleanup() {
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  sleep 1
  "${CHROME}" --headless --disable-gpu --no-sandbox --window-size=1440,1100 \
    --screenshot=docs/assets/screenshot-overview.png \
    "http://127.0.0.1:8765/app/" >/tmp/gan-ae-pages-chrome-overview.log 2>&1 || true
  "${CHROME}" --headless --disable-gpu --no-sandbox --window-size=1440,1100 \
    --screenshot=docs/assets/screenshot-landing.png \
    "http://127.0.0.1:8765/" >/tmp/gan-ae-pages-chrome-landing.log 2>&1 || true
  cleanup
  trap - EXIT
fi

if [ -f docs/assets/screenshot-overview.png ] && [ -f docs/assets/screenshot-landing.png ]; then
  ffmpeg -y -loop 1 -t 3 -i docs/assets/screenshot-landing.png \
    -loop 1 -t 4 -i docs/assets/screenshot-overview.png \
    -loop 1 -t 3 -i docs/assets/gan-samples.png \
    -filter_complex "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v0];[1:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v1];[2:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v2];[v0][v1][v2]concat=n=3:v=1:a=0,format=yuv420p[v]" \
    -map "[v]" -movflags +faststart docs/assets/demo-walkthrough.mp4 >/tmp/gan-ae-pages-ffmpeg.log 2>&1 || true
fi

echo "Built GitHub Pages site in docs/"
