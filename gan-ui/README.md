# GAN UI（React + Vite）

這個資料夾是前端介面，用來呼叫後端 FastAPI 產生 GAN 的 sample grid 圖片。

## 一鍵跑起來（推薦）

第一次請先安裝依賴：

```bash
# repo root
python -m pip install -r requirements.txt
cd gan-ui && npm ci
```

之後可用一個指令同時啟動 API + UI：

```bash
# repo root
scripts/dev_fullstack.sh
```

- API：`http://127.0.0.1:8000`
- UI：`http://127.0.0.1:5173`（Vite dev server）

## UI 功能概覽

這個 UI 以「本機使用」為前提，目標是讓 repo 內常用 CLI 指令都有對應的可視化操作入口：

- Overview：作品展示首頁；API 不在線時仍可作為靜態 demo 截圖頁
- Sampler：用 API 方式載入 checkpoint / 產生 grid（即時預覽）
- Workflow：一鍵流程（Data Report → 檢查 → 開始訓練）
  - 支援 GAN / AE/VAE 模式切換，並用簡單的紅黃綠提示顯示資料檢查結果
- Train GAN / Train AE-VAE：透過後端 job runner 以背景工作執行 `python -m ...`
- Data Tools：`data_report / prepare_data / validate_data`
- Eval：`sample_gan / eval_fid`
- Commands：從 `/api/capabilities` 動態產生表單，可執行所有後端 job
- Configs：編輯 YAML 並存到 `./.ai_cache/configs`，方便做訓練調整
  - 支援 Validate、Save & Run Data Report、Save & Train（auto/指定）
  - 支援 Overrides：用表單覆寫常用參數並套用到 YAML（不用手改）
  - 支援 Overlays：把 overrides 存成 `./.ai_cache/configs/overrides/<name>.yaml`，可重複套用
- Jobs：查看背景工作狀態、命令列、即時 logs、可取消
  - 支援 SSE：即時顯示 logs + metrics（讀取 `metrics.jsonl`）
- Files：瀏覽 repo / `AI_CACHE_ROOT` 下的檔案，預覽圖片、讀取文字檔（含 json/jsonl/yaml/log）
  - 支援 shortcuts（outputs/logs/data/.ai_cache/jobs）與 recent paths
- Runs：掃描 `./logs/**/meta.json` 顯示歷史 runs，查看 `metrics.jsonl` tail 並可瀏覽檔案

## 靜態 Demo 模式

若只部署 `npm run build` 產出的靜態檔，Overview 頁仍會顯示 demo mode、代表性 metrics、sample grid 與架構摘要。需要真正啟動訓練、取樣、檔案瀏覽時，再搭配本機 FastAPI。

## 手動啟動（兩個終端機）

Terminal A（API）：

```bash
scripts/serve_api.sh
```

Terminal B（UI）：

```bash
cd gan-ui
npm run dev
```

Vite 已在 `gan-ui/vite.config.js` 設定 proxy：前端對 `/api/*` 的請求會自動轉發到 `http://localhost:8000`。

## API 路由與行為

前端主要呼叫兩個動作：
- 載入 checkpoint：`POST /api/load`（也支援 `/api/gan/load` 等 alias）
- 產生 grid：`POST /api/generate`（也支援 `/api/gan/generate` 等 alias，回傳 `image/png`）

Checkpoint 路徑可用 repo 相對路徑，例如：`logs/stage3_wgangp/ckpt_epoch1.pt`。

## 背景工作（Jobs API）

UI 會使用以下 endpoints 以「白名單指令」啟動背景工作：

- `GET /api/jobs`：列出 jobs
- `POST /api/jobs/start`：啟動 job（例如 train/data tools/eval）
- `GET /api/jobs/{id}`：job 狀態
- `GET /api/jobs/{id}/logs?tail=200`：tail logs
- `POST /api/jobs/{id}/cancel`：取消 job

Logs 預設會寫在 `AI_CACHE_ROOT/jobs/<job_id>/job.log`（若未設定則使用 repo 的 `./.ai_cache`）。

## 檔案瀏覽（FS API）

UI 的 Files 與 JobPanel 的 Browse 功能會使用以下 endpoints：

- `GET /api/fs/list?path=...`：列出資料夾
- `GET /api/fs/read?path=...`：讀文字檔（有大小上限）
- `GET /api/fs/file?path=...`：下載/預覽檔案（圖片可直接 `<img src=...>`）

為避免誤用，後端限制只能存取：
- repo root
- `AI_CACHE_ROOT`（預設 `./.ai_cache`）

## 設定（可選）

預設 API base 為 `/api`。若你要直接連遠端 API（或不走 proxy），可建立 `gan-ui/.env.local`：

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

（參考範本：`gan-ui/.env.example`）
