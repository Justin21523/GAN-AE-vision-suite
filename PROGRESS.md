# 進度紀錄（繁體中文）

## 2025-12-25：註解化與可讀性基礎建設（第 1 階段）

### 目標
- 先從「核心可讀性」著手：讓設定檔、資料載入、基本模型/損失/指標的原理更容易被閱讀。
- 開始把程式碼補上英文註解（以 module docstring + function/class docstring 為主）。
- 建立此檔案作為之後每次改動的繁中進度/原理筆記。

### 已完成內容（本次）
- `src/utils/config.py`：補上英文說明，並加入「深層合併」與「key 相容層」。
  - 重要相容：`data.img_size` ↔ `data.image_size`、`train` ↔ `training`、`dataset` ↔ `name`
  - 若安裝了 `addict`，`load_config()` 回傳值可同時支援 `cfg["data"]["batch_size"]` 與 `cfg.data.batch_size`
- `src/utils/seed.py`：補上英文說明，並新增 `seed_worker()`（給 DataLoader worker 使用）。
- `src/data/transforms.py`：補上英文說明，新增 `build_transforms(cfg, train=...)`，可讀 YAML 的 `train_transforms/val_transforms` 規格。
- `src/data/datasets.py`：補上英文說明，新增 `build_dataset(cfg, train=...)` 與 `get_loader(...)`，並修正 paired/unpaired image folder 讀取目錄的明顯問題。
- GAN/損失/指標/Trainer：在以下檔案補上英文 docstring，並把關鍵邏輯用註解說清楚：
  - `src/models/gan/generator.py`
  - `src/models/gan/discriminator.py`
  - `src/models/gan/ema.py`
  - `src/losses/gan.py`
  - `src/losses/reconstruction.py`
  - `src/losses/vae_loss.py`
  - `src/metrics/image.py`
  - `src/training/ae_trainer.py`
  - `src/training/trainers.py`（placeholder 說明）
- AE/VAE 模型與工廠函式：補上英文 docstring，特別說明「lazy 建層」與 VAE reparameterization：
  - `src/models/ae/ae.py`
  - `src/models/ae/conv_ae.py`
  - `src/models/ae/vae.py`
  - `src/scripts/build_model.py`
- GAN 訓練/取樣/評估腳本：補上英文 docstring，並用註解標出訓練流程與 WGAN-GP/hinge loss 的位置：
  - `src/scripts/train_gan.py`
  - `src/scripts/sample_gan.py`
  - `src/scripts/eval_fid.py`
- 資料/指標 sanity check 腳本：補上英文 docstring，說明輸出圖檔與 PSNR/SSIM 的用途：
  - `src/validate_data.py`
- 推論服務層：補上英文 docstring，說明 checkpoint → 重建模型 → 取樣輸出 grid 的流程：
  - `src/service/gan_infer.py`
- 根目錄 `scripts/` placeholder 腳本：補上英文 docstring，避免空檔案讓人誤會已實作：
  - `scripts/distill_model.py`
  - `scripts/prune_model.py`
  - `scripts/quantize_model.py`
  - `scripts/run_online_learning.py`
  - `scripts/generate_fairness_report.py`
- 測試 placeholder：補上英文 docstring 交代預期測項範圍：
  - `tests/test_compression.py`
  - `tests/test_fairness.py`
  - `tests/test_online_learning.py`
- 前端（React/Vite）：補上英文註解，說明 API fallback 與取樣流程：
  - `gan-ui/src/api.ts`
  - `gan-ui/src/App.jsx`
  - `gan-ui/src/main.jsx`
  - `gan-ui/src/vite-env.d.ts`

### 原理筆記（摘要）

#### 1) 設定檔合併與相容層
這個 repo 在不同檔案裡出現過不同命名（例如 `img_size` / `image_size`）。為了讓舊/新程式都能讀同一份 YAML，本次在 `load_config()` 加上正規化：

```py
# 例：讓兩種 key 都可用
data["image_size"] = data.get("image_size", data.get("img_size"))
data["img_size"] = data.get("img_size", data.get("image_size"))
```

#### 2) 可重現性（seed）與 DataLoader worker
多 worker 的資料增強如果沒有同步 seed，很容易「看似同一個 seed 但結果不一樣」。`seed_worker()` 的重點是把 PyTorch 派生出的 worker seed 再餵給 NumPy/Python：

```py
worker_seed = torch.initial_seed() % 2**32
np.random.seed(worker_seed)
random.seed(worker_seed)
```

#### 3) WGAN-GP 的 gradient penalty
`compute_gradient_penalty()` 做的是在 real/fake 之間插值，對 critic 輸出對輸入求梯度，並把梯度範數拉回到 1 附近（近似 1-Lipschitz）。

### 下一步（第 2 階段建議）
- 繼續把 `src/models/ae/*`、`src/scripts/*`、`src/eval/*`、`src/api/*`、`src/app/*` 等檔案補齊英文註解（以「你會實際用到的入口」優先）。
- 若你希望「能跑起來」的體驗更一致，我可以再整理一份「目前 repo 內 config schema/入口點的統一建議」。

## 2025-12-25：讓入口可直接跑（第 2 階段）

### 目標
- 把「因為引用不存在模組而跑不動」的入口改成可直接使用現有核心（`src/service/gan_infer.py`）。
- 對仍缺少整套模組（img2img/warehouse/jobs/reporting）的 scripts，改為「可執行且錯誤訊息清楚」。
- 補齊 `scripts/*.sh` 與 `configs/*.yaml` 的英文註解，提升整體可讀性（接近你要的 90%+）。

### 已完成內容（本次）
- FastAPI 入口改為最小可用版本（不再依賴缺失的 warehouse/registry/jobs）：
  - `src/api/main.py`：提供 `/api/load`、`/api/generate`（含多組 alias）用於載入 checkpoint 與輸出 PNG grid。
- Gradio UI 改為最小可用版本（不再依賴 registry/job engine）：
  - `src/app/gradio_gan.py`：可載入 checkpoint、設定 n/nrow/seed、產生 grid 圖（並在 gradio 依賴有問題時給出提示）。
- 移除 GAN 取樣路徑對 torchvision 的硬依賴（避免環境 torch/torchvision 版本不合造成匯入失敗）：
  - `src/service/gan_infer.py`：以純 torch 實作 `_make_grid()` 取代 `torchvision.utils.make_grid`
- `scripts/` 中引用缺失模組的入口，改為「fail fast + 提示可用替代方案」：
  - `scripts/train_img2img.py`、`scripts/train_cyclegan.py`、`scripts/infer_img2img.py`
  - `scripts/eval_metrics.py`、`scripts/cleanup_checkpoints.py`、`scripts/export_model_card.py`
- Shell scripts 英文註解補齊並修正明顯不可用點：
  - `scripts/serve_api.sh`：移除會破壞 import 的 `PYTHONPATH=src`，並預設 `AI_CACHE_ROOT=./.ai_cache`
  - `scripts/setup_pre_commit.sh`：加入 `.pre-commit-config.yaml` / `requirements-dev.txt` 存在性檢查
  - `scripts/deploy_k8s.sh`：若缺少 `deploy/` 直接提示（避免跑一半才爆）
  - `scripts/monitoring_setup.sh`：補上用途/產物說明
- 全部 `configs/*.yaml` 於檔頭新增英文說明（含 placeholder 檔案），讓每份 config 的用途更清楚。

### 原理筆記（摘要）

#### 1) 為什麼改成「最小 FastAPI/Gradio」
目前 repo 內有可用的 GAN 取樣核心：`GANService.load_checkpoint()` + `generate_grid()`。
比起硬補一整套 warehouse/registry/jobs（且 repo 本身缺檔），先提供能跑的 API/UI 入口，讓你可以：
- 先把訓練產生的 `ckpt_epoch*.pt` 載入
- 再用固定 seed 反覆產生 grid 觀察結果

#### 2) API 的核心介面（概念）
API 其實只做兩件事：
```py
# 1) load: 伺服器端載入 checkpoint，建立 generator
svc.load_checkpoint(path)

# 2) generate: 用固定 seed 產生 grid，回傳 PNG bytes
img = svc.generate_grid(GenerateParams(...))
```

### 快速使用（建議）
- 開 API：`scripts/serve_api.sh`
- 開 Gradio：`python -m src.app.gradio_gan`
- 開 React UI：`cd gan-ui && npm ci && npm run dev`

### 補充：一鍵啟動（API + UI）
- 新增 `scripts/dev_fullstack.sh`，可用單一指令同時啟動 FastAPI + Vite dev server。
- `gan-ui/README.md` 已改為繁中「快速上手」，並補上 `.env.example` 說明 `VITE_API_BASE_URL` 的用途。

## 2025-12-25：刪除過時版本 + 解除 torchvision/torchmetrics 入口阻塞（第 3 階段）

### 目標
- 把 repo 內「跑不動/缺模組」的過時入口直接移除，避免後續維護與理解成本。
- 讓主要 pipeline（資料載入、GAN 訓練、API/UI）在 **不依賴 torchvision/torchmetrics** 的情況下仍可匯入並運行（缺資料時要能清楚報錯）。

### 已完成內容（本次）
- `src/data/datasets.py`：重寫為以 **torchvision-free** 為預設的資料載入層。
  - 支援 `celeba` / `imagefolder`：以資料夾掃描讀圖（缺資料夾或無圖片會 fail fast）。
  - 支援 `mnist`：直接解析 IDX(.gz) 原始檔（不需要 torchvision）。
  - 支援 `cifar10`：讀取 `cifar-10-batches-py/` pickle batches（不需要 torchvision）。
  - 移除 img2img 相關 dataset 與 `get_dataloader/get_dataset` 舊介面（已不再是本 repo 的主線 pipeline）。
- `src/metrics/fidkid.py`：改為 **lazy import** torchmetrics（避免 import-time 因 torchmetrics/torchvision 不存在而整個訓練入口掛掉）。
- `src/scripts/train_gan.py`：當 FID/KID 依賴缺失時，直接停用該段計算並提示原因（避免白跑 val loop）。
- 清理過時/不完整的程式碼與設定：
  - 刪除 `src/eval/*`（舊 runner stack，且含 torchvision 依賴）
  - 刪除根目錄 `scripts/*` 內 img2img/warehouse/compression/fairness/online-learning placeholder
  - 刪除不再對應現行入口的 configs（img2img、suite policy/defaults、compression、舊 AE/GAN schema）
  - 刪除 placeholder tests（compression/fairness/online learning）
- 小幅文件同步：
  - `AGENTS.md`：更新結構描述（`scripts/` 只保留 API/fullstack shell helpers）
  - `src/infer.py` / `src/__init__.py`：移除已刪除入口的提及

### 原理筆記（摘要）

#### 1) 為什麼「資料集讀取」要去掉 torchvision
你目前環境的 torchvision 匯入有機會因版本/編譯不合而直接爆掉（例如 `operator torchvision::nms does not exist`），這會讓「只想跑 GAN 訓練或 API」也被卡死。
所以改成：
- 只有在真的需要且環境支援時才使用 torchvision（本次乾脆完全不依賴它）
- 讀 MNIST/CIFAR10 用「標準原始檔格式」即可完成（IDX / python batches）

範例（MNIST IDX 解析的核心概念）：
```py
magic = struct.unpack(">I", f.read(4))[0]
if magic == 2051:  # images
    n, rows, cols = struct.unpack(">III", f.read(12))
    arr = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8).reshape(n, rows, cols)
```

#### 2) 為什麼 FID/KID 要 lazy import
FID/KID 通常透過 Inception 特徵抽取；不少實作會間接依賴 torchvision。
如果你只想先把訓練流程跑起來（或環境沒有 torchmetrics），那就應該：
- 入口可正常跑
- 指標缺依賴時「自動停用 + 清楚提示」

## 2025-12-25：AE/VAE 入口統一 + configs/tests 對齊（第 4 階段）

### 目標
- 把 AE/VAE 的訓練入口整理成「跟 GAN 訓練同一品質」：可用 CLI、可記錄、可存 checkpoint、可輸出樣本圖。
- 修正 AE/VAE 相關的明顯 bug（避免看起來有入口但其實跑不動）。
- 讓 tests/configs 不再依賴外部資料（避免拿到 repo 就因為缺 MNIST/CIFAR 檔而爆）。

### 已完成內容（本次）
- AE/VAE 模型修正：
  - `src/models/ae/ae.py`：修正輸出 channel 固定為 1 的問題，改為輸出 `img_channels`，並在尾端加上 `tanh`（與資料正規化 [-1,1] 對齊）。
  - `src/models/ae/vae.py`：修正 `ConvVAE` 的 decoder 路徑（原本直接把 latent 丟進 ConvTranspose 堆疊會炸），並為 VAE 類別加上 `is_vae=True` 供訓練腳本辨識。
- 工廠函式對齊：
  - `src/scripts/build_model.py`：`type: vae` 改為預設建 `ConvVAE`；`vae-fixed` 才用 `VariationalAutoEncoder`，並支援從 `data.image_size/img_size` 自動推導 `input_size`。
- AE/VAE 訓練入口統一：
  - `src/scripts/train_ae.py`：改為本 repo 的 **canonical AE/VAE trainer**（支援 AMP、beta-KL、checkpoint、每 epoch 輸出 input/recon grid）。
  - `src/scripts/train_vae.py`：改為向後相容 alias（直接呼叫 `train_ae`），避免維護兩份不同品質的入口。
- Configs 對齊（避免誤導）：
  - `configs/dataset_mnist.yaml` / `configs/dataset_celeba.yaml`：移除/修正不再使用的欄位（如錯置的 `input_size`），並把 `download` 預設改為 `false`（此 repo snapshot 不自動下載）。
  - `configs/model/vae_celeba.yaml`：改成可直接用 `train_ae` 跑的 canonical schema（`model/type/train/data/logging/save`）。
- Tests 對齊：
  - `tests/test_dataloader.py`：改為「自建暫存圖片資料夾」的自包含測試，不再依賴 MNIST 檔案存在。
  - `tests/test_*.py`：補上 module docstring（tests 現在 4/4 都有），提升可讀性。

### 原理筆記（摘要）

#### 1) 為什麼要先跑一次 dummy forward（warmup）
`AutoEncoder` 內部有「lazy 建層」（首次 forward 才知道 encoder feature 維度）。
如果一開始就開 AMP，第一個 forward 可能在 autocast 下以 fp16 建出 Linear/Decoder，會讓權重 dtype 不理想。
所以訓練入口會先用 fp32 dummy tensor 觸發建層，再進入 AMP 迴圈：
```py
dummy = torch.zeros(1, C, H, W, device=device, dtype=torch.float32)
_ = model(dummy)
```

#### 2) ConvVAE decoder 的正確資料流
VAE 的 decoder 需要先把 latent z 投影回 feature map，再做 ConvTranspose 上採樣：
```py
z = reparameterize(mu, logvar)
h = dec_fc(z)          # (B, C, H', W')
out = decoder(h)       # (B, C, H, W)
```
