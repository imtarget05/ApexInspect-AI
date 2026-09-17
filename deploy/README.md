# 🚀 ApexInspect AI — Free-Tier Deployment (Render API + Streamlit Cloud)

Chi phí: **0 USD**. Kiến trúc: **Render Web Service (Docker, API)** + **Streamlit Community Cloud (UI)** + Neon PostgreSQL + Groq.

> ⚠️ 2026-09: Hugging Face Docker Spaces **yêu cầu PRO subscription** (402 error trên
> free cpu-basic) nên kiến trúc 2 HF Spaces cũ bị loại. Spec gốc ở
> `docs/superpowers/specs/2026-09-17-free-tier-deployment-design.md` (phần HF mang
> tính lịch sử).

```text
push main → GitHub Actions (pytest + GHCR build + trigger Render)
              └─ Render: apexinspect-api (Docker, free, frankfurt) ─┐
Streamlit Cloud: streamlit_app.py (remote mode) ──BACKEND_URL───────┘
              └────┬────┘
         Neon PostgreSQL (free) + Groq (free)
```

**URLs live:**
- API: `https://apexinspect-api.onrender.com` (`/health`)
- UI: Streamlit Cloud app trỏ `BACKEND_URL` về API trên (xem Bước 4)

## Bước 1 — Secrets (đã xong ✅)

`DATABASE_URL` (Neon) + `GROQ_API_KEY` lấy từ `.env` local, đã đặt làm env vars
trên Render service `apexinspect-api` (`srv-daluk7vqj5pc73dmfr6g`). Xoay key khi cần
trên dashboard Render → Environment.

| Biến | Ý nghĩa |
|---|---|
| `APP_MODE=api` | container chạy uvicorn (xem `Dockerfile`) |
| `DATABASE_URL` | Neon connection string (**bắt buộc** — disk Render free là ephemeral) |
| `GROQ_API_KEY` | RCA agent (thiếu vẫn chạy fallback) |
| `API_KEY` | bảo vệ `POST /api/v1/mes/action` qua header `X-API-KEY` |

## Bước 2 — Render service (đã xong ✅)

- Service `apexinspect-api`, Docker (`./Dockerfile`), plan **free**, region **frankfurt**,
  `healthCheckPath: /health`, `autoDeploy: yes` (push main tự deploy).
- Khai báo hạ tầng bằng code: `render.yaml` (Blueprint) trong repo.
- CD (`.github/workflows/cd.yml`, job `deploy-render-api`) trigger deploy tường minh:
  Secrets/Vars cần trên GitHub → Settings → Secrets and variables → Actions:

| Loại | Tên | Giá trị |
|---|---|---|
| Secret | `RENDER_API_KEY` | Render API key (đã đặt ✅) |
| Variable | `RENDER_SERVICE_ID` | `srv-daluk7vqj5pc73dmfr6g` (đã đặt ✅) |
| Secret (optional) | `RENDER_DEPLOY_HOOK_URL` | Deploy Hook URL nếu muốn thay API trigger |

## Bước 3 — Deploy (tự động)

```bash
git push origin main
```

Render build Docker image lần đầu ~8-12 phút (OpenCV/ONNX). Theo dõi:
dashboard Render → service → Events/Logs, hoặc API:
`GET /v1/services/<id>/deploys` (status `live` = xong).

## Bước 4 — UI trên Streamlit Community Cloud (1 click, free)

1. Đăng ký [share.streamlit.io](https://share.streamlit.io) (login bằng GitHub).
2. **New app** → repo `imtarget05/ApexInspect-AI`, branch `main`,
   **Main file path: `streamlit_app.py`** (⚠️ KHÔNG dùng `src/ui/app.py` — file đó
   import cv2/detector/agent, vượt 1GB RAM free tier).
3. **App settings → Secrets**:
   ```toml
   BACKEND_URL = "https://apexinspect-api.onrender.com"
   API_KEY = "<khớp với API_KEY trên Render>"
   ```
   → `streamlit_app.py` ở remote mode chỉ import stdlib + streamlit (<150MB),
   mọi nghiệp vụ gọi API Render (chung Neon DB). Không đặt `BACKEND_URL` → app
   fallback chạy full pipeline local (chỉ dùng cho Docker/PC mạnh).

## Bước 5 — Smoke test sau deploy

```bash
# API
curl -fsS https://apexinspect-api.onrender.com/health
curl -fsS -X POST https://apexinspect-api.onrender.com/api/v1/inspections \
  -H 'Content-Type: application/json' \
  -d '{"line_id":"SMT-LINE-01","is_defective":true,"defect_classes":["short_circuit"],"confidence_scores":[0.94],"bounding_boxes":[[160,220,240,280]],"inference_time_ms":25.0}'
# Kết quả mong đợi: {"inspection_id":"...","status":"RECORDED",...}

# UI: mở URL Streamlit Cloud → gửi inspection mẫu ở tab 1, xem ticket ở tab 2.
```

Xác nhận dữ liệu: query Neon (SQL Editor) thấy row mới trong `inspections`.

## Troubleshooting

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| API cold-start chậm/timeout lần đầu | Render free sleep sau 15' idle | chờ ~1 phút, retry |
| API 500 khi POST inspection | `DATABASE_URL` sai/thiếu | kiểm tra env vars Render + logs |
| RCA fallback (không LLM) | thiếu `GROQ_API_KEY` | thêm env var (agent vẫn chạy) |
| `mes/action` 401 | sai `X-API-KEY` | so khớp `API_KEY` Render |
| Dữ liệu mất sau restart | SQLite thay vì Neon | đặt `DATABASE_URL` (Bước 1) |
| Streamlit Cloud OOM/crash | dùng nhầm `src/ui/app.py` | Main file phải là `streamlit_app.py` |
| Tạo HF Docker Space báo 402 | HF yêu cầu PRO cho Docker | đúng — không dùng HF nữa |
