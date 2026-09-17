# 🚀 ApexInspect AI — Free-Tier Deployment (Render API + Static Dashboard)

Chi phí: **0 USD**. Kiến trúc: **Render Web Service (Docker, API)** + **Render Static Site (dashboard HTML/CSS/JS thuần)** + Neon PostgreSQL + Groq.

> ⚠️ Hai kiến trúc cũ đã bị loại:
> - **HF Docker Spaces (2 Space)** — 2026-09: HF trả `402` vì Docker/Gradio Space trên free `cpu-basic` **yêu cầu PRO subscription**.
> - **Streamlit Community Cloud** — thay bằng static site để không phụ thuộc Streamlit, không OOM 1GB, không cần service Python riêng.
>
> Spec gốc ở `docs/superpowers/specs/2026-09-17-free-tier-deployment-design.md` (phần HF mang tính lịch sử).

```text
push main → GitHub Actions (pytest + GHCR build + trigger Render deploy)
              ├─ Render: apexinspect-api       (Docker, free, frankfurt) ──┐
              └─ Render: apexinspect-dashboard (static, CDN toàn cầu) ────┤
                                    fetch() trực tiếp ────────────────────┤
                                                          ┌───────────────┴────────────┐
                                                    Neon PostgreSQL (free)      Groq (free)
```

**URLs live:**
- API: `https://apexinspect-api.onrender.com` (`/health`)
- Dashboard: `https://apexinspect-dashboard.onrender.com` (xem `dashboard/`)
- Interactive API docs: `https://apexinspect-api.onrender.com/docs`

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

## Bước 2 — Render services (đã xong ✅)

Khai báo hạ tầng bằng code: `render.yaml` (Blueprint) trong repo — validate được bằng
`render blueprints validate ./render.yaml`.

- `apexinspect-api` — Docker (`./Dockerfile`), plan **free**, region **frankfurt**,
  `healthCheckPath: /health`, `autoDeploy: true` (push main tự deploy).
- `apexinspect-dashboard` — `runtime: static`, `staticPublishPath: ./dashboard`,
  `buildCommand: "echo static"` (không có build step).

CD (`.github/workflows/cd.yml`, job `deploy-render-api`) trigger deploy tường minh —
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

Render build Docker image lần đầu ~8-12 phút (OpenCV/ONNX). Static site build < 30s.
Theo dõi: dashboard Render → service → Events/Logs, hoặc API:
`GET /v1/services/<id>/deploys` (status `live` = xong).

## Bước 4 — Dashboard (static, không cần thao tác thủ công)

`dashboard/` là HTML/CSS/JS thuần, không framework, không build step, không chạy
Python phía client. Render tự publish thư mục này lên CDN mỗi lần push.

Ở lần mở đầu tiên, vào mục **⚙️ Cấu hình kết nối** trên dashboard:

| Trường | Giá trị |
|---|---|
| API Base URL | `https://apexinspect-api.onrender.com` (mặc định sẵn) |
| X-API-KEY | khớp `API_KEY` trên Render — **chỉ dùng cho thao tác MES approve/reject** |

Cả hai lưu trong `localStorage` của trình duyệt, **không bao giờ commit vào repo**.

Dashboard cung cấp 4 chức năng vận hành:
1. Health banner (tự phát hiện Render đang sleep)
2. Gửi inspection mẫu → nhận `inspection_id` / `ticket_id`
3. Bảng tickets gần nhất (click 1 ticket để đổ vào form MES)
4. MES approve/reject (gửi kèm header `X-API-KEY`)

> Giao diện đầy đủ (Live Feed camera sim + đồ thị Plotly) vẫn có ở `src/ui/app.py`
> khi chạy local/Docker: `streamlit run src/ui/app.py`.

## Bước 5 — Smoke test sau deploy

```bash
# API
curl -fsS https://apexinspect-api.onrender.com/health
curl -fsS -X POST https://apexinspect-api.onrender.com/api/v1/inspections \
  -H 'Content-Type: application/json' \
  -d '{"line_id":"SMT-LINE-01","is_defective":true,"defect_classes":["short_circuit"],"confidence_scores":[0.94],"bounding_boxes":[[160,220,240,280]],"inference_time_ms":25.0}'
# Kết quả mong đợi: {"inspection_id":"INSP-...","status":"RECORDED",...}

# Dashboard
curl -fsS -o /dev/null -w '%{http_code}\n' https://apexinspect-dashboard.onrender.com/
# Kết quả mong đợi: 200

# UI: mở https://apexinspect-dashboard.onrender.com → nhập X-API-KEY → gửi inspection mẫu → xem ticket.
```

Xác nhận dữ liệu: query Neon (SQL Editor) thấy row mới trong `inspections`.

## Troubleshooting

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| API cold-start chậm/timeout lần đầu | Render free sleep sau 15' idle | chờ ~1 phút, retry (dashboard báo rõ trên banner) |
| API 500 khi POST inspection | `DATABASE_URL` sai/thiếu | kiểm tra env vars Render + logs |
| RCA fallback (không LLM) | thiếu `GROQ_API_KEY` | thêm env var (agent vẫn chạy) |
| `mes/action` 401 | sai `X-API-KEY` | so khớp `API_KEY` Render, nhập lại ở mục Cấu hình |
| `mes/action` 403 | key đúng format nhưng sai giá trị | như trên |
| Dữ liệu mất sau restart | SQLite thay vì Neon | đặt `DATABASE_URL` (Bước 1) |
| Dashboard trắng / fetch fail | API chưa thức hoặc base URL sai | kiểm tra banner + `localStorage.apex_api_base` |
| Tạo HF Docker Space báo 402 | HF yêu cầu PRO cho Docker | đúng — không dùng HF nữa |
