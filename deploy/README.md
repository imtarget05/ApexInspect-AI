# 🚀 ApexInspect AI — Free-Tier Deployment Runbook (Hugging Face Spaces)

Chi phí: **0 USD**. Kiến trúc: 2 Space Docker SDK (UI + API) + Neon PostgreSQL + Groq.
Spec: `docs/superpowers/specs/2026-09-17-free-tier-deployment-design.md`.

## Kiến trúc

| Thành phần | Dịch vụ free | Ghi chú |
|---|---|---|
| Space **UI** (Streamlit) | HF Space Docker, CPU Basic | 2 vCPU/16GB, sleep sau 48h không dùng |
| Space **API** (FastAPI) | HF Space Docker, CPU Basic | URL public riêng cho REST gateway |
| Database | [Neon PostgreSQL](https://neon.tech) free | chung cho cả 2 Space |
| LLM | [Groq](https://console.groq.com) free tier | RCA agent |
| CI/CD | GitHub Actions + GHCR | tự động khi push `main` |

```text
GitHub Actions (push main)
  └─ pytest + build image → GHCR
      ├─ git push main → Space UI  (APP_MODE=ui  → Streamlit :7860)
      └─ git push main → Space API (APP_MODE=api → uvicorn   :7860)
                              └────┬────┘
                          Neon PostgreSQL (free) + Groq (free)
```

## Bước 1 — Neon PostgreSQL (5 phút)

1. Đăng ký [neon.tech](https://neon.tech) (free, không cần thẻ).
2. Tạo project → copy **Connection string** (dạng
   `postgresql://user:pass@ep-xxx.aws.neon.tech/neondb?sslmode=require`).
3. Đây là giá trị `DATABASE_URL`. Data thật phải nằm ở Neon — filesystem của
   Space là **ephemeral** (mất dữ liệu mỗi lần rebuild; SQLite chỉ là fallback demo).

## Bước 2 — Groq API key (2 phút)

1. Vào [console.groq.com](https://console.groq.com) → **API Keys** → Create.
2. Đây là giá trị `GROQ_API_KEY` (dùng cho RCA analysis agent).

## Bước 3 — Tạo 2 Hugging Face Space (5 phút)

Với **mỗi** Space (`apexinspect-ui`, `apexinspect-api` — tên tùy ý, cùng account):

1. [huggingface.co/new-space](https://huggingface.co/new-space) → SDK = **Docker**,
   hardware = **CPU Basic (free)**.
2. Vào **Settings → Variables and secrets**:

   | Cả 2 Space | Space API only |
   |---|---|
   | Variable `APP_MODE` = `ui` (Space UI) / `api` (Space API) | Secret `API_KEY` (chuỗi bí mật, bảo vệ `/api/v1/mes/action` bằng header `X-API-KEY`) |
   | Secret `DATABASE_URL` = connection string Neon | |
   | Secret `GROQ_API_KEY` = key Groq | |

   > **Quan trọng:** `APP_MODE` phải là **Variable** (public env var — đọc được lúc
   > runtime; Secret cũng hoạt động nhưng Variable giúp debug dễ hơn).

3. space_id dạng `tên-user-hf/apexinspect-ui` → dùng ở Bước 4.

## Bước 4 — GitHub secrets & vars (3 phút)

Repo GitHub → **Settings → Secrets and variables → Actions**:

| Loại | Tên | Giá trị |
|---|---|---|
| Secret | `HF_TOKEN` | [HF token write](https://huggingface.co/settings/tokens) (role `write`) |
| Variable | `HF_UI_SPACE` | `user-hf/apexinspect-ui` |
| Variable | `HF_API_SPACE` | `user-hf/apexinspect-api` |

Thiếu `HF_TOKEN` hoặc vars → các job deploy tự skip (pipeline vẫn xanh).

## Bước 5 — Deploy (tự động)

```bash
git push origin main
```

Pipeline CD: pytest + build image GHCR → push repo `main` vào cả 2 Space.
Space build lần đầu ~5-10 phút (cài OpenCV/ONNX Runtime).

## Bước 6 — Smoke test sau deploy

```bash
# API Space
curl -fsS https://<user>-apexinspect-api.hf.space/health
curl -fsS -X POST https://<user>-apexinspect-api.hf.space/api/v1/inspections \
  -H 'Content-Type: application/json' \
  -d '{"line_id":"SMT-LINE-01","is_defective":true,"defect_classes":["short_circuit"],"confidence_scores":[0.94],"bounding_boxes":[[160,220,240,280]],"inference_time_ms":25.0}'
# Kết quả mong đợi: {"inspection_id":"...","status":"RECORDED",...}

# UI Space
curl -fsS https://<user>-apexinspect-ui.hf.space/_stcore/health   # -> {"ok":true,...}
# Mở https://<user>-apexinspect-ui.hf.space → Live Feed + Incident Center hoạt động
```

Xác nhận dữ liệu: query Neon (SQL Editor) thấy row mới trong bảng `inspections`.

## Troubleshooting

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| Space build xong nhưng `Runtime error` | thiếu `APP_MODE` Variable | đặt lại theo Bước 3 |
| API trả 500 khi POST inspection | `DATABASE_URL` sai/thiếu | kiểm tra log Space, connection string Neon |
| RCA trả fallback (không LLM) | thiếu `GROQ_API_KEY` | thêm secret (agent vẫn chạy nhờ fallback) |
| `mes/action` trả 401/403 | thiếu/sai `X-API-KEY` | so khớp secret `API_KEY` Space API |
| Dữ liệu mất sau restart Space | không có `DATABASE_URL` (SQLite ephemeral) | đặt Neon như Bước 1 |
