# Spec: Static Operator Dashboard trên Render Static Site (thay thế Streamlit)

- **Ngày:** 2026-09-17
- **Trạng thái:** ✅ Implemented & verified live
- **Tiền đề:** `docs/superpowers/specs/2026-09-17-free-tier-deployment-design.md` (spec gốc — 2 HF Space)
- **Lý do thay thế:** Hugging Face API trả `402 Static Spaces are free for everyone, but hosting
  Gradio and Docker Spaces on free cpu-basic requires a PRO subscription.` → phương án 2 Docker Space
  không còn khả thi với ngân sách 0 đồng.

## 1. Kiến trúc chốt

```
                    ┌──────────── GitHub Actions (cd.yml) ────────────┐
                    │  ci: pytest → cd: build+push GHCR → trigger     │
                    │       Render deploy (API key / hook)            │
                    └───────────────────┬─────────────────────────────┘
                                        │ push main
                     ┌──────────────────┴───────────────────┐
                     ▼                                      ▼
   Render Web Service  apexinspect-api          Render Static Site  apexinspect-dashboard
   srv-daluk7vqj5pc73dmfr6g                     srv-dalv1kv40ujc73fbu950
   Docker, plan free, frankfurt                 plan free, publishPath ./dashboard
   https://apexinspect-api.onrender.com         https://apexinspect-dashboard.onrender.com
   env: APP_MODE/DATABASE_URL/GROQ_API_KEY/API_KEY   (không có env, chỉ file tĩnh)
                     ──────────────────┬───────────────────
                                        ▼
                        Neon PostgreSQL (free)   +   Groq API (free)
```

## 2. Quyết định thiết kế

| Quyết định | Lựa chọn | Lý do |
| :--- | :--- | :--- |
| Nền tảng UI | HTML + CSS + JS thuần, **không framework, không build step** | Render Static Site free phục vụ file tĩnh trực tiếp; không `npm install`, không `node_modules`, thời gian deploy ~2 phút |
| Vị trí host | **Render Static Site** (thay vì GitHub Pages) | Cùng workspace/dashboard với API, cùng CI, `render.yaml` là infra-as-code cho cả 2 service |
| Backend | **Không đổi** | `CORSMiddleware(allow_origins=["*"])` đã cho phép static origin gọi API |
| Bí mật | `API_KEY` nhập trên dashboard → `localStorage` | Không có secret nào trong file tĩnh được phục vụ công khai |
| Streamlit | **Xóa `streamlit_app.py`**, giữ `src/ui/app.py` | Bản full (Live Feed + Plotly) chỉ chạy local/Docker; bản 1GB Cloud đã bỏ |

## 3. Hợp đồng API mà dashboard dùng

| Method | Endpoint | Dùng cho |
| :--- | :--- | :--- |
| GET | `/health` | Banner trạng thái + phát hiện Render cold start |
| POST | `/api/v1/inspections` | Form gửi telemetry mẫu → hiển thị `inspection_id` / `ticket_id` |
| GET | `/api/v1/tickets/recent` | Bảng 10 ticket gần nhất |
| POST | `/api/v1/mes/action` | Phê duyệt/từ chối, cần header `X-API-KEY` |

## 4. Tiêu chí nghiệm thu (đã kiểm chứng)

- [x] `dashboard/index.html|styles.css|app.js` tồn tại, không tham chiếu Streamlit/framework.
- [x] `render.yaml` có `apexinspect-api` + `apexinspect-dashboard` (`staticPublishPath: ./dashboard`).
- [x] `pytest tests/test_dashboard.py tests/test_deploy_smoke.py` → 8 passed.
- [x] Static site live: `/` `200 text/html`, `/styles.css` `200 text/css`, `/app.js` `200 application/javascript`.
- [x] CORS preflight từ dashboard origin → `200` + `access-control-allow-origin` khớp origin.
- [x] Live E2E: 3 inspection lỗi liên tiếp tạo ticket `TICK-20260917-52EE` (`incident_triggered: true`),
      `POST /mes/action` với `X-API-KEY` → `200 EXECUTED`, `plc_dispatched: true`, `agent_resumed: true`;
      thiếu key → `401`.

## 5. Rủi ro còn lại

| Rủi ro | Giảm thiểu |
| :--- | :--- |
| Render free sleep sau ~15 phút idle | Banner dashboard cảnh báo cold start ~1 phút; `/health` là healthCheckPath nên Render giữ ấm khi có traffic |
| Checkpoint LangGraph nằm trong RAM instance | `agent_resumed: false` nếu instance đã restart giữa lúc tạo và lúc duyệt ticket — ticket vẫn `EXECUTED` và dispatch PLC bình thường |
| `API_KEY` gõ tay trên dashboard | Placeholder cho biết đã lưu; giá trị nằm `localStorage`, không gửi đi đâu ngoài API |