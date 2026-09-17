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
- `apexinspect-dashboard` — `type: static_site`, `staticPublishPath: ./dashboard`,
  `buildCommand: ""` (không có build step — Render phục vụ file tĩnh trực tiếp).

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
| API Base URL | `https://apexinspect-api.onrender.com` (đã là mặc định trong `app.js`) |
| X-API-KEY | khớp `API_KEY` trên Render — xem **Bước 6** để lấy/xoay key |

Cả hai lưu trong `localStorage` của trình duyệt (`apex_api_base`, `apex_api_key`),
**không bao giờ commit vào repo**. Key chỉ cần cho thao tác MES approve/reject.

Dashboard cung cấp 4 chức năng vận hành:
1. Health banner (tự phát hiện Render đang sleep)
2. Gửi inspection mẫu → nhận `inspection_id` / `ticket_id`
3. Bảng tickets gần nhất (click 1 ticket để đổ vào form MES)
4. MES approve/reject (gửi kèm header `X-API-KEY`)

> Giao diện đầy đủ (Live Feed camera sim + đồ thị Plotly) vẫn có ở `src/ui/app.py`
> khi chạy local/Docker: `streamlit run src/ui/app.py`.

## Bước 5 — Smoke test sau deploy (đã chạy ✅ ngày 2026-09-17)

```bash
# API
curl -fsS https://apexinspect-api.onrender.com/health
# → {"status":"ok","service":"apexinspect-gateway","version":"1.0.0"}

curl -fsS -X POST https://apexinspect-api.onrender.com/api/v1/inspections \
  -H 'Content-Type: application/json' \
  -d '{"line_id":"SMT-LINE-01","is_defective":true,"defect_classes":["short_circuit"],"confidence_scores":[0.94],"bounding_boxes":[[160,220,240,280]],"inference_time_ms":25.0}'
# → {"inspection_id":"INSP-...","status":"RECORDED","incident_triggered":false,"ticket_id":null}

# Dashboard (static assets)
for p in / /styles.css /app.js; do
  curl -sS -o /dev/null -w "$p %{http_code}\n" https://apexinspect-dashboard.onrender.com$p
done
# → / 200, /styles.css 200, /app.js 200
```

Bằng chứng E2E đã kiểm chứng trên hạ tầng thật (chạy lại toàn bộ ngày 2026-09-17 sau khi xoay `API_KEY`):

| Bước | Kết quả thực đo |
|---|---|
| `GET /health` (sau cold start) | `200 {"status":"ok","service":"apexinspect-gateway","version":"1.0.0"}` |
| CORS preflight từ `https://apexinspect-dashboard.onrender.com` | `200` + `access-control-allow-origin: https://apexinspect-dashboard.onrender.com` + `access-control-allow-headers: content-type,x-api-key` |
| Static assets `/`, `/styles.css`, `/app.js` | `200`, content-type `text/html` / `text/css` / `application/javascript` |
| 3 inspection lỗi liên tiếp khi **không** có ticket pending | `incident_triggered: true`, ticket `TICK-20260917-B517` (`severity=CRITICAL`, `action_type=HALT_LINE`, `root_cause_analysis` dài 1329 ký tự, `thread_id` đã set) |
| Inspection tiếp theo khi đã có ticket pending | `incident_triggered: false` — đúng logic chống tạo ticket trùng |
| `POST /mes/action` **không** có `X-API-KEY` | `401 Missing required 'X-API-KEY' authentication header.` |
| `POST /mes/action` với key **sai** | `403 Invalid 'X-API-KEY' credential.` |
| `POST /mes/action` với key **đúng** | `200 {"status":"EXECUTED","plc_dispatched":true,"ticket_id":"TICK-20260917-B517"}` |
| `GET /api/v1/lines/SMT-LINE-01/metrics` sau HALT | `status: HALTED`, dữ liệu đọc lại được từ Neon |
| Test suite local | `87 passed` (`.venv/bin/python -m pytest tests/ -q`, exit 0 — 2 lần chạy độc lập; `unittest discover` cũng xanh nhờ DB guard) |

> **Hạn chế đã biết (đo được, không phải giả định):** `agent_resumed` trả `false` khi container API đã
> restart/redeploy giữa lúc tạo ticket và lúc supervisor approve — vì LangGraph dùng `MemorySaver`
> (checkpoint trong RAM, mất khi process chết). Khi tạo và approve trong **cùng** một process, giá trị là
> `true`. Hành vi này nằm trong `InspectionService.resolve_ticket` (`try/except` + audit log ghi
> `AgentResumed=...`), không phải lỗi deploy: ticket, audit trail và lệnh PLC vẫn thực thi đúng.
> Muốn state HITL sống qua restart thì cần checkpointer bền vững (Postgres/Redis) — ngoài phạm vi bản free-tier này.

## Bước 6 — Xoay / lấy `API_KEY` cho dashboard

`API_KEY` là **credential duy nhất** bảo vệ `POST /api/v1/mes/action` (lệnh HALT_LINE/ROUTE_REWORK
ra PLC thật). Vì vậy giá trị mặc định yếu trong code (`dev-factory-key-secret`) **không dùng** ở production.

- Giá trị production hiện tại: đặt trên Render (Environment → `API_KEY`) **và** ghi trong file `.env`
  local (đã gitignore) dưới cùng tên `API_KEY` — đọc bằng `grep '^API_KEY=' .env` trên máy bạn.
- Không commit key, không dán vào issue/chat. Xoay định kỳ:

```bash
# 1) sinh key mới (không in ra màn hình)
NEW=$(python3 -c "import secrets; print('apex-mes-' + secrets.token_urlsafe(32))")

# 2) cập nhật Render (giữ nguyên các biến khác)
curl -fsS -X PUT \
  -H "Authorization: Bearer $RENDER_API_KEY" -H 'Content-Type: application/json' \
  https://api.render.com/v1/services/srv-daluk7vqj5pc73dmfr6g/env-vars \
  -d "[{\"key\":\"APP_MODE\",\"value\":\"api\"},
       {\"key\":\"DATABASE_URL\",\"value\":\"$DATABASE_URL\"},
       {\"key\":\"GROQ_API_KEY\",\"value\":\"$GROQ_API_KEY\"},
       {\"key\":\"API_KEY\",\"value\":\"$NEW\"}]"

# 3) Render KHÔNG tự redeploy khi đổi env qua API — trigger tường minh rồi chờ status=live
curl -fsS -X POST -H "Authorization: Bearer $RENDER_API_KEY" -H 'Content-Type: application/json' \
  -d '{"clearCache":"do_not_clear"}' \
  https://api.render.com/v1/services/srv-daluk7vqj5pc73dmfr6g/deploys

# 4) cập nhật .env local để dev/dashboard khớp
```

> ⚠️ Bắt buộc bước 3: cập nhật env var qua API **không** tự tạo deploy, service vẫn chạy key cũ
> cho tới khi redeploy (đã gặp thật: `403` dù đã PUT, sau khi deploy lại mới `200`).

## Troubleshooting

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| API cold-start chậm/timeout lần đầu | Render free sleep sau 15' idle | chờ ~1 phút, retry (dashboard báo rõ trên banner) |
| API 500 khi POST inspection | `DATABASE_URL` sai/thiếu | kiểm tra env vars Render + logs |
| RCA fallback (không LLM) | thiếu `GROQ_API_KEY` | thêm env var (agent vẫn chạy) |
| `mes/action` 401 | thiếu header / chưa nhập key ở mục Cấu hình | nhập `X-API-KEY` rồi Lưu |
| `mes/action` 403 sau khi vừa đổi `API_KEY` | đổi env qua API **không** tự redeploy | trigger deploy tường minh rồi chờ `live` (Bước 6, cảnh báo ⚠️) |
| `agent_resumed: false` khi approve | container đã restart sau lúc tạo ticket → mất checkpoint RAM | đúng hành vi `MemorySaver` (xem ghi chú Bước 5), không phải lỗi deploy |
| Dữ liệu mất sau restart | SQLite thay vì Neon | đặt `DATABASE_URL` (Bước 1) |
| Dashboard trắng / fetch fail | API chưa thức hoặc base URL sai | kiểm tra banner + `localStorage.apex_api_base` |
| Tạo HF Docker Space báo 402 | HF yêu cầu PRO cho Docker | đúng — không dùng HF nữa |
