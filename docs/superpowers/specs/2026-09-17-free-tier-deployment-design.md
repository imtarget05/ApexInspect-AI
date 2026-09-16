# Spec: Free-Tier Deployment — 2 Hugging Face Spaces (UI + API)

**Ngày:** 2026-09-17
**Trạng thái:** APPROVED (user duyệt thiết kế trong chat)
**Phạm vi:** Deploy toàn bộ ApexInspect AI lên hạ tầng 100% miễn phí theo đúng quy trình.

## 1. Bối cảnh

Hệ thống gồm 2 process: Streamlit Operator Console (`src/ui/app.py`, port 7860)
và FastAPI Ingestion Gateway (`src/backend/main.py`, cần uvicorn). Repo đã có
sẵn Dockerfile target port 7860, tích hợp Neon PostgreSQL + fallback SQLite,
Groq LLM, CI (`ci.yml`) và CD build GHCR + job push Hugging Face Space
(tùy chọn, hiện skip khi thiếu secret).

Yêu cầu: deploy toàn bộ lên hạ tầng miễn phí, đúng quy trình
(spec → plan → TDD → verify → commit). Azure bị loại (cần subscription).

## 2. Quyết định kiến trúc (user đã chốt)

- **Target chính: Hugging Face Space (Docker SDK, CPU Basic — free):**
  2 vCPU / 16GB RAM, public URL, sleep sau 48h không hoạt động.
- **2 Space riêng biệt, chung 1 Docker image:**
  - Space UI: `APP_MODE=ui` → `streamlit run src/ui/app.py` trên `${PORT:-7860}`.
  - Space API: `APP_MODE=api` → `uvicorn src.backend.main:app` trên `${PORT:-7860}`.
  - Cả hai cùng public URL riêng, cùng đọc/ghi một **Neon PostgreSQL** free tier.
  - `APP_MODE` là **Variable** (public env var) đặt trong Settings của từng Space —
    HF inject vào runtime nên một Dockerfile CMD duy nhất phân nhánh được.
- **Loại:** Render/Koyeb (chỉ 1 service free), Streamlit Cloud (1GB RAM, không chạy
  được FastAPI trong cùng app), Railway/Fly.io (không còn free tier thật).

## 3. Hạ tầng đích

| Thành phần | Dịch vụ | Secret/Variable |
|---|---|---|
| Space UI (Streamlit) | HF Space Docker, CPU Basic | `DATABASE_URL`, `GROQ_API_KEY`, `APP_MODE=ui` |
| Space API (FastAPI) | HF Space Docker, CPU Basic | `DATABASE_URL`, `GROQ_API_KEY`, `API_KEY`, `APP_MODE=api` |
| Database | Neon PostgreSQL free | `DATABASE_URL` (Neon connection string) |
| LLM | Groq free tier | `GROQ_API_KEY` |
| CI/CD | GitHub Actions | `HF_TOKEN` (secret), `HF_UI_SPACE`, `HF_API_SPACE` (vars) |
| Registry | GHCR | tự động qua `GITHUB_TOKEN` |

## 4. Thay đổi code (tối thiểu, không đụng business logic)

### 4.1 `src/backend/main.py` — thêm `GET /health`
- Response: `{"status": "ok", "service": "apexinspect-gateway", "version": app.version}`.
- Mục đích: liveness cho Docker HEALTHCHECK ở Space API + smoke test sau deploy.
- TDD: test trước trong `tests/test_backend.py` (unittest style, gọi trực tiếp function).

### 4.2 `Dockerfile` — CMD phân nhánh theo `APP_MODE`
- `api` → `uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}`
- mặc định (`ui`) → giữ nguyên `streamlit run src/ui/app.py ...`
- HEALTHCHECK phân nhánh: `api` → `GET /health`; `ui` → `/_stcore/health` (giữ nguyên).

### 4.3 `README.md` — YAML front-matter HF Space
- Cả 2 Space cùng repo gốc nên chỉ cần một front-matter:
  `sdk: docker`, `app_port: 7860`, `title`, `emoji`, `pinned: false`.
- HF chỉ đọc YAML front-matter đầu file; phần markdown README giữ nguyên.

### 4.4 `.github/workflows/cd.yml` — 2 job deploy HF
- Job 1: push main → Space UI (repo id `vars.HF_UI_SPACE`).
- Job 2: push main → Space API (repo id `vars.HF_API_SPACE`).
- Giữ hành vi skip-gracefully khi thiếu `HF_TOKEN`/vars (pipeline xanh out-of-the-box).
- Giữ nguyên job build GHCR hiện có.

### 4.5 `deploy/README.md` — hướng dẫn setup từng bước
- Tạo 2 HF Space (Docker SDK, CPU Basic), đặt Variable `APP_MODE` + Secrets.
- Tạo Neon DB, copy connection string vào `DATABASE_URL` (cả 2 Space).
- Tạo Groq key, HF token (write), khai báo GitHub secrets/vars.
- Quy trình smoke test sau deploy.

## 5. Ràng buộc dữ liệu & an toàn

- **HF Space filesystem là ephemeral** — SQLite `factory.db` mất dữ liệu mỗi lần
  rebuild. Do đó `DATABASE_URL` (Neon) là **bắt buộc** cho dữ liệu thật;
  SQLite chỉ là fallback demo khi thiếu biến.
- `init_db()` (`Base.metadata.create_all` + seed line `SMT-LINE-01`) chạy an toàn
  từ cả 2 process trên Postgres; khối migration SQLite PRAGMA đã bọc try/except.
- CORS hiện `allow_origins=["*"]` — chấp nhận cho demo/portfolio; ghi nhận trong
  tài liệu, không thay đổi trong phạm vi spec này (tách làm việc sau nếu cần).
- Test không bao giờ chạm Neon: `tests/conftest.py` đã override `DATABASE_URL`
  về SQLite throwaway — giữ nguyên cơ chế này.

## 6. Quy trình thực thi

1. Spec này được commit trước (`docs/superpowers/specs/`).
2. Viết kế hoạch triển khai `plans/plan-05-free-tier-deployment.md`
   (task 2–5 phút, ghi rõ file path + verify step).
3. Thực hiện theo TDD: test `/health` RED → implement GREEN.
4. Verify local: `pytest tests/` → `docker build` → `docker run` cả 2 mode
   (`APP_MODE=api`: curl `/health` + POST 1 inspection; `APP_MODE=ui`: Streamlit alive).
5. Commit theo từng task. User tự thao tác phần tài khoản (Space/Neon/Groq/secrets)
   theo `deploy/README.md`; smoke test live sau khi CD chạy.

## 7. Tiêu chí hoàn thành (Definition of Done)

- [ ] `GET /health` trả 200 với test pytest xanh.
- [ ] `docker build` thành công; container chạy được cả 2 `APP_MODE`.
- [ ] `pytest tests/` toàn bộ xanh (không regression).
- [ ] `cd.yml` có 2 job deploy HF, skip-gracefully khi thiếu secret.
- [ ] README front-matter hợp lệ cho HF Space.
- [ ] `deploy/README.md` đủ hướng dẫn để user tự cấu hình hạ tầng.
- [ ] Toàn bộ thay đổi đã commit lên `main`.

## 8. Những gì ngoài phạm vi

- Không đổi logic `InspectionService`, agent graph, RAG, PLC bridge.
- Không đổi Azure scripts (`deploy/terraform`, `deploy-azure.sh`) — giữ nguyên.
- Không thêm auth cho UI, không siết CORS (ghi nhận là việc tương lai).
