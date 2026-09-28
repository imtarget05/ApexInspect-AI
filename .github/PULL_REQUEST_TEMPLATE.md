# Pull Request — review checklist (aiqon: PR/code-review workflow evidence)

> Không merge khi CI đỏ. Reviewer kiểm từng ô trước khi Approve.

## Mô tả
- Vấn đề / JD requirement liên quan:
- Thay đổi chính (file/module):

## Checklist tác giả
- [ ] `pytest` / `npm test` / `dotnet test` pass local (dán log hoặc link CI run)
- [ ] Không commit secret (gitleaks gate trong `ci.yml` phải xanh)
- [ ] API thay đổi → cập nhật OpenAPI/spec + `docs/testing/TRACEABILITY_MATRIX.md` nếu có
- [ ] Endpoint mới → có test (happy + 400/404 + auth) và được `route-coverage`/`api-contract` bao phủ
- [ ] Không log PII / prompt body (gateway chỉ log counts + sha — xem `llm-gateway/server.py:TelemetryLog`)
- [ ] Docker build còn chạy (`docker compose config` + smoke `/health`)

## Checklist reviewer
- [ ] Hiểu data-flow thay đổi (UI → API → DB/data) và đồng ý với trade-off
- [ ] Error handling + status code đúng (400 validation / 401-403 auth / 404 / 409 conflict / 503 degraded)
- [ ] Không có technology nhồi nhét ngoài requirement (YAGNI — ADR nếu thêm dependency)

## Evidence sau merge
- Link CI run xanh:
- Screenshot/demo (nếu đổi UI/flow):
