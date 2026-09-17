import streamlit as st

# ── Lightweight launcher cho Streamlit Community Cloud ─────────────────────
# File gốc src/ui/app.py import nặng (cv2 / detector / agent) ngay tại top,
# vượt 1GB RAM free tier của Community Cloud. File này:
#   1. Chỉ import streamlit + stdlib (nhẹ, <150MB)
#   2. Nếu BACKEND_URL được cấu hình -> render Remote Console gọi API Render
#      (khuyến nghị cho Cloud: dùng chung Neon DB với API)
#   3. Nếu không -> exec file app gốc (full local pipeline, Docker/PC mạnh)
# Cách cấu hình trên Cloud: App settings -> Secrets:
#   BACKEND_URL = "https://apexinspect-api.onrender.com"
#   API_KEY     = "<khớp với API_KEY trên Render>"

_BACKEND_URL = ""
_API_KEY = ""
try:
    _BACKEND_URL = (st.secrets.get("BACKEND_URL", "") or "").strip()
    _API_KEY = (st.secrets.get("API_KEY", "") or "").strip()
except Exception:
    pass

import os as _os

_BACKEND_URL = _BACKEND_URL or _os.getenv("BACKEND_URL", "").strip()
_API_KEY = _API_KEY or _os.getenv("API_KEY", "").strip()

if _BACKEND_URL:
    import datetime as _dt
    import json as _json
    import urllib.request as _urlreq
    import urllib.error as _urlerr

    st.set_page_config(
        page_title="ApexInspect AI — Factory Console (Cloud)",
        page_icon="🏭",
        layout="wide",
    )
    st.title("🏭 ApexInspect AI — Factory Console (Cloud)")
    st.caption(f"Remote mode · API: `{_BACKEND_URL}`")

    def _api(method, path, payload=None):
        url = _BACKEND_URL.rstrip("/") + path
        data = _json.dumps(payload).encode() if payload is not None else None
        req = _urlreq.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json",
                     **({"X-API-KEY": _API_KEY} if _API_KEY else {})},
        )
        try:
            with _urlreq.urlopen(req, timeout=60) as resp:
                return resp.status, _json.loads(resp.read().decode() or "null")
        except _urlerr.HTTPError as e:
            try:
                return e.code, _json.loads(e.read().decode() or "null")
            except Exception:
                return e.code, {"detail": f"HTTP {e.code}"}
        except Exception as e:
            return 0, {"detail": f"connection failed: {e}"}

    status, health = _api("GET", "/health")
    if status == 200:
        st.success(f"✅ API online — `{health.get('service')}` v{health.get('version')}")
    else:
        st.error(f"❌ API unreachable ({health}) — Render free tier có thể đang sleep, chờ ~1 phút rồi Rerun.")
        st.stop()

    tab_live, tab_tickets, tab_mes = st.tabs(["📡 Gửi inspection", "🎫 Tickets", "🏭 MES approve"])

    with tab_live:
        st.subheader("Gửi inspection mẫu (smoke test live)")
        line_id = st.text_input("line_id", value="SMT-LINE-01")
        is_def = st.checkbox("is_defective", value=True)
        if st.button("POST /api/v1/inspections", type="primary"):
            payload = {
                "line_id": line_id,
                "is_defective": is_def,
                "defect_classes": ["short_circuit"] if is_def else [],
                "confidence_scores": [0.94] if is_def else [],
                "bounding_boxes": [[160, 220, 240, 280]] if is_def else [],
                "inference_time_ms": 25.0,
            }
            code, out = _api("POST", "/api/v1/inspections", payload)
            if code == 200:
                st.success(f"RECORDED `{out.get('inspection_id')}` · incident={out.get('incident_triggered')} · ticket={out.get('ticket_id')}")
            else:
                st.error(f"HTTP {code}: {out}")

    with tab_tickets:
        st.subheader("Tickets gần nhất")
        if st.button("GET /api/v1/tickets/recent"):
            code, out = _api("GET", "/api/v1/tickets/recent")
            if code == 200:
                if not out:
                    st.info("Chưa có ticket nào.")
                for t in out[:10]:
                    st.markdown(f"**{t.get('ticket_id')}** · `{t.get('status')}` · {t.get('action_type')} — {str(t.get('trigger_reason'))[:120]}")
            else:
                st.error(f"HTTP {code}: {out}")

    with tab_mes:
        st.subheader("MES approve / reject (cần API_KEY)")
        ticket_id = st.text_input("ticket_id", placeholder="TICK-...")
        action = st.selectbox("action", ["approve", "reject"])
        approved_by = st.text_input("approved_by", value="supervisor_cloud")
        if st.button("POST /api/v1/mes/action"):
            code, out = _api("POST", "/api/v1/mes/action",
                             {"ticket_id": ticket_id, "action": action, "approved_by": approved_by})
            if code == 200:
                st.success(f"{out.get('status')}: {out.get('message')}")
            else:
                st.error(f"HTTP {code}: {out}")

    st.divider()
    st.caption(f"⏱️ {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Full pipeline (camera sim + ONNX + agent) chạy ở bản Docker local: `streamlit run src/ui/app.py`")
else:
    # Local / Docker: chạy full app gốc (nặng, cần RAM lớn)
    _APP = _os.path.join(_os.path.dirname(__file__), "src", "ui", "app.py")
    with open(_APP, encoding="utf-8") as _f:
        _SRC = _f.read()
    exec(compile(_SRC, _APP, "exec"), {"__name__": "__main__", "__file__": _APP})
