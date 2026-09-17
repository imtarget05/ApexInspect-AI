"""Smoke tests cho static dashboard (Render Static Site, thay Streamlit).

RED-GREEN: viết trước khi có dashboard/ — phải FAIL cho tới khi implement xong.
"""
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASH = os.path.join(REPO, "dashboard")


def _read(name):
    with open(os.path.join(DASH, name), encoding="utf-8") as f:
        return f.read()


def test_dashboard_files_exist():
    assert os.path.isdir(DASH), "missing dashboard/ dir"
    for name in ("index.html", "styles.css", "app.js"):
        assert os.path.isfile(os.path.join(DASH, name)), f"missing dashboard/{name}"


def test_dashboard_no_streamlit_no_build_step():
    html, js = _read("index.html"), _read("app.js")
    assert "streamlit" not in (html + js).lower()
    assert "node_modules" not in (html + js).lower()
    # pure static: không CDN framework nặng
    for lib in ("react", "vue", "angular"):
        assert lib not in html.lower(), f"unexpected framework ref: {lib}"


def test_dashboard_talks_to_api_contract():
    html, js = _read("index.html"), _read("app.js")
    for endpoint in ("/health", "/api/v1/inspections",
                     "/api/v1/tickets/recent", "/api/v1/mes/action"):
        assert endpoint in js, f"app.js missing {endpoint}"
    assert "X-API-KEY" in js, "MES approve phải gửi X-API-KEY header"
    assert "apexinspect-api.onrender.com" in (html + js), "default API base URL"
    for hook in ("health-banner", "btn-inspect", "tickets-body", "btn-mes"):
        assert hook in html, f"index.html missing id={hook}"


def test_dashboard_api_base_configurable():
    js = _read("app.js")
    assert "localStorage" in js, "API base URL phải lưu localStorage được"


def test_render_yaml_has_static_site():
    import yaml
    with open(os.path.join(REPO, "render.yaml"), encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    names = [s.get("name") for s in doc.get("services", [])]
    assert "apexinspect-api" in names, "giữ nguyên API service"
    assert "apexinspect-dashboard" in names, "thêm static site dashboard"
    dash = next(s for s in doc["services"] if s.get("name") == "apexinspect-dashboard")
    blob = str(dash)
    assert "dashboard" in blob, "publish path phải trỏ dashboard/"


def test_streamlit_launcher_removed():
    assert not os.path.exists(os.path.join(REPO, "streamlit_app.py")), \
        "streamlit_app.py phải bị xóa (Docker local đã có src/ui/app.py full)"
