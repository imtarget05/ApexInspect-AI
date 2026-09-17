// ApexInspect AI — static operator dashboard (no framework, no build step).
// Nói chuyện trực tiếp với FastAPI: /health, /api/v1/inspections,
// /api/v1/tickets/recent, /api/v1/mes/action.
const DEFAULT_API = "https://apexinspect-api.onrender.com";

const $ = (id) => document.getElementById(id);

function apiBase() {
  return (localStorage.getItem("apex_api_base") || DEFAULT_API).replace(/\/$/, "");
}
function apiKey() {
  return localStorage.getItem("apex_api_key") || "";
}

async function checkHealth() {
  const banner = $("health-banner");
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 20000);
    const r = await fetch(apiBase() + "/health", { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) throw new Error("HTTP " + r.status);
    const body = await r.json();
    banner.className = "banner banner-ok";
    banner.textContent = `🟢 API online — ${body.service} v${body.version}`;
  } catch (e) {
    banner.className = "banner banner-bad";
    banner.textContent = "🔴 API chưa phản hồi (Render free có thể đang sleep — đợi ~1 phút rồi Refresh). Chi tiết: " + e.message;
  }
  $("foot-api").textContent = apiBase();
}

async function sendInspection() {
  const out = $("inspect-result");
  const payload = {
    line_id: $("insp-line").value.trim() || "SMT-LINE-01",
    is_defective: $("insp-defective").checked,
    defect_classes: [$("insp-class").value.trim() || "short_circuit"],
    confidence_scores: [parseFloat($("insp-conf").value) || 0.94],
    bounding_boxes: [[160, 220, 240, 280]],
    inference_time_ms: parseFloat($("insp-ms").value) || 25.0,
  };
  out.textContent = "⏳ Đang gửi…";
  try {
    const r = await fetch(apiBase() + "/api/v1/inspections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await r.json();
    out.textContent = JSON.stringify(body, null, 2);
    if (body.ticket_id) {
      $("mes-ticket").value = body.ticket_id;
      loadTickets();
    }
  } catch (e) {
    out.textContent = "❌ Lỗi: " + e.message;
  }
}

async function loadTickets() {
  const tbody = $("tickets-body");
  try {
    const r = await fetch(apiBase() + "/api/v1/tickets/recent");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const tickets = await r.json();
    if (!tickets.length) {
      tbody.innerHTML = '<tr><td colspan="5">Chưa có ticket.</td></tr>';
      return;
    }
    tbody.innerHTML = tickets.map((t) => `<tr>
      <td><a href="#" data-ticket="${t.ticket_id}" class="pick">${t.ticket_id}</a></td>
      <td>${t.line_id}</td><td>${t.severity}</td><td>${t.action_type}</td><td>${t.status}</td>
    </tr>`).join("");
    tbody.querySelectorAll(".pick").forEach((a) => a.addEventListener("click", (ev) => {
      ev.preventDefault();
      $("mes-ticket").value = a.dataset.ticket;
    }));
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5">❌ ${e.message}</td></tr>`;
  }
}

async function sendMes() {
  const out = $("mes-result");
  const key = apiKey();
  if (!key) {
    out.textContent = "❌ Chưa có X-API-KEY — nhập ở mục Cấu hình kết nối rồi Lưu.";
    return;
  }
  const payload = {
    ticket_id: $("mes-ticket").value.trim(),
    action: $("mes-action").value,
    approved_by: $("mes-by").value.trim() || "supervisor_on_duty",
  };
  if (!payload.ticket_id) {
    out.textContent = "❌ Chưa nhập ticket_id.";
    return;
  }
  out.textContent = "⏳ Đang thực thi…";
  try {
    const r = await fetch(apiBase() + "/api/v1/mes/action", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-KEY": key },
      body: JSON.stringify(payload),
    });
    const body = await r.json();
    out.textContent = `HTTP ${r.status}\n` + JSON.stringify(body, null, 2);
    loadTickets();
  } catch (e) {
    out.textContent = "❌ Lỗi: " + e.message;
  }
}

function loadConfig() {
  $("api-base").value = apiBase();
  // không đổ key ra màn hình — chỉ placeholder trạng thái
  $("api-key").placeholder = apiKey() ? "đã lưu •••••• (nhập mới để thay)" : "nhập key MES…";
}

document.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  checkHealth();
  loadTickets();
  $("btn-save").addEventListener("click", () => {
    const base = $("api-base").value.trim();
    if (base) localStorage.setItem("apex_api_base", base.replace(/\/$/, ""));
    if ($("api-key").value) localStorage.setItem("apex_api_key", $("api-key").value.trim());
    $("api-key").value = "";
    $("save-hint").textContent = "✅ Đã lưu.";
    loadConfig();
    checkHealth();
  });
  $("btn-inspect").addEventListener("click", sendInspection);
  $("btn-tickets").addEventListener("click", loadTickets);
  $("btn-mes").addEventListener("click", sendMes);
});
