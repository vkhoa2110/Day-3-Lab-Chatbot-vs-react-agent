import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.vinpearl_agent import VinpearlRoomAgent
from src.core.openai_provider import OpenAIProvider

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


def build_agent() -> VinpearlRoomAgent:
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")

    provider = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    model_name = os.getenv("DEFAULT_MODEL", "gpt-4o")
    api_key = os.getenv("OPENAI_API_KEY")

    if provider == "openai" and api_key:
        return VinpearlRoomAgent(llm=OpenAIProvider(model_name=model_name, api_key=api_key))

    return VinpearlRoomAgent()


INDEX_HTML = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vinpearl Room Agent</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #607080;
      --line: #dce2e8;
      --accent: #0b6b5d;
      --accent-2: #b0842f;
      --soft: #edf7f5;
      --danger: #9d2f2f;
      --shadow: 0 12px 32px rgba(23, 32, 42, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
    }

    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 48px;
    }

    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, var(--accent), #123f58);
      color: white;
      font-weight: 800;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }

    .subtle {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    label {
      display: block;
      color: #2f3d4a;
      font-weight: 650;
      font-size: 13px;
      margin: 0 0 6px;
    }

    .field { margin-bottom: 14px; }

    input, select, button, textarea {
      font: inherit;
      letter-spacing: 0;
    }

    input, select {
      width: 100%;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 11px;
      background: white;
      color: var(--ink);
      outline: none;
    }

    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(11, 107, 93, 0.12);
    }

    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .hint-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .chip {
      border: 1px solid var(--line);
      background: white;
      color: #2d3a44;
      height: 32px;
      border-radius: 999px;
      padding: 0 12px;
      cursor: pointer;
    }

    .chip:hover { border-color: var(--accent); color: var(--accent); }

    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: 100vh;
    }

    .topbar {
      height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      backdrop-filter: blur(10px);
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
    }

    .messages {
      overflow: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .message {
      max-width: 980px;
      display: grid;
      gap: 10px;
    }

    .message.user {
      align-self: flex-end;
      max-width: 720px;
    }

    .bubble {
      white-space: pre-wrap;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 15px;
      background: var(--panel);
      box-shadow: 0 6px 18px rgba(23, 32, 42, 0.04);
    }

    .user .bubble {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }

    .room-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }

    .room-card, .location-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      box-shadow: var(--shadow);
    }

    .room-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 8px;
    }

    .room-title {
      font-weight: 760;
      font-size: 16px;
      line-height: 1.25;
    }

    .price {
      color: var(--accent);
      font-weight: 800;
      white-space: nowrap;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0;
    }

    .badge {
      border-radius: 999px;
      background: var(--soft);
      color: #17483f;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 650;
    }

    .room-card dl {
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 6px 10px;
      margin: 10px 0 0;
      color: #2d3a44;
      font-size: 13px;
    }

    dt { color: var(--muted); }
    dd { margin: 0; }

    .option-list {
      display: grid;
      gap: 8px;
    }

    .location-card {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }

    .location-card strong { display: block; }
    .location-card span { color: var(--muted); font-size: 13px; }

    .select-location {
      height: 34px;
      border-radius: 8px;
      border: 1px solid var(--accent);
      color: var(--accent);
      background: white;
      padding: 0 12px;
      cursor: pointer;
      white-space: nowrap;
    }

    details.trace {
      border: 1px dashed #b8c3cc;
      border-radius: 8px;
      background: #fbfcfd;
      padding: 9px 11px;
      color: var(--muted);
      font-size: 12px;
    }

    details.trace pre {
      overflow: auto;
      white-space: pre-wrap;
      color: #26323c;
      margin: 10px 0 0;
      font-size: 12px;
    }

    .composer {
      border-top: 1px solid var(--line);
      background: white;
      padding: 14px 24px;
    }

    .composer-row {
      max-width: 1080px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 10px;
      align-items: end;
    }

    textarea {
      resize: none;
      min-height: 46px;
      max-height: 140px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      outline: none;
    }

    .primary, .secondary {
      height: 46px;
      border-radius: 8px;
      padding: 0 16px;
      border: 1px solid transparent;
      cursor: pointer;
      font-weight: 700;
    }

    .primary { background: var(--accent); color: white; }
    .secondary { background: #eef2f5; color: #26323c; }
    .primary:disabled { opacity: 0.55; cursor: wait; }

    .sidebar-search {
      width: 100%;
      margin-top: 2px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .empty {
      max-width: 760px;
      margin: auto;
      text-align: center;
      color: var(--muted);
    }

    .empty h2 {
      color: var(--ink);
      margin: 0 0 8px;
      font-size: 24px;
    }

    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main { height: auto; min-height: 70vh; }
      .messages { min-height: 58vh; }
      .composer-row { grid-template-columns: 1fr; }
      .primary, .secondary { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <div class="brand-mark">VP</div>
        <div>
          <h1>Vinpearl Room Agent</h1>
          <p class="subtle">Dataset demo 01/06/2026-31/08/2026</p>
        </div>
      </div>

      <section>
        <div class="field">
          <label for="hotelSelect">Cơ sở Vinpearl</label>
          <select id="hotelSelect">
            <option value="">Chọn cơ sở</option>
          </select>
        </div>
        <div class="field">
          <label for="locationInput">Địa chỉ hoặc tên cơ sở</label>
          <input id="locationInput" placeholder="Bãi Dài, Gành Dầu">
        </div>
        <div class="split">
          <div class="field">
            <label for="checkinInput">Nhận phòng</label>
            <input id="checkinInput" type="date" min="2026-06-01" max="2026-08-31">
          </div>
          <div class="field">
            <label for="checkoutInput">Trả phòng</label>
            <input id="checkoutInput" type="date" min="2026-06-02" max="2026-09-01">
          </div>
        </div>
        <div class="field">
          <label for="guestsInput">Số khách</label>
          <input id="guestsInput" type="number" min="1" max="12" value="2">
        </div>
        <button class="primary sidebar-search" id="sidebarSearchButton" type="button">Tìm phòng</button>
      </section>

      <section>
        <label>Gợi ý nhanh</label>
        <div class="hint-list" id="quickHints"></div>
      </section>
    </aside>

    <main>
      <header class="topbar">
        <div>
          <strong>Tìm phòng trống Vinpearl</strong>
          <p class="subtle">Chỉ trả lời các yêu cầu về phòng trống trong dataset demo.</p>
        </div>
        <div class="status"><span class="dot"></span><span id="statusText">Sẵn sàng</span></div>
      </header>

      <section class="messages" id="messages">
        <div class="empty" id="emptyState">
          <h2>Nhập cơ sở Vinpearl và ngày lưu trú</h2>
          <p>Ví dụ: "Tìm phòng tại Bãi Dài, Gành Dầu từ 15/07 đến 18/07 cho 2 khách".</p>
        </div>
      </section>

      <footer class="composer">
        <div class="composer-row">
          <textarea id="messageInput" rows="1" placeholder="Nhập câu hỏi về phòng trống Vinpearl"></textarea>
          <button class="secondary" id="clearButton">Xóa</button>
          <button class="primary" id="sendButton">Gửi</button>
        </div>
      </footer>
    </main>
  </div>

  <script>
    const messagesEl = document.getElementById("messages");
    const emptyState = document.getElementById("emptyState");
    const statusText = document.getElementById("statusText");
    const sendButton = document.getElementById("sendButton");
    const sidebarSearchButton = document.getElementById("sidebarSearchButton");
    const clearButton = document.getElementById("clearButton");
    const messageInput = document.getElementById("messageInput");
    const hotelSelect = document.getElementById("hotelSelect");
    const locationInput = document.getElementById("locationInput");
    const checkinInput = document.getElementById("checkinInput");
    const checkoutInput = document.getElementById("checkoutInput");
    const guestsInput = document.getElementById("guestsInput");
    const quickHints = document.getElementById("quickHints");

    let hotels = [];
    let chatContext = {};

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function selectedHotel() {
      return hotels.find((hotel) => hotel.hotel_id === hotelSelect.value);
    }

    function collectContext() {
      const hotel = selectedHotel();
      const location = locationInput.value.trim() || (hotel ? hotel.label : "");
      return {
        ...chatContext,
        hotel_id: hotel ? hotel.hotel_id : chatContext.hotel_id,
        location,
        checkin: checkinInput.value || chatContext.checkin,
        checkout: checkoutInput.value || chatContext.checkout,
        guests: Number(guestsInput.value || chatContext.guests || 2)
      };
    }

    function formatDateForMessage(value) {
      if (!value) return "";
      const [year, month, day] = value.split("-");
      return `${day}/${month}/${year}`;
    }

    function buildSearchMessage(context) {
      const hotel = selectedHotel();
      const location = context.location || (hotel ? hotel.label : "");
      const parts = ["Tim phong Vinpearl"];
      if (location) parts.push(`tai ${location}`);
      if (context.checkin && context.checkout) {
        parts.push(`tu ${formatDateForMessage(context.checkin)} den ${formatDateForMessage(context.checkout)}`);
      }
      if (context.guests) parts.push(`cho ${context.guests} khach`);
      return parts.join(" ");
    }

    function applyContext(context) {
      chatContext = { ...chatContext, ...(context || {}) };
      if (chatContext.hotel_id) hotelSelect.value = chatContext.hotel_id;
      if (chatContext.location) locationInput.value = chatContext.location;
      if (chatContext.checkin) checkinInput.value = chatContext.checkin;
      if (chatContext.checkout) checkoutInput.value = chatContext.checkout;
      if (chatContext.guests) guestsInput.value = chatContext.guests;
    }

    function addMessage(role, text, payload = {}) {
      emptyState?.remove();
      const article = document.createElement("article");
      article.className = `message ${role}`;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      article.appendChild(bubble);

      if (payload.location_options?.length) {
        article.appendChild(renderLocations(payload.location_options));
      }
      if (payload.room_cards?.length) {
        article.appendChild(renderRoomCards(payload.room_cards));
      }
      if (payload.trace_text) {
        const details = document.createElement("details");
        details.className = "trace";
        details.innerHTML = `<summary>ReAct trace</summary><pre>${escapeHtml(payload.trace_text)}</pre>`;
        article.appendChild(details);
      }

      messagesEl.appendChild(article);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderLocations(options) {
      const wrap = document.createElement("div");
      wrap.className = "option-list";
      options.forEach((item) => {
        const card = document.createElement("div");
        card.className = "location-card";
        const text = document.createElement("div");
        text.innerHTML = `<strong>${escapeHtml(item.hotel_name)}</strong><span>${escapeHtml(item.address)} · ${escapeHtml(item.region)}</span>`;
        const button = document.createElement("button");
        button.className = "select-location";
        button.type = "button";
        button.textContent = "Chọn";
        button.addEventListener("click", () => {
          chatContext.hotel_id = item.hotel_id;
          chatContext.location = `${item.hotel_name} - ${item.address}`;
          hotelSelect.value = item.hotel_id;
          locationInput.value = chatContext.location;
          messageInput.focus();
        });
        card.append(text, button);
        wrap.appendChild(card);
      });
      return wrap;
    }

    function renderRoomCards(cards) {
      const grid = document.createElement("div");
      grid.className = "room-grid";
      cards.forEach((room) => {
        const card = document.createElement("section");
        card.className = "room-card";
        const promotions = (room.promotions || []).map((item) => item.name_vi).join(", ");
        const amenities = (room.amenities || []).slice(0, 5).join(", ");
        const beds = (room.bed_options || []).join(", ");
        card.innerHTML = `
          <div class="room-head">
            <div class="room-title">${escapeHtml(room.room_name)}</div>
            <div class="price">${escapeHtml(room.total_display)}</div>
          </div>
          <div class="meta">
            <span class="badge">Còn ${escapeHtml(room.min_available_rooms)} phòng</span>
            <span class="badge">${escapeHtml(room.area_sqm)}m2</span>
            <span class="badge">Tối đa ${escapeHtml(room.max_guests)} khách</span>
          </div>
          <dl>
            <dt>Giường</dt><dd>${escapeHtml(beds)}</dd>
            <dt>Bữa ăn</dt><dd>${escapeHtml(room.meal_plan)}</dd>
            <dt>Chính sách</dt><dd>${escapeHtml(room.cancellation_policy)}</dd>
            <dt>Tiện ích</dt><dd>${escapeHtml(amenities)}</dd>
            <dt>Ưu đãi</dt><dd>${escapeHtml(promotions || "Không áp dụng")}</dd>
          </dl>
        `;
        grid.appendChild(card);
      });
      return grid;
    }

    async function sendMessage(text = "") {
      const message = (text || messageInput.value).trim();
      const context = collectContext();
      const fallbackMessage = "Kiểm tra phòng trống Vinpearl";
      if (!message && !context.location && !context.hotel_id) return;

      addMessage("user", message || fallbackMessage);
      messageInput.value = "";
      sendButton.disabled = true;
      sidebarSearchButton.disabled = true;
      statusText.textContent = "Đang xử lý";

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: message || fallbackMessage, context })
        });
        const payload = await response.json();
        applyContext(payload.context);
        addMessage("assistant", payload.answer, payload);
      } catch (error) {
        addMessage("assistant", "Không gọi được agent nội bộ. Vui lòng kiểm tra server.");
      } finally {
        sendButton.disabled = false;
        sidebarSearchButton.disabled = false;
        statusText.textContent = "Sẵn sàng";
        messageInput.focus();
      }
    }

    function sendSidebarSearch() {
      const context = collectContext();
      if (!context.location && !context.hotel_id) {
        addMessage("assistant", "Vui long chon hoac nhap co so Vinpearl truoc khi tim phong.");
        return;
      }
      if (!context.checkin || !context.checkout) {
        addMessage("assistant", "Vui long chon ngay nhan phong va ngay tra phong.");
        return;
      }
      sendMessage(buildSearchMessage(context));
    }

    async function loadLocations() {
      const response = await fetch("/api/locations");
      const payload = await response.json();
      hotels = payload.hotels || [];
      hotels.forEach((hotel) => {
        const option = document.createElement("option");
        option.value = hotel.hotel_id;
        option.textContent = hotel.label;
        hotelSelect.appendChild(option);
      });

      hotels.slice(0, 6).forEach((hotel) => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.type = "button";
        chip.textContent = hotel.region;
        chip.title = hotel.label;
        chip.addEventListener("click", () => {
          hotelSelect.value = hotel.hotel_id;
          locationInput.value = hotel.label;
          chatContext.hotel_id = hotel.hotel_id;
          chatContext.location = hotel.label;
        });
        quickHints.appendChild(chip);
      });
    }

    hotelSelect.addEventListener("change", () => {
      const hotel = selectedHotel();
      if (hotel) {
        locationInput.value = hotel.label;
        chatContext.hotel_id = hotel.hotel_id;
        chatContext.location = hotel.label;
      }
    });

    sendButton.addEventListener("click", () => sendMessage());
    sidebarSearchButton.addEventListener("click", () => sendSidebarSearch());
    clearButton.addEventListener("click", () => {
      messagesEl.innerHTML = "";
      messagesEl.appendChild(emptyState);
      chatContext = {};
      hotelSelect.value = "";
      locationInput.value = "";
      checkinInput.value = "";
      checkoutInput.value = "";
      guestsInput.value = "2";
      messageInput.value = "";
    });
    messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    loadLocations();
  </script>
</body>
</html>
"""


class VinpearlRequestHandler(BaseHTTPRequestHandler):
    agent = build_agent()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(INDEX_HTML)
            return
        if path == "/api/locations":
            self._send_json(
                {
                    "hotels": self.agent.locations_for_ui(),
                    "date_range": {
                        "from": self.agent.kb.min_date.isoformat(),
                        "to": self.agent.kb.max_date.isoformat(),
                    },
                }
            )
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/chat":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            message = payload.get("message", "")
            context = payload.get("context", {})
            self._send_json(self.agent.respond(message, context=context))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Vinpearl room agent UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VinpearlRequestHandler)
    print(f"Vinpearl Room Agent running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
