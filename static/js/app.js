/* ==========================================================================
   DocuMind AI – shared dashboard JS (theme, toasts, API client, helpers).
   Vanilla JS only. Loaded on every page via base.html.
   ========================================================================== */
"use strict";

const API = {
  base: window.API_BASE || "http://localhost:8000",
  prefix: "/api/v1",

  async request(path, options = {}) {
    const url = `${this.base}${this.prefix}${path}`;
    try {
      const res = await fetch(url, { headers: {}, ...options });
      const text = await res.text();
      const data = text ? JSON.parse(text) : null;
      if (!res.ok) {
        const detail = data && data.detail
          ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
          : `HTTP ${res.status}`;
        throw new Error(detail);
      }
      return data;
    } catch (err) {
      if (err instanceof TypeError) throw new Error("Cannot reach backend API. Is FastAPI running?");
      throw err;
    }
  },

  get: (p) => API.request(p),
  post: (p, body) => API.request(p, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }),
  del: (p) => API.request(p, { method: "DELETE" }),
  upload: (p, formData) => API.request(p, { method: "POST", body: formData }),
};

/* ------------------------------------------------------------------ theme */
(function initTheme() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const root = document.documentElement;
    root.classList.toggle("dark");
    localStorage.setItem("documind-theme", root.classList.contains("dark") ? "dark" : "light");
    window.dispatchEvent(new CustomEvent("themechange"));
  });
})();

/* mobile menu */
document.addEventListener("DOMContentLoaded", () => {
  const burger = document.getElementById("mobile-menu-btn");
  const nav = document.getElementById("mobile-nav");
  if (burger && nav) burger.addEventListener("click", () => nav.classList.toggle("hidden"));
  checkBackendHealth();
});

/* --------------------------------------------------------------- backend */
async function checkBackendHealth() {
  const dot = document.getElementById("status-dot");
  const txt = document.getElementById("status-text");
  if (!dot || !txt) return;
  try {
    const res = await fetch(`${API.base}/health`);
    const data = await res.json();
    const ok = data.status === "ok";
    dot.className = `h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-amber-500"}`;
    txt.textContent = ok
      ? `API online · ${data.faiss_vectors} vectors`
      : `API degraded (${data.database})`;
  } catch {
    dot.className = "h-2 w-2 rounded-full bg-rose-500";
    txt.textContent = "API offline";
  }
}

/* ------------------------------------------------------------------ toast */
function toast(message, type = "info", duration = 4200) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const styles = {
    success: "border-emerald-500/40 bg-emerald-50 dark:bg-emerald-950/70 text-emerald-800 dark:text-emerald-200",
    error:   "border-rose-500/40 bg-rose-50 dark:bg-rose-950/70 text-rose-800 dark:text-rose-200",
    info:    "border-brand-500/40 bg-indigo-50 dark:bg-indigo-950/70 text-indigo-800 dark:text-indigo-200",
    warn:    "border-amber-500/40 bg-amber-50 dark:bg-amber-950/70 text-amber-800 dark:text-amber-200",
  };
  const icons = { success: "✓", error: "✕", info: "ℹ", warn: "⚠" };
  const el = document.createElement("div");
  el.className = `toast flex items-start gap-3 rounded-xl border px-4 py-3 shadow-lg backdrop-blur text-sm font-medium ${styles[type] || styles.info}`;
  el.innerHTML = `<span class="mt-0.5 font-bold">${icons[type] || "ℹ"}</span><div class="flex-1">${escapeHtml(message)}</div>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity .3s, transform .3s";
    el.style.opacity = "0";
    el.style.transform = "translateY(6px)";
    setTimeout(() => el.remove(), 320);
  }, duration);
}

/* --------------------------------------------------------------- helpers */
function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function timeAgo(dateStr) {
  const diff = (Date.now() - new Date(dateStr).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/* Very small markdown renderer for answers (bold, bullets, citations). */
function renderAnswer(text) {
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Convert [Source N] tokens into styled citation chips.
  html = html.replace(/\[Source (\d+)\]/g,
    '<span class="cite-chip inline-flex items-center rounded bg-brand-600/10 dark:bg-brand-500/25 text-brand-700 dark:text-indigo-300 border border-brand-600/30 px-1.5 py-0.5 mx-0.5 align-middle text-[11px] font-semibold" data-source="$1">S$1</span>');
  // Bullets + paragraphs.
  const lines = html.split("\n");
  let out = "", inList = false;
  for (const line of lines) {
    const m = line.match(/^\s*[-•*]\s+(.*)/);
    if (m) {
      if (!inList) { out += "<ul>"; inList = true; }
      out += `<li>${m[1]}</li>`;
    } else {
      if (inList) { out += "</ul>"; inList = false; }
      if (line.trim()) out += `<p>${line}</p>`;
    }
  }
  if (inList) out += "</ul>";
  return out;
}

function statusBadge(status) {
  const map = {
    completed: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300",
    processing: "bg-amber-100 text-amber-700 dark:bg-amber-900/50 dark:text-amber-300",
    pending: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
    failed: "bg-rose-100 text-rose-700 dark:bg-rose-900/50 dark:text-rose-300",
  };
  return `<span class="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold capitalize ${map[status] || map.pending}">${status}</span>`;
}

function categoryBadge(category) {
  if (!category) return '<span class="text-xs text-slate-400">—</span>';
  const colors = {
    Invoice: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 border-emerald-500/30",
    Contract: "bg-sky-500/15 text-sky-600 dark:text-sky-300 border-sky-500/30",
    Report: "bg-violet-500/15 text-violet-600 dark:text-violet-300 border-violet-500/30",
    Resume: "bg-pink-500/15 text-pink-600 dark:text-pink-300 border-pink-500/30",
    "Research Paper": "bg-amber-500/15 text-amber-600 dark:text-amber-300 border-amber-500/30",
    Policy: "bg-teal-500/15 text-teal-600 dark:text-teal-300 border-teal-500/30",
    Other: "bg-slate-500/15 text-slate-600 dark:text-slate-300 border-slate-500/30",
  };
  const cls = colors[category] || colors.Other;
  return `<span class="inline-flex rounded-md border px-2 py-0.5 text-[11px] font-semibold ${cls}">${escapeHtml(category)}</span>`;
}

/* Chart.js theme helper: re-themes charts when dark mode toggles. */
function chartColors() {
  const dark = document.documentElement.classList.contains("dark");
  return {
    grid: dark ? "rgba(148,163,184,.12)" : "rgba(100,116,139,.12)",
    text: dark ? "#cbd5e1" : "#475569",
  };
}
