/* Small DOM and formatting helpers shared by every module. No dependencies. */

/**
 * Create an element. `props` may hold `class`, `text`, `html`, `dataset`,
 * event handlers (`onclick`…), or plain attributes. Children may be nodes,
 * strings, or null (skipped).
 *
 * Text always goes in through `text` / string children, which use
 * textContent — only trusted, static markup (the icons below) uses `html`.
 */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "html") el.innerHTML = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child);
  }
  return el;
}

const svg = (body, size = 16) =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" ` +
  `stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

/* Static, trusted SVG markup. */
export const ICONS = {
  logo: `<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
    <circle cx="12" cy="12" r="3.2" fill="currentColor"/>
    <circle cx="12" cy="12" r="7.5" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.55"/>
    <circle cx="12" cy="12" r="11" fill="none" stroke="currentColor" stroke-width="1" opacity="0.25"/></svg>`,
  chat: svg('<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>'),
  chart: svg('<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'),
  plus: svg('<path d="M12 5v14M5 12h14"/>'),
  search: svg('<circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/>', 14),
  book: svg('<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5"/>', 14),
  list: svg('<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"/>', 14),
  bolt: svg('<path d="M13 3 5 14h6l-1 7 8-11h-6z"/>', 14),
  send: svg('<path d="M5 12h14M13 6l6 6-6 6"/>', 17),
  stop: `<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/></svg>`,
  close: svg('<path d="M6 6l12 12M18 6 6 18"/>'),
  menu: svg('<path d="M4 7h16M4 12h16M4 17h16"/>', 18),
  panel: svg('<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M15 4v16"/>', 18),
  sun: svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
  moon: svg('<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>'),
  caret: svg('<path d="m9 6 6 6-6 6"/>', 12),
  trash: svg('<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>', 13),
  check: svg('<path d="m5 12.5 4.5 4.5L19 7.5"/>', 13),
  alert: svg('<path d="M12 9v4M12 17h.01M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>', 13),
  route: svg('<circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="6" r="2.5"/><path d="M8.5 18H15a3 3 0 0 0 3-3V8.5"/>', 13),
};

export const icon = (name, cls = "") => h("span", { class: `ico ${cls}`.trim(), html: ICONS[name] });

export const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
export const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
export const fmtCost = (usd) => (usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(3)}`);
export const fmtPct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;

/* localStorage that never throws (private windows, blocked storage). */
export const store = {
  get(key, fallback = null) {
    try {
      const value = localStorage.getItem(`cortex.${key}`);
      return value == null ? fallback : JSON.parse(value);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`cortex.${key}`, JSON.stringify(value));
    } catch {
      /* preference simply does not persist */
    }
  },
};

let toastTimer;
export function toast(message) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 4200);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
