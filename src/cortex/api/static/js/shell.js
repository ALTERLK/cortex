/* The app shell shared by both pages: brand, navigation, theme and language
 * switches, and the server status line. Each page calls mountShell() and
 * gets back the element where its own sidebar content goes.
 */

import { getJSON } from "./api.js";
import { applyI18n, getLang, setLang, t } from "./i18n.js";
import { h, icon, ICONS, store } from "./ui.js";

function currentTheme() {
  const chosen = document.documentElement.dataset.theme;
  if (chosen) return chosen;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  store.set("theme", theme);
  document.getElementById("theme-btn").replaceChildren(icon(theme === "dark" ? "sun" : "moon"));
}

/**
 * Build the sidebar into #sidebar and wire the mobile drawer.
 * @param {"chat"|"eval"} active  which nav entry to highlight
 * @returns {{body: HTMLElement, stats: Promise<object|null>}}
 */
export function mountShell(active) {
  const app = document.querySelector(".app");
  const sidebar = document.getElementById("sidebar");
  const body = h("div", { class: "sidebar-body" });

  const navLink = (key, href, iconName) =>
    h("a", { class: `nav-link${active === key ? " active" : ""}`, href },
      icon(iconName), h("span", { dataset: { i18n: `nav.${key}` } }));

  sidebar.replaceChildren(
    h("div", { class: "brand" },
      h("span", { class: "brand-mark", html: ICONS.logo }),
      h("span", { class: "brand-name", text: "Cortex" }),
      h("span", { class: "brand-version mono", id: "version" })),
    h("nav", { class: "nav" }, navLink("chat", "/", "chat"), navLink("eval", "/eval", "chart")),
    body,
    h("div", { class: "sidebar-foot" },
      h("div", { class: "status" },
        h("span", { class: "status-dot", id: "status-dot" }),
        h("span", { class: "status-text", id: "status-text", dataset: { i18n: "status.connecting" } })),
      h("div", { class: "foot-actions" },
        h("button", {
          class: "icon-btn text", id: "lang-btn", type: "button",
          dataset: { i18n: "lang.toggle" },
          onclick: () => setLang(getLang() === "zh" ? "en" : "zh"),
        }),
        h("button", {
          class: "icon-btn", id: "theme-btn", type: "button",
          dataset: { i18nTitle: "theme.toggle" },
          onclick: () => setTheme(currentTheme() === "dark" ? "light" : "dark"),
        }))),
  );
  document.getElementById("theme-btn").append(icon(currentTheme() === "dark" ? "sun" : "moon"));

  // Mobile: the sidebar is a drawer opened from the top bar.
  document.getElementById("menu-btn")?.addEventListener("click", () => app.classList.add("nav-open"));
  document.getElementById("scrim")?.addEventListener("click", () => {
    app.classList.remove("nav-open", "inspector-open");
    document.dispatchEvent(new CustomEvent("scrimclick"));
  });

  applyI18n();
  return { body, stats: loadStatus() };
}

/* /stats doubles as the health check: it proves the index is loaded. */
async function loadStatus() {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  try {
    const stats = await getJSON("/stats");
    dot.className = "status-dot ok";
    text.removeAttribute("data-i18n");
    text.textContent = stats.llm_model;
    text.title = stats.llm_model;
    document.getElementById("version").textContent = `v${stats.version}`;
    return stats;
  } catch {
    dot.className = "status-dot err";
    text.dataset.i18n = "status.offline";
    text.textContent = t("status.offline");
    return null;
  }
}
