/* The evidence panel: for one answer, the passages behind it (with where
 * each ranked) and a waterfall of the steps the request took.
 */

import { t } from "./i18n.js";
import { fmtCost, fmtInt, fmtMs, h, icon } from "./ui.js";

const CANDIDATES = 20; // each retriever contributes its top 20 before fusion

export function mountInspector() {
  const app = document.querySelector(".app");
  const panel = document.getElementById("inspector");
  let message = null;
  let tab = "sources";
  let focus = null;

  const body = h("div", { class: "inspector-body" });
  const sourcesTab = h("button", { class: "tab", type: "button", role: "tab", onclick: () => select("sources") });
  const traceTab = h("button", { class: "tab", type: "button", role: "tab", onclick: () => select("trace") });

  panel.replaceChildren(
    h("div", { class: "inspector-head" },
      h("h2", { dataset: { i18n: "insp.title" }, text: t("insp.title") }),
      h("button", { class: "icon-btn", type: "button", "aria-label": "Close", onclick: close }, icon("close"))),
    h("div", { class: "tabs", role: "tablist" }, sourcesTab, traceTab),
    body,
  );

  function select(next) {
    tab = next;
    render();
  }

  function open() {
    panel.hidden = false;
    app.classList.add("has-inspector", "inspector-open");
  }

  function close() {
    panel.hidden = true;
    app.classList.remove("has-inspector", "inspector-open");
    document.querySelectorAll(".cite.on").forEach((chip) => chip.classList.remove("on"));
  }

  /** Show the evidence for `msg`; `cite` scrolls to and highlights source N. */
  function show(msg, { cite = null, view = null } = {}) {
    message = msg;
    focus = cite;
    if (view) tab = view;
    else if (cite != null) tab = "sources";
    open();
    render();
  }

  function render() {
    sourcesTab.setAttribute("aria-selected", tab === "sources");
    traceTab.setAttribute("aria-selected", tab === "trace");
    sourcesTab.replaceChildren(t("insp.sources"));
    if (message) sourcesTab.append(h("span", { class: "count num", text: message.sources.length }));
    traceTab.replaceChildren(t("insp.trace"));

    if (!message) {
      body.replaceChildren(h("div", { class: "inspector-empty", text: t("insp.empty") }));
      return;
    }
    body.replaceChildren(...(tab === "sources" ? renderSources(message) : renderTrace(message)));
    if (tab === "sources" && focus != null) {
      const card = body.querySelector(`[data-source="${focus}"]`);
      card?.classList.add("focus");
      card?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  // ---------------------------------------------------------------- sources

  function renderSources(msg) {
    if (!msg.sources.length) return [h("div", { class: "inspector-empty", text: t("insp.nosources") })];
    const cited = new Set(msg.done?.citations?.cited ?? []);
    const hasRerank = msg.sources.some((s) => s.trace?.rerank_score != null);
    // Fused scores are tiny (two 1/(60+rank) terms), so the meter is drawn
    // relative to the best score in this answer.
    const bestFused = Math.max(...msg.sources.map((s) => s.trace?.rrf_score ?? 0), 0);

    const legend = h("div", { class: "legend" },
      h("span", { class: "dense" }, h("i"), t("insp.dense")),
      h("span", { class: "sparse" }, h("i"), t("insp.sparse")),
      hasRerank ? h("span", { class: "rerank" }, h("i"), t("insp.rerank")) : null);

    return [legend, ...msg.sources.map((source, i) => {
      const n = i + 1;
      const trace = source.trace ?? {};
      const unused = msg.done && !cited.has(n);
      const card = h("article", {
        class: `passage${unused ? " unused" : ""}`, dataset: { source: n },
      });
      const more = h("button", {
        class: "passage-more", type: "button", text: t("insp.more"),
        onclick: () => {
          const openNow = card.classList.toggle("open");
          more.textContent = t(openNow ? "insp.less" : "insp.more");
        },
      });
      card.append(
        h("div", { class: "passage-head" },
          h("span", { class: "n", text: n }),
          h("span", { class: "path", text: source.source }),
          h("span", { class: "chunk", text: unused ? t("insp.unused") : t("insp.chunk", { n: source.chunk_index }) })),
        h("div", { class: "signals" },
          ...rankRow("dense", t("insp.dense"), trace.dense_rank),
          ...rankRow("sparse", t("insp.sparse"), trace.sparse_rank),
          ...(trace.rrf_score != null
            ? meterRow("fused", t("insp.fused"), trace.rrf_score.toFixed(4), bestFused ? trace.rrf_score / bestFused : 0)
            : []),
          ...(trace.rerank_score != null
            ? meterRow("", t("insp.rerank"), trace.rerank_score.toFixed(3), Math.max(0, Math.min(1, trace.rerank_score)))
            : [])),
        h("p", { class: "passage-text", text: source.text }),
      );
      if (source.text.length > 420) card.append(more);  // short passages need no toggle
      else card.classList.add("open");
      return card;
    })];
  }

  /* One row per retriever: twenty slots, the passage's position lit. */
  function rankRow(kind, label, rank) {
    const slots = Array.from({ length: CANDIDATES }, (_, i) => h("i", { class: i + 1 === rank ? "on" : "" }));
    return [
      h("span", { class: "name", text: label }),
      h("div", { class: `track ${kind}`, role: "img", "aria-label": rank ? `${label}: ${rank}` : `${label}: —` }, ...slots),
      rank
        ? h("span", { class: "val", text: `#${rank}` })
        : h("span", { class: "val none", text: t("insp.notfound") }),
    ];
  }

  function meterRow(kind, label, value, fraction) {
    return [
      h("span", { class: "name", text: label }),
      h("div", { class: `meter ${kind}` }, h("i", { style: `transform: scaleX(${fraction.toFixed(3)})` })),
      h("span", { class: "val", text: value }),
    ];
  }

  // ------------------------------------------------------------------ trace

  function renderTrace(msg) {
    if (!msg.spans.length) return [h("div", { class: "inspector-empty", text: t("trace.empty") })];
    const out = [];
    const done = msg.done;
    const llmCalls = msg.spans.filter((s) => s.kind === "llm").length;

    if (done) {
      out.push(h("div", { class: "trace-summary" },
        stat(fmtMs(done.latency_ms), t("trace.time")),
        stat(llmCalls, t("trace.calls")),
        stat(fmtInt(done.input_tokens + done.output_tokens), t("trace.tokens")),
        stat(fmtCost(done.cost_usd_est), t("trace.cost"))));
    }
    if (msg.rewrite) {
      out.push(h("p", { class: "trace-note" }, t("trace.rewrite"), " ", h("b", { text: msg.rewrite.rewritten })));
    }

    // Spans run one after another, so each bar starts where the last ended.
    const total = msg.spans.reduce((sum, s) => sum + (s.ms || 0), 0) || 1;
    let offset = 0;
    for (const span of msg.spans) {
      if (span.kind === "marker") {
        out.push(h("div", { class: "span-marker", text: t("trace.escalated") }));
        continue;
      }
      const left = (offset / total) * 100;
      const width = (span.ms / total) * 100;
      offset += span.ms;
      const tokens = span.input_tokens
        ? `${fmtInt(span.input_tokens)} in · ${fmtInt(span.output_tokens)} out` : "";
      const detail = [span.detail, tokens].filter(Boolean).join("  ·  ");
      out.push(h("div", { class: `span ${span.kind}` },
        h("span", { class: "name", text: spanLabel(span), title: span.name }),
        h("div", { class: "lane" }, h("div", { class: "bar", style: `left:${left.toFixed(2)}%;width:${width.toFixed(2)}%` })),
        h("span", { class: "dur", text: fmtMs(span.ms) }),
        detail ? h("div", { class: "span-detail", text: detail, title: detail }) : null));
    }
    return out;
  }

  const stat = (value, label) => h("div", {}, h("b", { text: value }), h("span", { text: label }));

  function spanLabel(span) {
    const key = `span.${span.name}`;
    const label = t(key);
    return label === key ? span.name : label;
  }

  document.addEventListener("langchange", () => {
    panel.querySelector("h2").textContent = t("insp.title");
    if (!panel.hidden) render();
  });
  document.addEventListener("scrimclick", close);

  return { show, close, refresh: (msg) => { if (!panel.hidden && msg === message) render(); } };
}
