/* Evaluation dashboard.
 *
 * Everything shown here comes from GET /eval/results, which summarises the
 * runs saved by scripts/run_eval.py using the same EvalReport class as the
 * CLI. This file only lays the numbers out; it computes no metric itself
 * beyond differences between two runs.
 */

import { getJSON } from "./api.js";
import { addStrings, applyI18n, t } from "./i18n.js";
import { mountShell } from "./shell.js";
import { fmtInt, h, ICONS } from "./ui.js";

addStrings({
  en: {
    "ev.title": "Evaluation",
    "ev.lead": "Every change to retrieval or to the answering pipeline is measured on the same question set before it is kept. These are the saved runs.",
    "ev.questions": "questions",
    "ev.categories": "question types",
    "ev.documents": "documents in the corpus",
    "ev.chunks": "indexed chunks",
    "ev.empty": "No saved runs yet. Run scripts/run_eval.py to create one.",
    "ev.ret.title": "Retrieval: what each stage added",
    "ev.ret.lead": "Share of answerable questions whose source document appears in the top 5 passages (hit-rate@5). Each row adds one technique to the row above; the solid segment is what that technique gained.",
    "ev.mrr": "MRR",
    "ev.correct": "Correct",
    "ev.hit": "Hit@5",
    "ev.fixed": "{cat}: {a} → {b}",
    "ev.mode.title": "Answer mode: quality against cost",
    "ev.mode.lead": "The same questions answered three ways: one retrieval pass (RAG), the tool-use loop (Agent), and Auto, which uses the agent only when needed. Higher is better; further right costs more.",
    "ev.mode.note": "Tokens are as billed, including a fixed per-call overhead of about 4k tokens added by the API gateway, which weighs on pipelines that make several calls. An agent run is scored on every passage it retrieved, so its \"source found\" is not hit@5. The judge is the same model as the system under test.",
    "ev.take.title": "Against single-pass RAG",
    "ev.take.base": "the baseline: {score} correctness at {cost} per 1,000 questions.",
    "ev.take.line": "{delta} correctness for ×{mult} the cost.",
    "ev.take.cat": "Largest gain — {cat}: {a} → {b}",
    "ev.nav.ret": "Retrieval stages",
    "ev.nav.mode": "Quality against cost",
    "ev.nav.cat": "By question type",
    "ev.nav.q": "Every question",
    "ev.col.mode": "Pipeline",
    "ev.col.correct": "Correctness",
    "ev.col.found": "Source found",
    "ev.col.grounded": "Valid citations",
    "ev.col.calls": "LLM calls",
    "ev.col.tokens": "Tokens",
    "ev.col.latency": "Latency",
    "ev.col.cost": "Cost / 1k questions",
    "ev.col.agent": "Used agent",
    "ev.axis.cost": "cost per 1,000 questions (USD)",
    "ev.axis.correct": "correctness (1–5)",
    "ev.cat.title": "Correctness by question type",
    "ev.cat.lead": "Where each pipeline wins. A blended average hides this: a pipeline can be worse overall and still be the right choice for one kind of question.",
    "ev.cat.type": "Question type",
    "ev.q.title": "Every question",
    "ev.q.lead": "Per-question results for one run. Select a row to read the answer.",
    "ev.q.all": "All",
    "ev.q.expected": "Answer",
    "cat.factual": "Factual",
    "cat.crosslingual": "Cross-lingual",
    "cat.multihop": "Multi-hop",
    "cat.followup": "Follow-up",
    "cat.unanswerable": "Unanswerable",
  },
  zh: {
    "ev.title": "评测",
    "ev.lead": "检索和回答链路上的每一处改动，都先在同一套题目上量过再决定是否保留。下面是保存下来的各次运行结果。",
    "ev.questions": "道题目",
    "ev.categories": "种题型",
    "ev.documents": "篇语料文档",
    "ev.chunks": "个索引分块",
    "ev.empty": "还没有保存的评测结果。运行 scripts/run_eval.py 生成。",
    "ev.ret.title": "检索：每一步带来了什么",
    "ev.ret.lead": "可回答的问题中，来源文档出现在前 5 个段落里的比例（hit-rate@5）。每一行在上一行基础上加一项技术，深色部分是这项技术带来的提升。",
    "ev.mrr": "MRR",
    "ev.correct": "正确性",
    "ev.hit": "Hit@5",
    "ev.fixed": "{cat}：{a} → {b}",
    "ev.mode.title": "回答模式：质量与成本",
    "ev.mode.lead": "同一批问题用三种方式回答：单次检索（RAG）、工具调用循环（Agent），以及只在需要时才启用 Agent 的自动模式。越高越好，越靠右越贵。",
    "ev.mode.note": "token 数为实际计费值，其中包含 API 网关每次调用固定附加的约 4k token 开销，调用次数多的链路受影响更大。Agent 的“找到来源”按整个过程中检索到的全部段落计算，不等同于 hit@5。裁判模型与被测系统是同一个模型。",
    "ev.take.title": "对比单次检索 RAG",
    "ev.take.base": "基线：正确性 {score}，每千题 {cost}。",
    "ev.take.line": "正确性 {delta}，成本为 ×{mult}。",
    "ev.take.cat": "提升最大 — {cat}：{a} → {b}",
    "ev.nav.ret": "检索各阶段",
    "ev.nav.mode": "质量与成本",
    "ev.nav.cat": "按题型",
    "ev.nav.q": "逐题结果",
    "ev.col.mode": "链路",
    "ev.col.correct": "正确性",
    "ev.col.found": "找到来源",
    "ev.col.grounded": "引用有效",
    "ev.col.calls": "LLM 调用",
    "ev.col.tokens": "Tokens",
    "ev.col.latency": "耗时",
    "ev.col.cost": "每千题成本",
    "ev.col.agent": "启用 Agent",
    "ev.axis.cost": "每千题成本（美元）",
    "ev.axis.correct": "正确性（1–5）",
    "ev.cat.title": "各题型的正确性",
    "ev.cat.lead": "每种链路分别赢在哪里。总平均分会掩盖这一点：一条链路整体更差，却可能正是某类问题的最佳选择。",
    "ev.cat.type": "题型",
    "ev.q.title": "逐题结果",
    "ev.q.lead": "某一次运行的逐题结果。点击一行查看回答。",
    "ev.q.all": "全部",
    "ev.q.expected": "回答",
    "cat.factual": "事实题",
    "cat.crosslingual": "跨语言",
    "cat.multihop": "多跳",
    "cat.followup": "追问",
    "cat.unanswerable": "无法回答",
  },
});

const CATEGORY_ORDER = ["factual", "crosslingual", "multihop", "followup", "unanswerable"];
const SVG_NS = "http://www.w3.org/2000/svg";

const pct = (x, digits = 1) => (x == null ? "—" : `${(x * 100).toFixed(digits)}%`);
const score = (x) => (x == null ? "—" : x.toFixed(2));
// A question costs a fraction of a cent, so cost is shown per 1,000 questions.
const per1k = (usd) => (usd == null ? "—" : `$${(usd * 1000).toFixed(2)}`);
const catName = (c) => { const key = `cat.${c}`; const s = t(key); return s === key ? c : s; };

function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const child of children) el.append(child);
  return el;
}

document.getElementById("menu-btn").innerHTML = ICONS.menu;
const { body: sidebarBody, stats } = mountShell("eval");
const root = document.getElementById("eval");

// The sidebar lists the page's sections; each entry scrolls to its block.
const SECTIONS = [["ret", "block-ret"], ["mode", "block-mode"], ["cat", "block-cat"], ["q", "block-q"]];
const sectionNav = h("nav", { class: "section-nav" });
sidebarBody.append(sectionNav);

function renderSectionNav() {
  sectionNav.replaceChildren(...SECTIONS
    .filter(([, id]) => document.getElementById(id))
    .map(([key, id]) => h("button", {
      type: "button", text: t(`ev.nav.${key}`),
      onclick: () => {
        document.getElementById(id).scrollIntoView({ behavior: "smooth", block: "start" });
        document.querySelector(".app").classList.remove("nav-open");
      },
    })));
}
let runs = [];
let corpus = null;
let selectedRun = null;
let selectedCategory = null;

// ------------------------------------------------------------------ header

function header() {
  const any = runs[0]?.summary;
  const categories = any ? Object.keys(any.by_category).length : 0;
  return h("header", { class: "eval-head" },
    h("h1", { text: t("ev.title") }),
    h("p", { text: t("ev.lead") }),
    any ? h("div", { class: "facts" },
      fact(any.questions, t("ev.questions")),
      fact(categories, t("ev.categories")),
      corpus ? fact(fmtInt(corpus.documents), t("ev.documents")) : null,
      corpus ? fact(fmtInt(corpus.chunks), t("ev.chunks")) : null) : null);
}

const fact = (value, label) => h("div", { class: "fact" }, h("b", { text: value }), h("span", { text: label }));

// -------------------------------------------------------- retrieval ladder

function retrievalBlock(group) {
  // The axis is zoomed to the range the runs occupy; its ends are labelled,
  // so a longer bar is never read as "twice as good".
  const values = group.map((r) => r.summary.hit_rate);
  const lo = Math.floor(Math.min(...values) * 10) / 10;
  const hi = Math.min(1, Math.ceil(Math.max(...values) * 20) / 20 + 0.05);
  const pos = (v) => `${(((v - lo) / (hi - lo)) * 100).toFixed(2)}%`;
  const ticks = [];
  for (let v = lo; v <= hi + 1e-9; v += 0.05) ticks.push(v);

  const rows = group.map((run, i) => {
    const prev = group[i - 1];
    const value = run.summary.hit_rate;
    const base = prev ? prev.summary.hit_rate : value;
    return h("div", { class: "run-row" },
      h("div", { class: "label" }, run.meta.label ?? run.id, h("small", { text: biggestGain(run, prev) })),
      h("div", { class: "lane" },
        h("div", { class: "fill", style: `width:${pos(Math.min(base, value))}` }),
        prev && value > base
          ? h("div", { class: "fill gain", style: `left:${pos(base)};width:calc(${pos(value)} - ${pos(base)})` })
          : null),
      h("div", { class: "value" }, pct(value),
        prev ? h("div", { class: "delta", text: `${value >= base ? "+" : ""}${((value - base) * 100).toFixed(1)}` }) : null),
      h("div", { class: "cell", text: run.summary.mrr.toFixed(3) }),
      h("div", { class: "cell", text: score(run.summary.correctness) }));
  });

  return h("section", { class: "block" },
    h("h2", { text: t("ev.ret.title") }),
    h("p", { class: "lead", text: t("ev.ret.lead") }),
    h("div", { class: "card ladder scroll-x" },
      h("div", { class: "axis" },
        h("span"),
        h("div", { class: "ticks" }, ...ticks.map((v) => h("span", { style: `left:${pos(v)}`, text: pct(v, 0) }))),
        h("span", { class: "col-h", text: t("ev.hit") }),
        h("span", { class: "col-h", text: t("ev.mrr") }),
        h("span", { class: "col-h", text: t("ev.correct") })),
      ...rows));
}

/* The question type that improved most over the previous run. */
function biggestGain(run, prev) {
  if (!prev) return "";
  let best = null;
  for (const [cat, now] of Object.entries(run.summary.by_category)) {
    const before = prev.summary.by_category[cat];
    if (now.hit_rate == null || before?.hit_rate == null) continue;
    const gain = now.hit_rate - before.hit_rate;
    if (gain > 0.001 && (!best || gain > best.gain)) best = { cat, gain, a: before.hit_rate, b: now.hit_rate };
  }
  return best ? t("ev.fixed", { cat: catName(best.cat), a: pct(best.a, 0), b: pct(best.b, 0) }) : "";
}

// ------------------------------------------------------- mode comparison

function modeBlock(group) {
  const baseline = group.find((r) => r.meta.mode === "rag") ?? group[0];
  const best = (key, higher = true) => {
    const vals = group.map((r) => r.summary[key]).filter((v) => v != null);
    return vals.length ? (higher ? Math.max(...vals) : Math.min(...vals)) : null;
  };
  const bestCorrect = best("correctness");
  const bestCost = best("avg_cost_usd", false);

  const body = group.map((run) => {
    const m = run.summary;
    const mult = m.avg_cost_usd && baseline.summary.avg_cost_usd
      ? m.avg_cost_usd / baseline.summary.avg_cost_usd : null;
    return h("tr", {},
      h("td", { class: "name" }, run.meta.label ?? run.id),
      h("td", { class: m.correctness === bestCorrect ? "best" : "", text: score(m.correctness) }),
      h("td", { text: pct(m.hit_rate) }),
      h("td", { text: pct(m.grounded_rate, 0) }),
      h("td", { text: m.avg_llm_calls?.toFixed(2) ?? "—" }),
      h("td", { text: m.avg_input_tokens != null ? fmtInt(m.avg_input_tokens + m.avg_output_tokens) : "—" }),
      h("td", { text: m.avg_latency_ms != null ? `${(m.avg_latency_ms / 1000).toFixed(1)} s` : "—" }),
      h("td", { class: m.avg_cost_usd === bestCost ? "best" : "" },
        per1k(m.avg_cost_usd),
        mult && run !== baseline ? h("span", { class: "mult", text: `×${mult.toFixed(1)}` }) : null),
      h("td", { text: pct(m.agent_share, 0) }));
  });

  const table = h("table", { class: "grid" },
    h("thead", {}, h("tr", {},
      ...["mode", "correct", "found", "grounded", "calls", "tokens", "latency", "cost", "agent"]
        .map((c) => h("th", { text: t(`ev.col.${c}`) })))),
    h("tbody", {}, ...body));

  return h("section", { class: "block" },
    h("h2", { text: t("ev.mode.title") }),
    h("p", { class: "lead", text: t("ev.mode.lead") }),
    h("div", { class: "card" },
      h("div", { class: "tradeoff" },
        h("div", { class: "plot" }, tradeoffPlot(group)),
        takeaways(group, baseline)),
      h("div", { class: "scroll-x" }, table)),
    h("p", { class: "note", text: t("ev.mode.note") }));
}

/* One line per pipeline: how it differs from the single-pass baseline. */
function takeaways(group, baseline) {
  const base = baseline.summary;
  const items = group.map((run) => {
    const m = run.summary;
    if (run === baseline) {
      return h("li", {}, h("b", { text: shortLabel(run) }),
        t("ev.take.base", { score: score(m.correctness), cost: per1k(m.avg_cost_usd) }));
    }
    const delta = m.correctness - base.correctness;
    const mult = m.avg_cost_usd && base.avg_cost_usd ? m.avg_cost_usd / base.avg_cost_usd : null;
    // The question type where this pipeline gains most over the baseline.
    let best = null;
    for (const [cat, now] of Object.entries(m.by_category)) {
      const gain = now.correctness - (base.by_category[cat]?.correctness ?? now.correctness);
      if (!best || gain > best.gain) best = { cat, gain, a: base.by_category[cat]?.correctness, b: now.correctness };
    }
    return h("li", {}, h("b", { text: shortLabel(run) }),
      t("ev.take.line", { delta: `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`, mult: mult ? mult.toFixed(1) : "—" }),
      best && best.gain > 0.05
        ? h("small", { text: t("ev.take.cat", { cat: catName(best.cat), a: score(best.a), b: score(best.b) }) })
        : null);
  });
  return h("div", { class: "takeaways" },
    h("div", { class: "eyebrow", text: t("ev.take.title") }), h("ul", {}, ...items));
}

/* Scatter: cost on x, correctness on y. Both axes are drawn from the data. */
function tradeoffPlot(group) {
  const W = 380, H = 250, M = { l: 58, r: 16, t: 14, b: 38 };
  const pts = group.filter((r) => r.summary.avg_cost_usd != null);
  if (!pts.length) return h("span");
  const xs = pts.map((r) => r.summary.avg_cost_usd);
  const ys = pts.map((r) => r.summary.correctness);
  const xMax = Math.max(...xs) * 1.18;
  const yLo = Math.floor((Math.min(...ys) - 0.15) * 4) / 4;
  const yHi = Math.ceil((Math.max(...ys) + 0.15) * 4) / 4;
  const x = (v) => M.l + (v / xMax) * (W - M.l - M.r);
  const y = (v) => H - M.b - ((v - yLo) / (yHi - yLo)) * (H - M.t - M.b);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("ev.mode.title") });
  for (let v = yLo; v <= yHi + 1e-9; v += 0.25) {
    svg.append(s("line", { class: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
    const label = s("text", { x: M.l - 7, y: y(v) + 3.5, "text-anchor": "end" });
    label.textContent = v.toFixed(2);
    svg.append(label);
  }
  const step = xMax / 4;
  for (let i = 0; i <= 4; i += 1) {
    const label = s("text", { x: x(step * i), y: H - M.b + 15, "text-anchor": "middle" });
    label.textContent = per1k(step * i);
    svg.append(label);
  }
  svg.append(s("line", { class: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
  const xTitle = s("text", { x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle" });
  xTitle.textContent = t("ev.axis.cost");
  const yTitle = s("text", { x: 12, y: (M.t + H - M.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(M.t + H - M.b) / 2})` });
  yTitle.textContent = t("ev.axis.correct");
  svg.append(xTitle, yTitle);

  // Points first, then labels. Each label tries a few positions around its
  // point and takes the first that stays inside the plot and clear of every
  // point and every label already placed.
  const dots = pts.map((run) => ({
    run, cx: x(run.summary.avg_cost_usd), cy: y(run.summary.correctness), text: shortLabel(run),
  }));
  dots.forEach((d) => svg.append(s("circle", { class: `pt ${d.run.meta.mode}`, cx: d.cx, cy: d.cy, r: 6 })));

  const placed = [];
  const overlaps = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  const OFFSETS = [[10, -9, "start"], [-10, -9, "end"], [10, 15, "start"], [-10, 15, "end"],
                   [0, -13, "middle"], [0, 21, "middle"], [10, -24, "start"], [-10, -24, "end"]];
  for (const d of dots) {
    const width = d.text.length * 6.3;
    let choice = null;
    for (const [dx, dy, anchor] of OFFSETS) {
      const tx = d.cx + dx;
      const x0 = anchor === "start" ? tx : anchor === "end" ? tx - width : tx - width / 2;
      const box = { x0, x1: x0 + width, y0: d.cy + dy - 10, y1: d.cy + dy + 3 };
      const inside = box.x0 >= M.l && box.x1 <= W - 2 && box.y0 >= 2 && box.y1 <= H - M.b;
      const clear = placed.every((other) => !overlaps(box, other))
        && dots.every((o) => o === d || !overlaps(box, { x0: o.cx - 7, x1: o.cx + 7, y0: o.cy - 7, y1: o.cy + 7 }));
      if (inside && clear) { choice = { tx, ty: d.cy + dy, anchor, box }; break; }
    }
    choice ??= { tx: d.cx + 10, ty: d.cy - 9, anchor: "start", box: { x0: 0, x1: 0, y0: 0, y1: 0 } };
    placed.push(choice.box);
    const label = s("text", { class: "pt-label", x: choice.tx, y: choice.ty, "text-anchor": choice.anchor });
    label.textContent = d.text;
    svg.append(label);
  }
  return svg;
}

const shortLabel = (run) => (run.meta.label ?? run.id).split(/[(:]/)[0].trim()
  + (run.meta.auto_strategy ? ` · ${run.meta.auto_strategy}` : "");

// ------------------------------------------------- correctness by category

function categoryBlock(group) {
  const cats = CATEGORY_ORDER.filter((c) => group.some((r) => r.summary.by_category[c]));
  const rows = cats.map((cat) => {
    const values = group.map((r) => r.summary.by_category[cat]?.correctness ?? null);
    const top = Math.max(...values.filter((v) => v != null));
    return h("tr", {},
      h("td", { class: "name" }, catName(cat),
        h("small", { text: `n = ${group[0].summary.by_category[cat]?.count ?? "—"}` })),
      ...values.map((v) => h("td", {
        class: `heat${v === top ? " best" : ""}`,
        // Opacity grows with the score, so the strong cells stand out.
        style: `--heat:${v == null ? 0 : (((v - 1) / 4) ** 2 * 0.3).toFixed(3)}`,
      }, h("span", { text: score(v) }))));
  });
  return h("section", { class: "block" },
    h("h2", { text: t("ev.cat.title") }),
    h("p", { class: "lead", text: t("ev.cat.lead") }),
    h("div", { class: "card scroll-x" },
      h("table", { class: "grid" },
        h("thead", {}, h("tr", {}, h("th", { text: t("ev.cat.type") }),
          ...group.map((r) => h("th", { text: shortLabel(r) })))),
        h("tbody", {}, ...rows))));
}

// --------------------------------------------------------- every question

function questionBlock() {
  const run = runs.find((r) => r.id === selectedRun) ?? runs.at(-1);
  const cats = CATEGORY_ORDER.filter((c) => run.summary.by_category[c]);
  const rows = run.results.filter((r) => !selectedCategory || r.category === selectedCategory);

  const list = h("div", {}, ...rows.flatMap((r) => {
    const detail = h("p", { class: "q-detail", hidden: true },
      h("b", { text: r.question }), r.answer_preview || "—");
    const row = h("button", {
      class: "q-row", type: "button", "aria-expanded": "false",
      onclick: () => {
        detail.hidden = !detail.hidden;
        row.setAttribute("aria-expanded", String(!detail.hidden));
      },
    },
      h("span", { class: `hit ${!r.answerable ? "na" : r.hit ? "yes" : "no"}`, text: !r.answerable ? "–" : r.hit ? "✓" : "✗" }),
      h("span", { class: "cat", text: catName(r.category) }),
      h("span", { class: "q", text: r.question }),
      h("span", { class: `score${r.correctness < 3 ? " low" : ""}`, text: `${r.correctness.toFixed(0)}/5` }),
      h("span", { class: "mode", text: r.tokens ? r.mode : "" }),
      h("span", { class: "tok", text: r.tokens ? fmtInt(r.tokens) : "" }));
    return [row, detail];
  }));

  const select = h("select", {
    id: "run-select", "aria-label": t("ev.q.title"),
    onchange: (event) => { selectedRun = event.target.value; render(); },
  }, ...runs.map((r) => h("option", { value: r.id, selected: r.id === run.id, text: r.meta.label ?? r.id })));

  const filter = (cat, label) => h("button", {
    class: "chip", type: "button", "aria-pressed": String(selectedCategory === cat),
    onclick: () => { selectedCategory = cat; render(); },
  }, label);

  return h("section", { class: "block" },
    h("h2", { text: t("ev.q.title") }),
    h("p", { class: "lead", text: t("ev.q.lead") }),
    h("div", { class: "card" },
      h("div", { class: "toolbar" }, select,
        h("div", { class: "filters" }, filter(null, t("ev.q.all")), ...cats.map((c) => filter(c, catName(c))))),
      list));
}

// ------------------------------------------------------------------ render

function render() {
  const scrollTop = root.scrollTop;
  if (!runs.length) {
    root.replaceChildren(header(), h("p", { class: "eval-empty", text: t("ev.empty") }));
    return;
  }
  const retrieval = runs.filter((r) => r.meta.group === "retrieval");
  const modes = runs.filter((r) => r.meta.group === "mode");
  const withId = (id, el) => { el.id = id; return el; };
  root.replaceChildren(...[
    header(),
    retrieval.length ? withId("block-ret", retrievalBlock(retrieval)) : null,
    modes.length ? withId("block-mode", modeBlock(modes)) : null,
    modes.length ? withId("block-cat", categoryBlock(modes)) : null,
    withId("block-q", questionBlock()),
  ].filter(Boolean));
  root.scrollTop = scrollTop;
  renderSectionNav();
}

document.addEventListener("langchange", render);

(async () => {
  applyI18n();
  try {
    [{ runs }, corpus] = await Promise.all([getJSON("/eval/results"), stats]);
  } catch {
    runs = [];
  }
  render();
})();
