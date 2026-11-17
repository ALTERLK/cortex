/* The conversation: sending a question and rendering the answer as it streams.
 *
 * One message is one plain object (see newMessage). Stream events only
 * update that object; small render functions then redraw the part of the
 * message that changed. Nothing reads state back out of the DOM.
 *
 * The browser owns the conversation: prior turns are sent with each
 * request and the server keeps no session.
 */

import { ApiError, ask } from "./api.js";
import { applyI18n, getLang, t } from "./i18n.js";
import { renderMarkdown } from "./markdown.js";
import { copyText, fmtCost, fmtInt, fmtMs, h, icon, store, toast } from "./ui.js";

const HISTORY_SENT = 12;

// Example questions, one per kind of question the system handles. The text follows
// the interface language; the documents are mostly English either way, so the
// Chinese versions also show cross-language retrieval.
const EXAMPLES = [
  { tag: "ex.fact", mode: "auto", q: {
    en: 'What does a wildcard "*" in CORS allowed origins prevent you from doing?',
    zh: "在 CORS 里，允许的来源设成通配符 * 会妨碍什么？" } },
  { tag: "ex.cross", mode: "auto", q: {
    en: "什么时候应该用 async def 定义路径操作函数，什么时候用普通 def？",
    zh: "什么时候应该用 async def 定义路径操作函数，什么时候用普通 def？" } },
  { tag: "ex.multi", mode: "agent", q: {
    en: "How would you start the FastAPI dev server inside a uv-managed project environment?",
    zh: "怎么在 uv 管理的项目环境里启动 FastAPI 开发服务器？" } },
  { tag: "ex.trap", mode: "auto", q: {
    en: "How does Kubernetes decide when to restart a Pod?",
    zh: "Kubernetes 是怎么决定什么时候重启一个 Pod 的？" } },
];

/* Route reasons the server sends in auto mode, mapped to translated text. */
const ROUTE_REASONS = {
  "answered in a single pass": "route.single",
  "single pass came up empty; escalated": "route.escalated",
};

const TOOL_LABELS = {
  search_knowledge_base: "act.search",
  read_neighbors: "act.neighbors",
  list_documents: "act.list",
};

export function mountChat(inspector) {
  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("question");
  const sendBtn = document.getElementById("send");
  const modeHint = document.getElementById("mode-hint");
  const modeButtons = [...document.querySelectorAll("[data-mode]")];

  let mode = store.get("mode", "auto");
  let history = [];          // completed turns, sent with the next request
  let messages = [];
  let controller = null;     // AbortController of the request in flight

  // ------------------------------------------------------------------ mode

  function setMode(next) {
    mode = next;
    store.set("mode", next);
    modeButtons.forEach((b) => b.setAttribute("aria-checked", b.dataset.mode === next));
    modeHint.textContent = t(`mode.${next}.hint`);
  }
  modeButtons.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

  // --------------------------------------------------------------- welcome

  function renderWelcome() {
    chat.replaceChildren(
      h("div", { class: "col welcome", id: "welcome" },
        h("h1", {}, h("span", { text: t("welcome.title") }), h("span", { text: t("welcome.title2") })),
        h("p", { text: t("welcome.body") }),
        h("div", { class: "examples" }, ...EXAMPLES.map((ex) =>
          h("button", {
            class: "example", type: "button",
            onclick: () => { setMode(ex.mode); send(ex.q[getLang()]); },
          },
            h("span", { class: "chip", text: t(ex.tag) }),
            h("span", { class: "q", text: ex.q[getLang()] }))))),
    );
  }

  function reset() {
    controller?.abort();
    history = [];
    messages = [];
    inspector.close();
    renderWelcome();
    input.focus();
  }

  // -------------------------------------------------------------- messages

  function newMessage(question) {
    const el = {
      meta: h("div", { class: "msg-meta" }),
      activity: h("div", { class: "activity" }),
      thinking: h("div", { class: "thinking" }),
      answer: h("div", { class: "answer" }),
      sources: h("div", { class: "sources" }),
      foot: h("div", { class: "msg-foot" }),
    };
    const root = h("div", { class: "col msg msg-assistant" },
      h("div", { class: "avatar", html: icon("logo").innerHTML }),
      h("div", { class: "msg-body" }, el.meta, el.activity, el.thinking, el.answer, el.sources, el.foot));
    return {
      question, root, el,
      requested: mode,
      phase: "retrieving",   // drives the status line until text arrives
      route: null,           // {mode, reason} in auto mode
      rewrite: null,         // {original, rewritten} for a rewritten follow-up
      sources: [],
      activity: [],          // agent tool calls, for the inline list
      spans: [],             // timed steps, for the trace waterfall
      answer: "",
      done: null,            // the final stats event
      error: null,
      stopped: false,
    };
  }

  function renderMeta(msg) {
    const parts = [];
    if (msg.route) {
      const reasonKey = ROUTE_REASONS[msg.route.reason];
      parts.push(h("span", { class: "chip accent" }, icon("route"),
        `${t(`route.${msg.route.mode}`)} · ${reasonKey ? t(reasonKey) : msg.route.reason}`));
    } else if (msg.done) {
      parts.push(h("span", { class: "chip", text: t(`meta.${msg.done.mode}`) }));
    }
    if (msg.rewrite) {
      parts.push(h("span", { class: "rewrite" }, t("meta.searched"), " ", h("b", { text: msg.rewrite.rewritten })));
    }
    msg.el.meta.replaceChildren(...parts);
  }

  function renderActivity(msg) {
    msg.el.activity.replaceChildren(...msg.activity.map((act) => {
      if (act.note) {
        return h("div", { class: "act note" }, icon("alert"), h("span", { class: "label", text: t(act.note) }));
      }
      const added = act.added ? t("act.new", { n: act.added }) : t("act.none");
      return h("div", { class: "act" },
        icon(act.name === "search_knowledge_base" ? "search" : act.name === "read_neighbors" ? "book" : "list"),
        h("span", { class: "label", text: act.label, title: act.label }),
        h("span", { class: "detail num", text: `${act.settled ? `${added} · ` : ""}${fmtMs(act.ms)}` }));
    }));
  }

  function renderThinking(msg) {
    const active = !msg.done && !msg.error && !msg.stopped && !msg.answer;
    msg.el.thinking.hidden = !active;
    if (!active) return;
    const label = msg.phase === "reading"
      ? t("state.reading", { n: msg.sources.length })
      : t(`state.${msg.phase}`);
    msg.el.thinking.replaceChildren(
      h("span", { class: "dots" }, h("i"), h("i"), h("i")), h("span", { text: label }));
  }

  /* Streaming re-renders the markdown at most once per frame. */
  function renderAnswer(msg) {
    if (msg.frame) return;
    msg.frame = requestAnimationFrame(() => {
      msg.frame = 0;
      const stick = nearBottom();
      renderMarkdown(msg.el.answer, msg.answer, msg.sources.length);
      if (stick) scrollToBottom();
    });
  }

  function renderSources(msg) {
    const cited = new Set(msg.done?.citations?.cited ?? []);
    msg.el.sources.replaceChildren(...msg.sources.map((source, i) => {
      const n = i + 1;
      // Three trailing segments keep en/… and zh/… translations apart.
      const short = source.source.split("/").slice(-3).join("/");
      return h("button", {
        class: `src${msg.done && !cited.has(n) ? " unused" : ""}`, type: "button",
        title: source.source,
        onclick: () => openEvidence(msg, n),
      }, h("span", { class: "n", text: n }), h("span", { class: "path", text: short }));
    }));
  }

  function renderFoot(msg) {
    const { done } = msg;
    if (msg.error) {
      msg.el.foot.replaceChildren(h("div", { class: "msg-error", text: msg.error }));
      return;
    }
    if (!done) {
      msg.el.foot.replaceChildren(msg.stopped ? h("span", { text: t("foot.stopped") }) : "");
      return;
    }
    const sep = () => h("span", { class: "sep", text: "·" });
    const parts = [
      h("span", { class: "num", text: fmtMs(done.latency_ms) }), sep(),
      h("span", { class: "num", text: t("foot.tokens", { in: fmtInt(done.input_tokens), out: fmtInt(done.output_tokens) }) }), sep(),
      h("span", { class: "num", text: fmtCost(done.cost_usd_est) }),
    ];
    if (done.iterations) parts.push(sep(), h("span", { class: "num", text: t("foot.steps", { n: done.iterations }) }));
    parts.push(sep(), verification(done));
    if (done.stop_reason && done.stop_reason !== "answered") {
      parts.push(sep(), h("span", { class: "verify warn", text: t("foot.limit") }));
    }
    const copyBtn = h("button", {
      class: "link", type: "button", text: t("foot.copy"),
      onclick: async () => {
        copyBtn.textContent = (await copyText(msg.answer)) ? t("foot.copied") : t("foot.copy");
        setTimeout(() => { copyBtn.textContent = t("foot.copy"); }, 1500);
      },
    });
    parts.push(h("span", { class: "actions" },
      h("button", { class: "link", type: "button", text: t("foot.evidence"), onclick: () => openEvidence(msg, null, "trace") }),
      copyBtn));
    msg.el.foot.replaceChildren(...parts);
  }

  /* What the server's deterministic citation check found. */
  function verification(done) {
    const { cited = [], invalid = [] } = done.citations ?? {};
    if (done.refusal) return h("span", { class: "verify", text: t("foot.refusal") });
    if (invalid.length) {
      return h("span", { class: "verify warn" }, icon("alert"),
        t("foot.invalid", { list: invalid.map((n) => `[${n}]`).join(", ") }));
    }
    if (cited.length) {
      return h("span", { class: "verify ok" }, icon("check"),
        cited.length === 1 ? t("foot.verified.one") : t("foot.verified", { n: cited.length }));
    }
    return h("span", { class: "verify warn" }, icon("alert"), t("foot.uncited"));
  }

  function renderAll(msg) {
    renderMeta(msg);
    renderActivity(msg);
    renderThinking(msg);
    renderSources(msg);
    renderFoot(msg);
  }

  function openEvidence(msg, cite, view) {
    msg.root.querySelectorAll(".cite").forEach((chip) =>
      chip.classList.toggle("on", cite != null && Number(chip.dataset.cite) === cite));
    inspector.show(msg, { cite, view });
  }

  // --------------------------------------------------------- stream events

  /* Each handler updates the message object, then redraws what changed. */
  const handlers = {
    route(msg, data) {
      msg.route = data;
      renderMeta(msg);
    },
    rewrite(msg, data) {
      msg.rewrite = data;
      renderMeta(msg);
    },
    sources(msg, data) {
      msg.sources.push(...data);
      const last = msg.activity.at(-1);
      if (last && !last.note && !last.settled) last.added += data.length;  // found by the latest tool call
      else msg.phase = "reading";
      renderSources(msg);
      renderActivity(msg);
      renderThinking(msg);
    },
    step(msg, data) {
      msg.spans.push(data);
      settleActivity(msg);
      if (data.name === "plan") msg.phase = "planning";
      if (data.name === "retrieve") msg.phase = "writing";
      renderThinking(msg);
    },
    tool(msg, data) {
      settleActivity(msg);
      const args = data.arguments ?? {};
      const label = data.name === "search_knowledge_base" ? args.query
        : data.name === "read_neighbors" ? `${args.source} · ${args.chunk_index}`
        : t(TOOL_LABELS[data.name] ?? data.name);
      msg.activity.push({ name: data.name, label: label ?? data.name, ms: data.ms, added: 0, settled: false });
      msg.spans.push({ kind: "tool", name: data.name, ms: data.ms, detail: label });
      msg.phase = "planning";
      renderActivity(msg);
      renderThinking(msg);
    },
    delta(msg, text) {
      msg.answer += text;
      renderThinking(msg);
      renderAnswer(msg);
    },
    answer(msg, data) {
      msg.answer = data.text;   // authoritative final text
      renderThinking(msg);
      renderAnswer(msg);
    },
    escalate(msg) {
      // The single pass failed and the agent takes over: drop what the
      // first pass showed, and say so in the activity list and the trace.
      msg.sources = [];
      msg.answer = "";
      msg.rewrite = null;
      msg.el.answer.replaceChildren();
      msg.activity.push({ note: "act.escalated" });
      msg.spans.push({ kind: "marker" });
      msg.phase = "escalating";
      renderAll(msg);
    },
    error(msg, data) {
      msg.error = t("err.failed", { detail: data.detail });
    },
    done(msg, data) {
      msg.done = data;
      settleActivity(msg);
    },
  };

  /* A tool call's "N new" count is final once the next event arrives. */
  function settleActivity(msg) {
    msg.activity.forEach((act) => { if (!act.note) act.settled = true; });
  }

  // ----------------------------------------------------------------- send

  const nearBottom = () => chat.scrollHeight - chat.scrollTop - chat.clientHeight < 140;
  const scrollToBottom = () => { chat.scrollTop = chat.scrollHeight; };

  function setBusy(busy) {
    sendBtn.classList.toggle("stop", busy);
    sendBtn.replaceChildren(icon(busy ? "stop" : "send"));
    sendBtn.title = t(busy ? "composer.stop" : "composer.send");
    sendBtn.setAttribute("aria-label", sendBtn.title);
  }

  async function send(question) {
    if (controller) return;
    document.getElementById("welcome")?.remove();
    chat.append(h("div", { class: "col msg msg-user" }, h("div", { class: "bubble", text: question })));

    const msg = newMessage(question);
    messages.push(msg);
    chat.append(msg.root);
    renderAll(msg);
    scrollToBottom();

    controller = new AbortController();
    setBusy(true);
    try {
      await ask(
        { question, mode, history: history.slice(-HISTORY_SENT) },
        {
          signal: controller.signal,
          onEvent: (event, data) => {
            const stick = nearBottom();
            handlers[event]?.(msg, data);
            if (stick) scrollToBottom();
          },
        },
      );
    } catch (error) {
      if (error.name === "AbortError") msg.stopped = true;
      else if (error instanceof ApiError) msg.error = t("err.failed", { detail: error.message });
      else msg.error = t("err.offline");
    } finally {
      controller = null;
      setBusy(false);
      settleActivity(msg);
      renderAll(msg);
      renderAnswer(msg);
      inspector.refresh(msg);
      if (msg.done && msg.answer) {
        history.push({ role: "user", content: question }, { role: "assistant", content: msg.answer });
      }
      input.focus();
    }
  }

  // -------------------------------------------------------------- composer

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (controller) return controller.abort();   // the button is "Stop" while a request runs
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    autosize();
    send(question);
  });

  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (event) => {
    // Enter sends; Shift+Enter adds a line. `isComposing` is true while an
    // input method (Chinese, Japanese…) is still choosing characters, and
    // that Enter confirms the candidate rather than sending.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  // A citation chip opens the evidence panel on that passage.
  chat.addEventListener("click", (event) => {
    const chip = event.target.closest(".cite");
    if (!chip) return;
    const msg = messages.find((m) => m.root.contains(chip));
    if (msg) openEvidence(msg, Number(chip.dataset.cite));
  });

  document.getElementById("new-chat")?.addEventListener("click", reset);
  document.getElementById("inspector-btn")?.addEventListener("click", () => {
    inspector.show(messages.at(-1) ?? null);
  });

  document.addEventListener("langchange", () => {
    setMode(mode);
    setBusy(Boolean(controller));
    if (!messages.length) renderWelcome();
    messages.forEach(renderAll);
  });

  setMode(mode);
  setBusy(false);
  renderWelcome();
  applyI18n();
  input.focus();
}
