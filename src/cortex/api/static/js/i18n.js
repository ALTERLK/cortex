/* Interface strings in English and Chinese.
 *
 * Static markup declares its key with data-i18n (text), data-i18n-placeholder
 * or data-i18n-title; applyI18n() fills those in. Code that builds DOM calls
 * t(key, vars). Switching language fires a "langchange" event so modules
 * can re-render what they built.
 */

import { store } from "./ui.js";

const STRINGS = {
  en: {
    "nav.chat": "Chat",
    "nav.eval": "Evaluation",
    "nav.new": "New chat",
    "status.connecting": "connecting…",
    "status.offline": "server offline",
    "theme.toggle": "Switch theme",
    "lang.toggle": "中文",

    "kb.title": "Knowledge base",
    "kb.documents": "documents",
    "kb.chunks": "chunks",
    "kb.filter": "Filter documents…",
    "kb.empty": "No documents match.",
    "kb.none": "The knowledge base is empty. Add a document to get started.",
    "kb.drop.title": "Add a document",
    "kb.drop.hint": "Drop a .md, .txt or .pdf here, or click — up to {mb} MB",
    "kb.uploading": "Uploading {name}…",
    "kb.indexing": "Indexing {name}…",
    "kb.indexed": "{name} indexed: {n} chunks. It is searchable now.",
    "kb.deleted": "{name} removed from the knowledge base.",
    "kb.delete": "Remove this document",
    "kb.rewrite.on": "query rewrite",
    "kb.uploads": "uploads",
    "kb.readonly": "read-only demo",

    "welcome.title": "Ask your documents.",
    "welcome.title2": "See how every answer was found.",
    "welcome.body": "Each answer cites the passages it used. Open a citation to see where that passage ranked in embedding search and in keyword search, and what the whole request cost.",
    "ex.fact": "Single fact",
    "ex.cross": "Chinese question, English docs",
    "ex.multi": "Needs several searches",
    "ex.trap": "Not in the documents",

    "composer.placeholder": "Ask a question about the indexed documents…",
    "composer.send": "Send",
    "composer.stop": "Stop",
    "mode.auto": "Auto",
    "mode.rag": "RAG",
    "mode.agent": "Agent",
    "mode.auto.hint": "One pass first; the agent takes over only if that finds nothing",
    "mode.rag.hint": "Retrieve once, then answer — fastest and cheapest",
    "mode.agent.hint": "The model plans its own searches — slower, several times the cost",

    "state.retrieving": "Searching the knowledge base",
    "state.reading": "Reading {n} passages",
    "state.planning": "Planning the next step",
    "state.writing": "Writing the answer",
    "state.escalating": "Nothing found in one pass — handing over to the agent",

    "route.rag": "Auto → RAG",
    "route.agent": "Auto → Agent",
    "route.single": "answered in a single pass",
    "route.escalated": "single pass found nothing; escalated",
    "meta.rag": "RAG",
    "meta.agent": "Agent",
    "meta.searched": "Searched for:",
    "act.search": "search",
    "act.neighbors": "read around",
    "act.list": "list documents",
    "act.new": "{n} new",
    "act.none": "nothing new",
    "act.escalated": "Single pass found nothing to cite. Escalated to the agent.",

    "foot.tokens": "{in} in · {out} out tokens",
    "foot.steps": "{n} LLM calls",
    "foot.verified": "{n} citations verified",
    "foot.verified.one": "1 citation verified",
    "foot.invalid": "cites {list}, which matches no source",
    "foot.uncited": "no citations in this answer",
    "foot.refusal": "not covered by the documents",
    "foot.stopped": "stopped",
    "foot.limit": "stopped at the step limit",
    "foot.copy": "Copy",
    "foot.copied": "Copied",
    "foot.evidence": "Evidence",

    "insp.title": "Evidence",
    "insp.sources": "Sources",
    "insp.trace": "Trace",
    "insp.empty": "Ask a question, then open a citation to see the passage behind it.",
    "insp.nosources": "This answer used no passages.",
    "insp.dense": "Embedding rank",
    "insp.sparse": "Keyword rank (BM25)",
    "insp.fused": "Fused score (RRF)",
    "insp.rerank": "Rerank score",
    "insp.notfound": "not in top 20",
    "insp.more": "Show full passage",
    "insp.less": "Show less",
    "insp.chunk": "chunk {n}",
    "insp.unused": "retrieved, not cited",
    "trace.time": "total time",
    "trace.calls": "LLM calls",
    "trace.tokens": "tokens",
    "trace.cost": "est. cost",
    "trace.rewrite": "Follow-up rewritten for retrieval:",
    "trace.escalated": "First pass discarded — escalated to the agent",
    "trace.empty": "The trace appears when the answer is complete.",
    "span.rewrite": "rewrite query",
    "span.retrieve": "retrieve",
    "span.generate": "generate",
    "span.route": "route",
    "span.plan": "plan",
    "span.answer": "answer",
    "span.search_knowledge_base": "search",
    "span.read_neighbors": "read around",
    "span.list_documents": "list documents",

    "err.offline": "Cannot reach the server.",
    "err.failed": "The request failed: {detail}",
  },

  zh: {
    "nav.chat": "对话",
    "nav.eval": "评测",
    "nav.new": "新对话",
    "status.connecting": "连接中…",
    "status.offline": "服务未启动",
    "theme.toggle": "切换主题",
    "lang.toggle": "EN",

    "kb.title": "知识库",
    "kb.documents": "篇文档",
    "kb.chunks": "个分块",
    "kb.filter": "筛选文档…",
    "kb.empty": "没有匹配的文档。",
    "kb.none": "知识库还是空的，先添加一篇文档。",
    "kb.drop.title": "添加文档",
    "kb.drop.hint": "拖入 .md / .txt / .pdf，或点击选择 — 最大 {mb} MB",
    "kb.uploading": "正在上传 {name}…",
    "kb.indexing": "正在索引 {name}…",
    "kb.indexed": "{name} 已索引，共 {n} 个分块，现在可以检索了。",
    "kb.deleted": "已从知识库移除 {name}。",
    "kb.delete": "移除这篇文档",
    "kb.rewrite.on": "查询改写",
    "kb.uploads": "上传的文档",
    "kb.readonly": "只读演示",

    "welcome.title": "向你的文档提问，",
    "welcome.title2": "并看到每个答案是怎么找到的。",
    "welcome.body": "每个回答都会标注它引用的段落。点开引用，可以看到这段内容在向量检索和关键词检索里各排第几，以及这次请求的完整开销。",
    "ex.fact": "单一事实",
    "ex.cross": "中文提问，英文文档",
    "ex.multi": "需要多次检索",
    "ex.trap": "文档里没有",

    "composer.placeholder": "就已索引的文档提一个问题…",
    "composer.send": "发送",
    "composer.stop": "停止",
    "mode.auto": "自动",
    "mode.rag": "RAG",
    "mode.agent": "Agent",
    "mode.auto.hint": "先单次检索作答；找不到依据时才交给 Agent",
    "mode.rag.hint": "检索一次后直接作答 — 最快、最省",
    "mode.agent.hint": "由模型自己规划多次检索 — 更慢，成本是数倍",

    "state.retrieving": "正在检索知识库",
    "state.reading": "正在阅读 {n} 个段落",
    "state.planning": "正在规划下一步",
    "state.writing": "正在撰写回答",
    "state.escalating": "单次检索没有找到依据 — 交给 Agent 继续",

    "route.rag": "自动 → RAG",
    "route.agent": "自动 → Agent",
    "route.single": "单次检索即可作答",
    "route.escalated": "单次检索无果，已升级",
    "meta.rag": "RAG",
    "meta.agent": "Agent",
    "meta.searched": "实际检索：",
    "act.search": "检索",
    "act.neighbors": "阅读上下文",
    "act.list": "列出文档",
    "act.new": "新增 {n} 段",
    "act.none": "没有新内容",
    "act.escalated": "单次检索没有可引用的内容，已升级到 Agent。",

    "foot.tokens": "输入 {in} · 输出 {out} tokens",
    "foot.steps": "{n} 次 LLM 调用",
    "foot.verified": "{n} 处引用已校验",
    "foot.verified.one": "1 处引用已校验",
    "foot.invalid": "引用了不存在的来源 {list}",
    "foot.uncited": "这个回答没有引用",
    "foot.refusal": "文档中没有相关内容",
    "foot.stopped": "已停止",
    "foot.limit": "达到步数上限后停止",
    "foot.copy": "复制",
    "foot.copied": "已复制",
    "foot.evidence": "查看依据",

    "insp.title": "依据",
    "insp.sources": "来源",
    "insp.trace": "链路",
    "insp.empty": "提一个问题，然后点开引用，查看它背后的段落。",
    "insp.nosources": "这个回答没有用到任何段落。",
    "insp.dense": "向量检索排名",
    "insp.sparse": "关键词排名 (BM25)",
    "insp.fused": "融合得分 (RRF)",
    "insp.rerank": "重排得分",
    "insp.notfound": "未进前 20",
    "insp.more": "展开完整段落",
    "insp.less": "收起",
    "insp.chunk": "第 {n} 块",
    "insp.unused": "已检索，未被引用",
    "trace.time": "总耗时",
    "trace.calls": "LLM 调用",
    "trace.tokens": "tokens",
    "trace.cost": "预估成本",
    "trace.rewrite": "追问已改写后再检索：",
    "trace.escalated": "首轮结果被丢弃 — 已升级到 Agent",
    "trace.empty": "回答完成后会显示完整链路。",
    "span.rewrite": "改写查询",
    "span.retrieve": "检索",
    "span.generate": "生成",
    "span.route": "路由",
    "span.plan": "规划",
    "span.answer": "作答",
    "span.search_knowledge_base": "检索",
    "span.read_neighbors": "阅读上下文",
    "span.list_documents": "列出文档",

    "err.offline": "无法连接到服务。",
    "err.failed": "请求失败：{detail}",
  },
};

let lang = store.get("lang") || ((navigator.language || "en").toLowerCase().startsWith("zh") ? "zh" : "en");

export const getLang = () => lang;

export function t(key, vars = {}) {
  const template = STRINGS[lang][key] ?? STRINGS.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) => (name in vars ? vars[name] : `{${name}}`));
}

/* Extra strings a page registers for itself (the evaluation page does). */
export function addStrings(extra) {
  for (const code of Object.keys(extra)) Object.assign(STRINGS[code], extra[code]);
}

export function applyI18n(root = document) {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  root.querySelectorAll("[data-i18n-title]").forEach((el) => {
    el.title = t(el.dataset.i18nTitle);
    el.setAttribute("aria-label", el.title);
  });
}

export function setLang(next) {
  lang = next;
  store.set("lang", next);
  applyI18n();
  document.dispatchEvent(new CustomEvent("langchange"));
}
