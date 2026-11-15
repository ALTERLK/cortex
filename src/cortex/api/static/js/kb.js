/* Knowledge-base panel in the sidebar: what is indexed, and adding to it. */

import { deleteDocument, getJSON, uploadDocument, waitForJob } from "./api.js";
import { t } from "./i18n.js";
import { fmtInt, h, icon, toast } from "./ui.js";

const ACCEPTED = [".md", ".txt", ".pdf"];

export function mountKnowledgeBase(container, statsPromise) {
  let stats = null;
  let docs = [];
  let filter = "";
  let fresh = null;      // source of the document that was just uploaded
  let busy = false;

  const statsEl = h("div", { class: "kb-stats" });
  const configEl = h("div", { class: "kb-config" });
  const listEl = h("div", { class: "kb-list" });
  const filterEl = h("input", {
    class: "kb-filter", id: "kb-filter", type: "search", autocomplete: "off",
    oninput: (event) => { filter = event.target.value.trim().toLowerCase(); renderList(); },
  });
  const fileInput = h("input", {
    type: "file", id: "kb-file", accept: ACCEPTED.join(","), hidden: true,
    onchange: (event) => { addFile(event.target.files[0]); event.target.value = ""; },
  });
  const dropTitle = h("strong");
  const dropHint = h("span");
  const progressBar = h("i");
  const progress = h("div", { class: "progress", hidden: true }, progressBar);
  const dropzone = h("div", {
    class: "dropzone", id: "dropzone", role: "button", tabindex: "0", hidden: true,
    onclick: () => { if (!busy) fileInput.click(); },
    onkeydown: (event) => { if ((event.key === "Enter" || event.key === " ") && !busy) fileInput.click(); },
    ondragover: (event) => { event.preventDefault(); dropzone.classList.add("over"); },
    ondragleave: () => dropzone.classList.remove("over"),
    ondrop: (event) => {
      event.preventDefault();
      dropzone.classList.remove("over");
      addFile(event.dataTransfer.files[0]);
    },
  }, dropTitle, dropHint, progress);

  container.append(
    h("div", { class: "kb" },
      h("div", { class: "kb-head" },
        h("div", { class: "eyebrow", dataset: { i18n: "kb.title" } }), statsEl, configEl),
      filterEl, listEl, dropzone, fileInput),
  );

  function renderHead() {
    filterEl.placeholder = t("kb.filter");
    if (!stats) return;
    statsEl.replaceChildren(
      h("div", { class: "kb-stat" }, h("b", { class: "num", text: fmtInt(stats.documents) }), h("span", { text: t("kb.documents") })),
      h("div", { class: "kb-stat" }, h("b", { class: "num", text: fmtInt(stats.chunks) }), h("span", { text: t("kb.chunks") })),
    );
    // replaceChildren() would render a null argument as the text "null".
    configEl.replaceChildren(...[
      h("span", { class: "chip", text: stats.retrieval_mode.replace("_", " + ") }),
      stats.query_rewrite && h("span", { class: "chip", text: t("kb.rewrite.on") }),
      stats.read_only && h("span", { class: "chip warn", text: t("kb.readonly") }),
    ].filter(Boolean));
    dropzone.hidden = stats.read_only;
    if (!busy) {
      dropTitle.textContent = t("kb.drop.title");
      dropHint.textContent = t("kb.drop.hint", { mb: stats.max_upload_mb });
    }
  }

  /* Group documents by their top-level folder; uploads always come first. */
  function renderList() {
    const visible = docs.filter((d) => d.source.toLowerCase().includes(filter));
    if (!visible.length) {
      listEl.replaceChildren(h("div", { class: "kb-empty", text: t(docs.length ? "kb.empty" : "kb.none") }));
      return;
    }
    const groups = new Map();
    for (const doc of visible) {
      const slash = doc.source.indexOf("/");
      const key = slash === -1 ? "·" : doc.source.slice(0, slash);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(doc);
    }
    listEl.replaceChildren(...[...groups].map(([key, items]) => {
      const isUploads = key === "uploads";
      return h("details", { class: "kb-group", open: isUploads || Boolean(filter) },
        h("summary", {},
          icon("caret", "caret"),
          h("span", { text: isUploads ? t("kb.uploads") : key }),
          h("span", { class: "count num", text: fmtInt(items.length) })),
        ...items.map((doc) => docRow(doc, key)));
    }));
  }

  function docRow(doc, groupKey) {
    const name = groupKey === "·" ? doc.source : doc.source.slice(groupKey.length + 1);
    return h("div", { class: `kb-doc${doc.source === fresh ? " fresh" : ""}`, title: doc.source },
      h("span", { class: "name", text: name }),
      h("span", { class: "chunks num", text: doc.chunks }),
      doc.uploaded && !stats?.read_only
        ? h("button", {
            class: "icon-btn", type: "button", title: t("kb.delete"), "aria-label": t("kb.delete"),
            onclick: () => removeDoc(doc),
          }, icon("trash"))
        : null);
  }

  async function refresh() {
    try {
      [stats, { documents: docs }] = await Promise.all([getJSON("/stats"), getJSON("/documents")]);
    } catch {
      return; // the status line already says the server is unreachable
    }
    renderHead();
    renderList();
  }

  async function addFile(file) {
    if (!file || busy) return;
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED.includes(ext)) return toast(t("kb.drop.hint", { mb: stats?.max_upload_mb ?? 5 }));

    busy = true;
    dropzone.classList.add("busy");
    progress.hidden = false;
    progressBar.style.transform = "scaleX(0.08)";
    dropTitle.textContent = t("kb.uploading", { name: file.name });
    dropHint.textContent = "";
    try {
      const { job_id: jobId, source } = await uploadDocument(file);
      dropTitle.textContent = t("kb.indexing", { name: file.name });
      const job = await waitForJob(jobId, (done, total) => {
        progressBar.style.transform = `scaleX(${total ? Math.max(0.15, done / total) : 0.5})`;
      });
      progressBar.style.transform = "scaleX(1)";
      fresh = source;
      await refresh();
      toast(t("kb.indexed", { name: file.name, n: job.chunks_stored }));
    } catch (error) {
      toast(t("err.failed", { detail: error.message }));
    } finally {
      busy = false;
      dropzone.classList.remove("busy");
      progress.hidden = true;
      renderHead();
    }
  }

  async function removeDoc(doc) {
    try {
      await deleteDocument(doc.source);
      await refresh();
      toast(t("kb.deleted", { name: doc.source.replace(/^uploads\//, "") }));
    } catch (error) {
      toast(t("err.failed", { detail: error.message }));
    }
  }

  document.addEventListener("langchange", () => { renderHead(); renderList(); });
  statsPromise.then((loaded) => { if (loaded) refresh(); });
}
