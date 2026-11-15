/* Render an LLM answer: markdown → sanitized HTML → citation chips.
 *
 * The answer is model output built from retrieved documents, so it is
 * untrusted. Three steps keep it safe:
 *   1. `marked` turns the markdown into an HTML string.
 *   2. `DOMPurify` parses that string and keeps only an allowlist of
 *      formatting tags — scripts, event handlers, forms, and images (which
 *      would fetch arbitrary URLs) never reach the page.
 *   3. Citation markers are replaced by walking TEXT NODES of the sanitized
 *      tree and building the chips with createElement, so no string is
 *      ever concatenated into HTML after sanitizing.
 *
 * Both libraries are vendored under /static/vendor — pinned versions, no
 * CDN, so the UI works without internet access.
 */

/* global marked, DOMPurify */

marked.setOptions({ gfm: true, breaks: false });

const SANITIZE = {
  ALLOWED_TAGS: [
    "p", "br", "strong", "em", "del", "code", "pre", "blockquote", "hr",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td", "a", "span",
  ],
  ALLOWED_ATTR: ["href", "title", "class"],
};

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

const CITATION = /\[(\d{1,3})\]/g;

/**
 * Render `text` into `container`.
 * A marker [N] becomes a chip only when 1 <= N <= sourceCount; anything
 * else is left as plain text, so an invented citation stays visibly inert.
 */
export function renderMarkdown(container, text, sourceCount) {
  container.innerHTML = DOMPurify.sanitize(marked.parse(text), SANITIZE);
  linkCitations(container, sourceCount);
  addCopyButtons(container);
}

function linkCitations(root, sourceCount) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement.closest("code, pre, a")
        ? NodeFilter.FILTER_REJECT       // `items[0]` in code is not a citation
        : NodeFilter.FILTER_ACCEPT,
  });
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const node of textNodes) {
    const parts = node.nodeValue.split(CITATION);  // odd indexes hold the numbers
    if (parts.length === 1) continue;
    const fragment = document.createDocumentFragment();
    parts.forEach((part, i) => {
      const n = Number(part);
      if (i % 2 === 1 && n >= 1 && n <= sourceCount) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "cite";
        chip.dataset.cite = n;
        chip.textContent = n;
        fragment.append(chip);
      } else {
        fragment.append(i % 2 === 1 ? `[${part}]` : part);
      }
    });
    node.replaceWith(fragment);
  }
}

function addCopyButtons(root) {
  root.querySelectorAll("pre").forEach((pre) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-code";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      const code = pre.querySelector("code")?.textContent ?? "";
      try {
        await navigator.clipboard.writeText(code);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Select and copy";
      }
      setTimeout(() => { button.textContent = "Copy"; }, 1600);
    });
    pre.append(button);
  });
}
