/* Cortex chat frontend — vanilla JS, no dependencies.
 *
 * NOTE (learning): the entire interaction model is three steps:
 *   1. read the input, POST it to /ask as JSON
 *   2. clone <template> nodes and fill them with the response
 *   3. append to the chat container and scroll down
 * No framework needed — the browser's own APIs (fetch, template, dataset)
 * cover everything a chat UI requires.
 */

"use strict";

const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const toast = document.getElementById("toast");

// NOTE (learning): textContent (not innerHTML) everywhere user/LLM text is
// inserted — this is what prevents XSS. The only HTML we ever construct
// ourselves is the citation chips, built from numbers we validated.
function el(templateId) {
  return document.getElementById(templateId).content.firstElementChild.cloneNode(true);
}

function scrollToBottom() {
  chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4000);
}

// --------------------------------------------------------------- rendering

function addUserMessage(text) {
  const node = el("tpl-user-msg");
  node.querySelector(".bubble-user").textContent = text;
  chat.appendChild(node);
  scrollToBottom();
}

function addTyping() {
  const node = el("tpl-typing");
  chat.appendChild(node);
  scrollToBottom();
  return node;
}

/* Replace [N] markers with clickable chips. We split the answer on the
 * marker pattern so every text part goes in via textContent (XSS-safe),
 * and only the chips — whose content is a validated integer — are elements. */
function renderAnswer(container, answer, sourceCount) {
  const parts = answer.split(/\[(\d{1,2})\]/g);
  parts.forEach((part, i) => {
    if (i % 2 === 0) {
      container.appendChild(document.createTextNode(part));
    } else {
      const n = parseInt(part, 10);
      if (n >= 1 && n <= sourceCount) {
        const chip = document.createElement("span");
        chip.className = "cite-chip";
        chip.textContent = n;
        chip.dataset.cite = n;
        container.appendChild(chip);
      } else {
        container.appendChild(document.createTextNode(`[${part}]`));
      }
    }
  });
}

function renderSources(container, sources) {
  sources.forEach((src, i) => {
    const card = el("tpl-source-card");
    card.dataset.sourceIndex = i + 1;
    card.querySelector(".source-num").textContent = i + 1;
    card.querySelector(".source-name").textContent = src.source;
    card.querySelector(".source-score").textContent = src.score.toFixed(3);
    card.querySelector(".source-text").textContent = src.text;
    container.appendChild(card);
  });
}

function addAssistantMessage(data) {
  const node = el("tpl-assistant-msg");
  const sources = data.sources || [];

  renderAnswer(node.querySelector(".answer"), data.answer, sources.length);
  renderSources(node.querySelector(".sources"), sources);

  const cost = data.cost_usd_est;
  node.querySelector(".stats").textContent =
    `${(data.latency_ms / 1000).toFixed(1)}s · ` +
    `${data.input_tokens}↑ ${data.output_tokens}↓ tokens · ` +
    `$${cost.toFixed(5)}`;

  chat.appendChild(node);
  scrollToBottom();
}

// Clicking a [N] chip opens and highlights the matching source card.
chat.addEventListener("click", (event) => {
  const chip = event.target.closest(".cite-chip");
  if (!chip) return;
  const bubble = chip.closest(".bubble-assistant");
  const card = bubble.querySelector(`.source-card[data-source-index="${chip.dataset.cite}"]`);
  if (!card) return;
  card.open = true;
  card.classList.remove("flash");
  void card.offsetWidth; // restart the flash animation if clicked twice
  card.classList.add("flash");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

// ------------------------------------------------------------------ /ask

async function ask(question) {
  sendBtn.disabled = true;
  input.disabled = true;
  addUserMessage(question);
  const typing = addTyping();

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `Server error (${resp.status})`);
    }
    addAssistantMessage(await resp.json());
  } catch (err) {
    showToast(err.message === "Failed to fetch" ? "Cannot reach the server." : err.message);
  } finally {
    typing.remove();
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendBtn.disabled) return;
  input.value = "";
  document.querySelector(".welcome")?.remove();
  ask(question);
});

// ---------------------------------------------------------------- /health

async function checkHealth() {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  try {
    const resp = await fetch("/health");
    if (!resp.ok) throw new Error();
    dot.className = "status-dot ok";
    text.textContent = "online";
  } catch {
    dot.className = "status-dot err";
    text.textContent = "offline";
  }
}

checkHealth();
input.focus();
