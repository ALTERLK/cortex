/* Entry point of the chat page: build the shell, then mount each panel. */

import { mountChat } from "./chat.js";
import { mountInspector } from "./inspector.js";
import { mountKnowledgeBase } from "./kb.js";
import { mountShell } from "./shell.js";
import { h, icon, ICONS } from "./ui.js";

document.getElementById("menu-btn").innerHTML = ICONS.menu;
document.getElementById("inspector-btn").innerHTML = ICONS.panel;

const { body, stats } = mountShell("chat");

body.append(
  h("div", { style: "padding: 12px 16px 0" },
    h("button", { class: "btn", id: "new-chat", type: "button", style: "width: 100%; justify-content: center" },
      icon("plus"), h("span", { dataset: { i18n: "nav.new" } }))),
);
mountKnowledgeBase(body, stats);
mountChat(mountInspector());
