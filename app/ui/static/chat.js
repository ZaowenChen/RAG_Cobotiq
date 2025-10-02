const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("input-form");
const textareaEl = document.getElementById("user-input");
const audienceEl = document.getElementById("audience-level");
const robotEl = document.getElementById("robot-model");
const sendButton = document.getElementById("send-button");
const template = document.getElementById("message-template");

function appendMessage({ role, content, citations }) {
  const clone = template.content.firstElementChild.cloneNode(true);
  clone.classList.add(role === "user" ? "user" : "bot");
  clone.querySelector(".content").textContent = content;

  const citationsEl = clone.querySelector(".citations");
  citationsEl.innerHTML = "";
  if (citations && citations.length) {
    citations.forEach((citation) => {
      const div = document.createElement("div");
      div.className = "citation";
      const link = document.createElement("a");
      link.href = citation.source_uri || "#";
      link.textContent = `${citation.citation_id} ${citation.doc_title}`;
      link.target = "_blank";
      div.appendChild(link);
      citationsEl.appendChild(div);
    });
  }

  messagesEl.appendChild(clone);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(event) {
  event.preventDefault();
  const message = textareaEl.value.trim();
  if (!message) return;

  appendMessage({ role: "user", content: message });

  textareaEl.value = "";
  textareaEl.style.height = "";
  sendButton.disabled = true;

  const params = new URLSearchParams({ query: message });
  if (audienceEl.value) params.set("audience_level", audienceEl.value);
  if (robotEl.value) params.set("robot_model", robotEl.value);

  appendMessage({ role: "bot", content: "Thinking…" });
  const pending = messagesEl.lastElementChild;

  try {
    const response = await fetch(`/search?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }
    const data = await response.json();

    const answer = data.answer || "No answer generated. See retrieved passages below.";
    const citations = data.citations || [];

    pending.querySelector(".content").textContent = answer;
    const citationsEl = pending.querySelector(".citations");
    citationsEl.innerHTML = "";
    citations.forEach((citation) => {
      const div = document.createElement("div");
      div.className = "citation";
      const link = document.createElement("a");
      link.href = citation.source_uri || "#";
      link.textContent = `${citation.citation_id} ${citation.doc_title}`;
      link.target = "_blank";
      div.appendChild(link);
      citationsEl.appendChild(div);
    });

    if (data.error) {
      const banner = document.createElement("div");
      banner.className = "error-banner";
      banner.textContent = data.error;
      pending.appendChild(banner);
    }

    if (!data.answer && data.results) {
      const citationsEl = pending.querySelector(".citations");
      const top = data.results.slice(0, 3);
      top.forEach((item, index) => {
        const div = document.createElement("div");
        div.className = "citation";
        const link = document.createElement("a");
        link.href = item.source_uri || "#";
        link.textContent = `[R${index + 1}] ${item.doc_title}`;
        link.target = "_blank";
        div.appendChild(link);
        citationsEl.appendChild(div);
      });
    }
  } catch (error) {
    pending.querySelector(".content").textContent = String(error);
    const citationsEl = pending.querySelector(".citations");
    citationsEl.innerHTML = "";
  } finally {
    sendButton.disabled = false;
  }
}

textareaEl.addEventListener("input", () => {
  textareaEl.style.height = "auto";
  textareaEl.style.height = `${textareaEl.scrollHeight}px`;
});

formEl.addEventListener("submit", sendMessage);

appendMessage({
  role: "bot",
  content: "Hi! Ask me about release notes, troubleshooting, or wiring diagrams. I cite the docs I use.",
  citations: [],
});
