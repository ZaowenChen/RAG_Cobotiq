const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("input-form");
const textareaEl = document.getElementById("user-input");
const audienceEl = document.getElementById("audience-level");
const robotEl = document.getElementById("robot-model");
const sendButton = document.getElementById("send-button");
const template = document.getElementById("message-template");

function renderCitations(container, citations = []) {
  if (!container) return;
  container.innerHTML = "";
  if (!citations.length) {
    container.style.display = "none";
    return;
  }
  citations.forEach((citation) => {
    const div = document.createElement("div");
    div.className = "citation";
    const link = document.createElement("a");
    link.href = citation.source_uri || "#";
    link.textContent = `${citation.citation_id} ${citation.doc_title}`;
    link.target = "_blank";
    div.appendChild(link);
    container.appendChild(div);
  });
  container.style.display = "block";
}

function renderFigures(container, figures = []) {
  if (!container) return;
  container.innerHTML = "";
  if (!figures.length) {
    container.style.display = "none";
    return;
  }

  figures.forEach((figure) => {
    const card = document.createElement("div");
    card.className = "figure-card";

    if (figure.media_url) {
      const img = document.createElement("img");
      img.src = figure.media_url;
      img.alt = figure.caption || figure.doc_title || "Figure";
      img.loading = "lazy";
      card.appendChild(img);
    }

    const caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = figure.caption || "Figure";
    card.appendChild(caption);

    const sourceLink = document.createElement("a");
    sourceLink.href = figure.source_uri || "#";
    sourceLink.target = "_blank";
    sourceLink.textContent = figure.doc_title || "Open source";
    card.appendChild(sourceLink);

    container.appendChild(card);
  });

  container.style.display = "grid";
}

function appendMessage({ role, content, citations = [], figures = [] }) {
  const clone = template.content.firstElementChild.cloneNode(true);
  clone.classList.add(role === "user" ? "user" : "bot");
  clone.querySelector(".content").textContent = content;

  renderCitations(clone.querySelector(".citations"), citations);
  renderFigures(clone.querySelector(".figures"), figures);

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
    const figures = data.figures || [];

    pending.querySelector(".content").textContent = answer;
    renderCitations(pending.querySelector(".citations"), citations);
    renderFigures(pending.querySelector(".figures"), figures);

    if (data.error) {
      const banner = document.createElement("div");
      banner.className = "error-banner";
      banner.textContent = data.error;
      pending.appendChild(banner);
    }

    if (!data.answer && data.results) {
      const fallback = data.results.slice(0, 3).map((item, index) => ({
        citation_id: `[R${index + 1}]`,
        doc_title: item.doc_title,
        source_uri: item.source_uri,
      }));
      renderCitations(pending.querySelector(".citations"), fallback);
    }
  } catch (error) {
    pending.querySelector(".content").textContent = String(error);
    renderCitations(pending.querySelector(".citations"), []);
    renderFigures(pending.querySelector(".figures"), []);
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
  figures: [],
});
