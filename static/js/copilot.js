/* Care.AI Copilot chat. Sends questions to /copilot/<id>/ask as JSON and
   appends answers in place. Everything is rendered with text nodes — never
   innerHTML — so neither questions nor AI answers can inject markup. The
   plain form POST still works without JavaScript. */
(function () {
  "use strict";

  var search = document.getElementById("patientSearch");
  if (search) {
    search.addEventListener("input", function () {
      var q = search.value.toLowerCase();
      document.querySelectorAll(".copilot-patient").forEach(function (item) {
        item.hidden = item.dataset.name.toLowerCase().indexOf(q) === -1;
      });
    });
  }

  var form = document.getElementById("copilotForm");
  var chat = document.getElementById("chatWindow");
  if (!form || !chat) return;

  var input = document.getElementById("questionInput");
  var submit = form.querySelector("button[type=submit]");
  var askUrl = form.dataset.askUrl;
  var labels = { user: form.dataset.labelUser || "You", assistant: form.dataset.labelAssistant || "Copilot" };
  var busy = false;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function scrollToEnd() { chat.scrollTop = chat.scrollHeight; }

  function bubble(sender, text, meta) {
    var welcome = chat.querySelector(".copilot-welcome");
    if (welcome) welcome.remove();
    meta = meta || {};
    var node = el("div", "chat-msg " + sender + (meta.safety === "urgent" ? " urgent" : "") + (meta.pending ? " pending" : ""));
    node.appendChild(el("small", null, labels[sender] || sender));
    node.appendChild(el("p", null, text));
    if (meta.created_at) node.appendChild(el("span", null, meta.created_at));
    if (meta.source_label) node.appendChild(el("em", "copilot-source " + (meta.source || ""), meta.source_label));
    chat.appendChild(node);
    scrollToEnd();
    return node;
  }

  function setBusy(state) {
    busy = state;
    submit.disabled = state;
    document.querySelectorAll("[data-prompt]").forEach(function (b) { b.disabled = state; });
  }

  function ask(question) {
    question = (question || "").trim();
    if (!question || busy) return;
    bubble("user", question);
    input.value = "";
    setBusy(true);
    var pending = bubble("assistant", form.dataset.labelThinking || "…", { pending: true });

    fetch(askUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ question: question })
    })
      .then(function (response) {
        return response.json().catch(function () { return {}; }).then(function (data) {
          if (!response.ok) throw new Error(data.error || String(response.status));
          return data;
        });
      })
      .then(function (data) {
        pending.remove();
        bubble("assistant", data.answer, data);
      })
      .catch(function (error) {
        pending.remove();
        bubble("assistant", (form.dataset.labelError || "Something went wrong.") + " (" + error.message + ")");
      })
      .then(function () { setBusy(false); input.focus(); });
  }

  form.addEventListener("submit", function (event) { event.preventDefault(); ask(input.value); });
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(input.value); }
  });
  document.querySelectorAll("[data-prompt]").forEach(function (button) {
    button.addEventListener("click", function () { ask(button.dataset.prompt); });
  });

  scrollToEnd();
})();
