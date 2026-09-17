/* Care.AI floating care assistant.
   Talks to the /care-bot/* routes in app.py and renders their structured
   replies with text nodes only — never innerHTML — so neither user input
   nor AI output can inject markup. */
(function () {
  "use strict";

  var root = document.getElementById("care-bot");
  if (!root) return;

  var ui = {};
  try { ui = JSON.parse(document.getElementById("care-bot-ui").textContent); } catch (e) { /* defaults below */ }

  var panel = root.querySelector(".care-bot-panel");
  var launcher = root.querySelector(".care-bot-launcher");
  var log = root.querySelector("[data-care-bot-log]");
  var quick = root.querySelector("[data-care-bot-quick]");
  var form = root.querySelector("[data-care-bot-form]");
  var input = form.querySelector("textarea");
  var sendButton = form.querySelector("button[type=submit]");
  var patientId = root.dataset.patientId || null;
  var OPEN_KEY = "careai-bot-open";
  var loaded = false;
  var busy = false;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function scrollToEnd() { log.scrollTop = log.scrollHeight; }

  function request(url, body) {
    var options = { credentials: "same-origin", headers: { "Accept": "application/json" } };
    if (body !== undefined) {
      options.method = "POST";
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    return fetch(url, options).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (data) {
        if (!response.ok) throw new Error(data.error || String(response.status));
        return data;
      });
    });
  }

  // ------------------------------------------------------------------ render

  function bubble(className) {
    var node = el("div", "care-bot-msg " + (className || ""));
    log.appendChild(node);
    return node;
  }

  function addUser(text) {
    bubble("user").appendChild(el("p", null, text));
    scrollToEnd();
  }

  function renderQuestion(node, reply) {
    if (reply.lead) node.appendChild(el("p", "care-bot-lead", reply.lead));
    if (reply.intro) node.appendChild(el("small", null, reply.intro));
    var meta = el("small");
    meta.appendChild(el("span", "care-bot-counter", reply.counter));
    meta.appendChild(document.createTextNode(" · " + reply.title));
    node.appendChild(meta);
    node.appendChild(el("p", null, reply.text));
    if (reply.notice) node.appendChild(el("small", "care-bot-notice", reply.notice));
  }

  function addList(node, title, items, className) {
    if (!items || !items.length) return;
    if (title) node.appendChild(el("h4", null, title));
    var list = el("ul", className);
    items.forEach(function (line) { list.appendChild(el("li", null, line)); });
    node.appendChild(list);
  }

  // Most useful first: what to do, why, how to cope, when to act sooner;
  // then the possible explanations and the answers it was based on.
  function renderSummary(node, reply) {
    node.classList.add("care-bot-summary");
    node.appendChild(el("h3", null, reply.title));
    node.appendChild(el("span", "care-bot-level " + reply.level, reply.level_label));
    node.appendChild(el("p", "care-bot-action", reply.action));
    if (reply.level === "emergency") node.appendChild(callLink());
    if (reply.stopped_early) node.appendChild(el("small", null, reply.stopped_early));

    if (reply.analysis) {
      node.appendChild(el("h4", null, reply.analysis_title));
      node.appendChild(el("p", "care-bot-analysis", reply.analysis));
    }
    addList(node, reply.advice_title, reply.advice);
    addList(node, reply.watch_title, reply.watch_for, "care-bot-watch");

    if (reply.causes && reply.causes.length) {
      node.appendChild(el("h4", null, reply.causes_title));
      var causes = el("ul", "care-bot-causes");
      reply.causes.forEach(function (cause) {
        var li = el("li");
        li.appendChild(el("span", "care-bot-likelihood " + cause.likelihood, cause.likelihood_label));
        li.appendChild(el("b", null, cause.name));
        if (cause.why) li.appendChild(el("span", "care-bot-why", cause.why));
        causes.appendChild(li);
      });
      node.appendChild(causes);
    }

    if (reply.answers && reply.answers.length) {
      node.appendChild(el("h4", null, reply.answers_title));
      var answers = el("ul", "care-bot-answers");
      reply.answers.forEach(function (item) {
        var li = el("li");
        li.appendChild(el("span", null, item.question));
        li.appendChild(el("b", null, item.answer));
        answers.appendChild(li);
      });
      node.appendChild(answers);
    }
    if (reply.notice) node.appendChild(el("small", "care-bot-notice", reply.notice));
    node.appendChild(el("p", "care-bot-disclaimer", reply.disclaimer));

    if (reply.handoff_available) {
      var send = el("button", "care-bot-handoff coral", ui.handoff || "Send summary to my care team");
      send.type = "button";
      send.dataset.handoff = "1";
      send.addEventListener("click", function () { sendHandoff(send); });
      node.appendChild(send);
    }
  }

  function callLink() {
    var link = el("a", "care-bot-call", "☎ 112");
    link.href = "tel:112";
    return link;
  }

  function addBot(reply) {
    reply = reply || { kind: "text", text: "" };
    var alarm = reply.kind === "emergency" || reply.kind === "crisis";
    var node = bubble(alarm ? "alert" : "");

    if (reply.kind === "question") {
      renderQuestion(node, reply);
    } else if (reply.kind === "summary") {
      renderSummary(node, reply);
    } else {
      if (alarm && reply.level_label) node.appendChild(el("strong", null, reply.level_label));
      node.appendChild(el("p", null, reply.text));
      if (alarm) node.appendChild(callLink());
      if (reply.notice) node.appendChild(el("small", "care-bot-notice", reply.notice));
    }

    // A sent handoff retires every earlier "send" button in the transcript.
    if (reply.kind === "handoff") {
      log.querySelectorAll("[data-handoff]").forEach(function (button) { button.remove(); });
    }
    scrollToEnd();
  }

  function showQuickReplies(reply) {
    quick.textContent = "";
    var options = reply && reply.kind === "question" ? reply.quick_replies : null;
    if (!options || !options.length) { quick.hidden = true; return; }
    options.forEach(function (label) {
      var button = el("button", null, label);
      button.type = "button";
      button.addEventListener("click", function () { send(label); });
      quick.appendChild(button);
    });
    quick.hidden = false;
  }

  function renderTranscript(messages) {
    log.textContent = "";
    var last = null;
    (messages || []).forEach(function (message) {
      if (message.role === "user") {
        addUser(message.text);
      } else {
        last = message.reply || { kind: "text", text: message.text };
        addBot(last);
      }
    });
    showQuickReplies(last);
  }

  // ---------------------------------------------------------------- actions

  function setBusy(state) {
    busy = state;
    sendButton.disabled = state;
    quick.querySelectorAll("button").forEach(function (button) { button.disabled = state; });
  }

  function thinking() {
    var node = bubble("care-bot-typing");
    node.appendChild(el("p", null, ui.thinking || "…"));
    scrollToEnd();
    return node;
  }

  function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    addUser(text);
    input.value = "";
    autosize();
    setBusy(true);
    var pending = thinking();
    request(root.dataset.messageUrl, { text: text, patient_id: patientId })
      .then(function (data) { pending.remove(); addBot(data.reply); showQuickReplies(data.reply); })
      .catch(function () { pending.remove(); addBot({ kind: "text", text: ui.error || "Error" }); })
      .then(function () { setBusy(false); input.focus(); });
  }

  function sendHandoff(button) {
    button.disabled = true;
    request(root.dataset.handoffUrl, { patient_id: patientId })
      .then(function (data) { addBot({ kind: "handoff", text: data.text }); })
      .catch(function (error) { button.disabled = false; addBot({ kind: "text", text: error.message || ui.error }); });
  }

  function restart() {
    if (busy) return;
    setBusy(true);
    request(root.dataset.restartUrl, { patient_id: patientId })
      .then(function (data) { renderTranscript(data.messages); })
      .catch(function () { addBot({ kind: "text", text: ui.error || "Error" }); })
      .then(function () { setBusy(false); input.focus(); });
  }

  function load() {
    if (loaded) return;
    loaded = true;
    var url = root.dataset.sessionUrl + (patientId ? "?patient_id=" + encodeURIComponent(patientId) : "");
    request(url)
      .then(function (data) { renderTranscript(data.messages); })
      .catch(function () { loaded = false; addBot({ kind: "text", text: ui.error || "Error" }); });
  }

  function setOpen(open) {
    panel.hidden = !open;
    launcher.setAttribute("aria-expanded", open ? "true" : "false");
    try { sessionStorage.setItem(OPEN_KEY, open ? "1" : "0"); } catch (e) { /* non-fatal */ }
    if (open) { load(); input.focus(); } else { launcher.focus(); }
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  }

  // ----------------------------------------------------------------- wiring

  root.querySelectorAll("[data-care-bot-toggle]").forEach(function (button) {
    button.addEventListener("click", function () { setOpen(panel.hidden); });
  });
  root.querySelector("[data-care-bot-restart]").addEventListener("click", restart);

  form.addEventListener("submit", function (event) { event.preventDefault(); send(input.value); });
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(input.value); }
  });
  panel.addEventListener("keydown", function (event) {
    if (event.key === "Escape") setOpen(false);
  });

  // Stay open across page navigation within the same tab.
  try { if (sessionStorage.getItem(OPEN_KEY) === "1") setOpen(true); } catch (e) { /* non-fatal */ }
})();
