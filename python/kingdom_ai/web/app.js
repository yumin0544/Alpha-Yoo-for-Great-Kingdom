"use strict";
const $ = id => document.getElementById(id);
const colorName = {Black: "파랑", White: "주황"};
let current = null, requesting = false, selectedMode = "human_ai", focusedAction = 0;
let toastTimer, pendingSettings = null, renderedRevision = -1, renderingRequesting = null;
const pointButtons = [];

function svgElement(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  $("grid").append(node);
}
for (let i = 0; i < 9; i++) {
  const p = 100 + i * 100;
  svgElement("line", {x1:100, y1:p, x2:900, y2:p, stroke:"#ac8d5d", "stroke-width":1.7});
  svgElement("line", {x1:p, y1:100, x2:p, y2:900, stroke:"#ac8d5d", "stroke-width":1.7});
  const attrs = {fill:"#9b7e50", "font-size":22, "font-family":"Segoe UI, sans-serif", "text-anchor":"middle"};
  svgElement("text", {...attrs, x:p, y:53}, String(i + 1));
  svgElement("text", {...attrs, x:48, y:p + 8}, String(i + 1));
}
for (const [x,y] of [[300,300],[700,300],[300,700],[700,700]]) {
  svgElement("circle", {cx:x, cy:y, r:4, fill:"#a38552"});
}
for (let action = 0; action < 81; action++) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "point empty";
  button.style.left = `${10 + action % 9 * 10}%`;
  button.style.top = `${10 + Math.floor(action / 9) * 10}%`;
  button.tabIndex = action === 0 ? 0 : -1;
  const piece = document.createElement("span");
  piece.className = "piece";
  button.append(piece);
  button.addEventListener("click", () => playAction(action));
  button.addEventListener("focus", () => { focusedAction = action; });
  button.addEventListener("keydown", event => {
    const steps = {ArrowLeft:-1, ArrowRight:1, ArrowUp:-9, ArrowDown:9};
    if (!(event.key in steps)) return;
    event.preventDefault();
    let next = action + steps[event.key];
    if ((event.key === "ArrowLeft" && action % 9 === 0) || (event.key === "ArrowRight" && action % 9 === 8)) return;
    while (next >= 0 && next < 81) {
      if (!pointButtons[next].disabled) {
        pointButtons[action].tabIndex = -1;
        pointButtons[next].tabIndex = 0;
        pointButtons[next].focus();
        return;
      }
      const previous = next;
      next += steps[event.key];
      if (Math.floor(previous / 9) !== Math.floor(next / 9) && Math.abs(steps[event.key]) === 1) return;
    }
  });
  $("points").append(button);
  pointButtons.push(button);
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 5500);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache:"no-store"} : {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) {
    if (data.state) render(data.state);
    throw new Error(data.error || "요청을 처리하지 못했습니다.");
  }
  return data;
}

function render(state) {
  if (current && current.session_id === state.session_id && state.revision < current.revision) return;
  if (current && current.session_id === state.session_id && state.revision === renderedRevision && renderingRequesting === requesting) return;
  if (current?.session_id !== state.session_id) renderedRevision = -1;
  current = state;
  renderingRequesting = requesting;
  const legal = new Set(state.legal_actions);
  const canPlay = state.can_play && !requesting;
  $("board").className = `board turn-${state.to_play === "Black" ? "black" : "white"}`;
  const last = state.history.at(-1);
  for (let action = 0; action < 81; action++) {
    const cell = state.cells[action], owner = state.ownership[action], button = pointButtons[action];
    let type = cell === "Black" ? "black" : cell === "White" ? "white" : cell === "Neutral" ? "neutral" :
      owner === "Black" ? "territory-black" : owner === "White" ? "territory-white" : "empty";
    button.className = `point ${type}${last?.action === action ? " last" : ""}`;
    button.disabled = !canPlay || !legal.has(action);
    const label = type === "neutral" ? "중립 돌" : type.startsWith("territory-") ? `${colorName[owner]}의 완성된 집` :
      cell === "Empty" ? "빈 자리" : `${colorName[cell]} 돌`;
    button.setAttribute("aria-label", `${Math.floor(action / 9) + 1}행 ${action % 9 + 1}열, ${label}`);
    button.title = button.getAttribute("aria-label");
  }
  if (pointButtons[focusedAction]?.disabled) {
    const first = pointButtons.findIndex(button => !button.disabled);
    if (first >= 0) focusedAction = first;
  }
  pointButtons.forEach((button,index) => { button.tabIndex = index === focusedAction && !button.disabled ? 0 : -1; });
  for (const color of ["Black", "White"]) {
    const prefix = color === "Black" ? "black" : "white";
    const isHuman = state.settings.mode === "human_human" || color === state.settings.human;
    $(`${prefix}-name`).textContent = state.settings.mode === "human_human" ? `${colorName[color]} 플레이어` : isHuman ? "나" : "AI";
    $(`${prefix}-remaining`).textContent = state.remaining[color];
    const score = $(`${prefix}-score`);
    score.replaceChildren(document.createTextNode(String(state.score[color])));
    const unit = document.createElement("small"); unit.textContent = "칸"; score.append(unit);
    $(`${prefix}-card`).classList.toggle("active", !state.finished && state.to_play === color);
  }
  $("move-count").textContent = `${state.history.length}수`;
  $("turn-panel").classList.toggle("thinking", state.busy);
  $("turn-panel").classList.toggle("finished", state.finished);
  $("pass-button").disabled = !canPlay || !legal.has(81);
  $("start-button").disabled = requesting;
  $("restart-button").disabled = requesting;
  if (state.error) {
    $("turn-eyebrow").textContent = "LET’S TRY AGAIN";
    $("turn-title").textContent = "AI를 준비하지 못했어요";
    $("turn-description").textContent = state.error;
  } else if (state.finished) {
    $("turn-eyebrow").textContent = "KINGDOM COMPLETE";
    $("turn-title").textContent = `${colorName[state.winner]}의 승리!`;
    const ending = {"상대 돌 포획":"상대 돌 포획으로", "자충수":"자충수로", "연속 패스":"연속 패스로"};
    $("turn-description").textContent = `${ending[state.reason] || state.reason} 대국이 끝났습니다. 새 게임으로 다시 도전해 보세요.`;
  } else if (state.busy) {
    $("turn-eyebrow").textContent = state.ready ? "THINKING…" : "GETTING READY…";
    $("turn-title").textContent = state.ready ? "AI가 생각하고 있어요" : "상대를 준비하고 있어요";
    $("turn-description").textContent = state.ready ? "다음 한 수를 찾고 있어요. 잠시 기다려 주세요." : "선택한 상대를 불러오고 있습니다.";
  } else {
    $("turn-eyebrow").textContent = "YOUR TURN";
    $("turn-title").textContent = state.settings.mode === "human_human" ? `${colorName[state.to_play]}의 차례입니다` : "지금, 나의 차례입니다";
    $("turn-description").textContent = state.consecutive_passes ? "앞선 차례에 패스했어요. 이번에도 패스하면 집을 세고 대국이 끝납니다." : "빈 교차점에 돌을 놓아 보세요. 가장자리와 중립 돌도 벽이 됩니다.";
  }
  if (renderedRevision !== state.revision) {
    renderedRevision = state.revision;
    $("history").replaceChildren();
    const visible = state.history.slice(-6).reverse();
    $("history-label").textContent = state.history.length ? `전체 ${state.history.length}수` : "대국 시작";
    if (!visible.length) {
      const empty = document.createElement("li"); empty.className = "history-empty";
      empty.textContent = "첫 돌을 놓아 보세요."; $("history").append(empty);
    }
    for (const move of visible) {
      const li = document.createElement("li");
      const number = document.createElement("span"); number.className = "history-number"; number.textContent = String(move.number).padStart(2,"0");
      const piece = document.createElement("span"); piece.className = `history-piece ${move.player === "Black" ? "black" : "white"}`;
      const player = document.createElement("span"); player.className = "history-label"; player.textContent = colorName[move.player];
      const label = document.createElement("span"); label.className = "history-label"; label.textContent = move.label;
      li.append(number,piece,player,label); $("history").append(li);
    }
  }
}
async function playAction(action) {
  if (!current?.can_play || requesting) return;
  requesting = true; render(current);
  try { render(await api("/api/move", {action,revision:current.revision,session_id:current.session_id})); }
  catch(error) { toast(error.message); }
  finally { requesting = false; render(current); }
}
function selectedSettings() {
  return {mode:selectedMode, human:$("human-select").value, model:$("model-select").value,
    simulations:Number($("simulations-select").value), device:$("model-select").value === "mcts" ? "cpu" : $("device-select").value, seed:42};
}
async function startGame(settings) {
  requesting = true; render(current);
  try { render(await api("/api/new", {settings,revision:current.revision,session_id:current.session_id})); }
  catch(error) { toast(error.message); }
  finally { requesting = false; render(current); }
}
function requestRestart(settings) {
  if (!current || requesting) return;
  if (current.history.length && !current.finished) {
    pendingSettings = settings; $("restart-dialog").showModal();
  } else startGame(settings);
}
function setMode(mode) {
  selectedMode = mode;
  for (const button of document.querySelectorAll("[data-mode]")) {
    button.classList.toggle("selected", button.dataset.mode === mode);
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }
  for (const id of ["color-row","model-row","ai-row"]) $(id).hidden = mode === "human_human";
}
document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));
$("pass-button").addEventListener("click", () => playAction(81));
$("restart-button").addEventListener("click", () => requestRestart(current.settings));
$("settings-form").addEventListener("submit", event => {event.preventDefault(); requestRestart(selectedSettings());});
$("restart-cancel").addEventListener("click", () => $("restart-dialog").close());
$("restart-confirm").addEventListener("click", () => {$("restart-dialog").close(); startGame(pendingSettings);});
$("help-open").addEventListener("click", () => $("help-dialog").showModal());
for (const id of ["help-close","help-done"]) $(id).addEventListener("click", () => $("help-dialog").close());
$("model-select").addEventListener("change", () => { $("device-select").disabled = $("model-select").value === "mcts"; });

async function initialize() {
  try {
    const [models,state] = await Promise.all([api("/api/models"),api("/api/state")]);
    for (const model of models.models) {
      const option = document.createElement("option"); option.value = model.id; option.textContent = model.name.replace(/^runs\//, "");
      $("model-select").append(option);
    }
    setMode(state.settings.mode);
    $("human-select").value = state.settings.human;
    $("model-select").value = state.settings.model;
    $("simulations-select").value = String(state.settings.simulations);
    $("device-select").value = state.settings.device;
    $("device-select").disabled = state.settings.model === "mcts";
    render(state);
  } catch(error) {
    $("turn-title").textContent = "게임에 연결하지 못했어요";
    $("turn-description").textContent = "게임 실행 창이 켜져 있는지 확인하고 새로고침해 주세요.";
    toast(error.message);
  }
  setTimeout(poll, 500);
}
async function poll() {
  if (!requesting) {
    try { render(await api("/api/state")); }
    catch(error) { renderedRevision = -1; $("pass-button").disabled = true; pointButtons.forEach(button => {button.disabled = true;});
      $("turn-title").textContent = "게임 연결이 끊겼어요";
      $("turn-description").textContent = "게임 실행 창을 켜고 잠시 기다려 주세요."; }
  }
  setTimeout(poll, document.hidden ? 1800 : 500);
}
initialize();
