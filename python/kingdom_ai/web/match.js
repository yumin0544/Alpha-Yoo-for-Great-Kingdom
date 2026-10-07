"use strict";
const $ = id => document.getElementById(id);
const reasons = {Capture:"포획", Suicide:"자충수", TwoPasses:"연속 패스"};
const statuses = {loading:"모델 준비",running:"대결 중",stopping:"중단 대기",complete:"완료",cancelled:"중단",failed:"오류",incomplete:"일부 기록"};
let serverId = null, selectedId = null, current = null, runs = [], requesting = false, connected = true;
let page = 1, pages = 1, visualKey = "", rowsKey = "", rowsRequest = 0, toastTimer, stopId;
let replay = null, replayRequest = 0;
let workersEdited = false;

function node(tag, text, className) {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  if (className) value.className = className;
  return value;
}
function names(view) {
  return Object.fromEntries(view.participants.map(item => [item.id,item.name]));
}
function number(value) { return Number(value).toLocaleString("ko-KR"); }
function rate(value) { return value == null ? "—" : `${(value * 100).toFixed(1)}%`; }
function duration(seconds) {
  if (seconds == null) return "—";
  const total = Math.max(0,Math.floor(seconds));
  if (total < 60) return `${total}초`;
  if (total < 3600) return `${Math.floor(total/60)}분 ${total%60}초`;
  return `${Math.floor(total/3600)}시간 ${Math.floor(total%3600/60)}분`;
}
function notify(message) {
  clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false;
  toastTimer = setTimeout(() => {$("toast").hidden = true;},6500);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache:"no-store"} : {
    method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...body,session_id:serverId})
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "요청을 처리하지 못했습니다.");
  return data;
}
function updateStart() {
  const busy = runs.some(run => ["loading","running","stopping"].includes(run.status)) ||
    (current && ["loading","running","stopping"].includes(current.status));
  const enough = $("model-a").options.length >= 2;
  $("start-match").disabled = requesting || busy || !enough || !serverId || !connected;
  $("start-hint").textContent = !connected ? "실행 창과의 연결을 확인해 주세요." : !enough ?
    "runs 폴더에 저장된 best.pt 모델이 두 개 필요합니다." : busy ?
    "진행 중인 대결이 끝난 뒤 새 대결을 시작할 수 있습니다." : "결과는 자동으로 저장되며, 선후공이 같은 횟수로 배정됩니다.";
}
async function loadCatalog() {
  const data = await api("/api/matches");
  serverId = data.session_id; runs = data.runs;
  const select = $("run-select");
  const existing = new Map([...select.options].map(option => [option.value,option]));
  for (const run of runs) {
    const label = names(run);
    const date = run.started_at_utc ? new Date(run.started_at_utc).toLocaleString("ko-KR",{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}) : "";
    const text = `${date} · ${label.a || "A"} vs ${label.b || "B"} · ${number(run.completed)}/${number(run.target)} · ${statuses[run.status] || run.status}`;
    let option = existing.get(run.id);
    if (!option) {option = node("option"); option.value = run.id; select.append(option);}
    option.textContent = text;
  }
  for (const option of [...select.options]) if (!runs.some(run => run.id === option.value)) option.remove();
  if (!runs.length) {const option=node("option","아직 대결 기록이 없습니다.");option.value="";select.append(option);}
  if (selectedId && !runs.some(run=>run.id===selectedId)) {
    selectedId=null;current=null;visualKey="";rowsKey="";page=1;
  }
  if (!selectedId && runs.length) selectedId = runs[0].id;
  select.value = selectedId || "";
  $("empty-state").hidden = Boolean(selectedId);
  $("match-results").hidden = !selectedId;
  updateStart();
}
function svg(tag, attrs, text) {
  const item = document.createElementNS("http://www.w3.org/2000/svg",tag);
  for (const [key,value] of Object.entries(attrs)) item.setAttribute(key,value);
  if (text !== undefined) item.textContent = text;
  return item;
}
function drawTrend(view) {
  const root = $("trend-chart"); root.replaceChildren();
  if (!view.curve.length) {root.setAttribute("aria-label","완료한 대국이 없습니다.");root.append(node("div","첫 판이 끝나면 승률의 흐름이 표시됩니다.","chart-empty"));return;}
  const chart = svg("svg",{viewBox:"0 0 640 180",preserveAspectRatio:"none","aria-hidden":"true"});
  for (const value of [0,0.5,1]) {
    const y = 144 - value*128;
    chart.append(svg("line",{x1:38,x2:622,y1:y,y2:y,stroke:value===0.5?"#d4ddc8":"#e9eee0","stroke-dasharray":value===0.5?"4 5":"0"}));
    chart.append(svg("text",{x:0,y:y+3,fill:"#a0aa90","font-size":10},`${value*100}%`));
  }
  const x = game => 38 + (game - 1)/Math.max(1,view.completed-1)*584;
  for (const [key,color] of [["a","#39789a"],["b","#d78c4e"]]) {
    const points = view.curve.map(item=>`${x(item.game)},${144-(key==="a"?item.rate:1-item.rate)*128}`).join(" ");
    chart.append(svg("polyline",{points,fill:"none",stroke:color,"stroke-width":2,"vector-effect":"non-scaling-stroke","stroke-linejoin":"round"}));
    const last = view.curve.at(-1), value = key === "a" ? last.rate : 1-last.rate;
    chart.append(svg("circle",{cx:x(last.game),cy:144-value*128,r:3,fill:color}));
  }
  chart.append(svg("text",{x:38,y:169,fill:"#a0aa90","font-size":10},"1판"));
  chart.append(svg("text",{x:622,y:169,fill:"#a0aa90","font-size":10,"text-anchor":"end"},`${number(view.completed)}판`));
  root.append(chart);
  root.setAttribute("aria-label",`${view.completed}판 완료, 모델 A 누적 승률 ${rate(view.win_rate_a)}, 모델 B ${rate(1-view.win_rate_a)}`);
}
function drawDetails(view) {
  $("color-stats").replaceChildren();
  for (const key of ["a","b"]) {
    const row = node("div",undefined,"color-row"); row.append(node("span",key.toUpperCase(),`letter ${key}`));
    for (const [color,label,dot] of [["Black","흑 · 선공","black-dot"],["White","백 · 후공","white-dot"]]) {
      const stats = view.colors[key][color], result = node("div",undefined,"color-result"), title=node("span",label);
      title.prepend(node("i",undefined,dot));
      result.append(title,node("strong",rate(stats.games?stats.wins/stats.games:null)),node("small",`${number(stats.wins)} / ${number(stats.games)}판`));row.append(result);
    }
    $("color-stats").append(row);
  }
  $("ending-stats").replaceChildren();
  for (const [reason,label] of Object.entries(reasons)) {
    const row=node("div",undefined,"ending-row"), track=node("div",undefined,"ending-track"), fill=node("span");
    fill.style.width=`${view.completed?view.endings[reason]/view.completed*100:0}%`;track.append(fill);
    row.append(node("span",label),track,node("strong",`${number(view.endings[reason])}판`));$("ending-stats").append(row);
  }
  $("mean-plies").textContent = `평균 수순 ${view.mean_plies==null?"—":view.mean_plies.toFixed(1)+"수"}`;
  $("ratings").replaceChildren();
  for (const key of ["a","b"]) {
    const value=node("div",undefined,"rating-value");value.append(node("span",key.toUpperCase(),`letter ${key}`));
    const initial=view.protocol.initial_ratings?.[key];
    value.append(document.createTextNode(initial==null?"—":initial.toFixed(1)));
    if (view.ratings) {
      const delta=view.ratings.after[key]-view.ratings.before[key];
      value.append(document.createTextNode(" → "),node("strong",view.ratings.after[key].toFixed(1)),node("small",`${delta>=0?"+":""}${delta.toFixed(1)}`));
    } else value.append(node("small","미적용"));
    $("ratings").append(value);
  }
  $("rating-note").textContent = `대결 전체 완료 시 K=${view.protocol.k_per_series??32}를 한 번 적용합니다. 이번 상대·설정에 대한 내부 평가입니다.`;
  drawTrend(view);
}
function render(view) {
  current = view;
  const label=names(view);
  for (const key of ["a","b"]) {
    $(`name-${key}`).textContent=label[key] || key.toUpperCase(); $(`name-${key}`).title=label[key] || "";
    $(`rate-${key}`).textContent=rate(view.completed?view.wins[key]/view.completed:null);
    $(`record-${key}`).textContent=`${number(view.wins[key])}승 · ${number(view.completed-view.wins[key])}패`;
  }
  $("run-status").textContent=statuses[view.status] || view.status;$("run-status").dataset.status=view.status;
  $("progress-count").textContent=`${number(view.completed)} / ${number(view.target)}판`;
  const percentage=view.target?view.completed/view.target*100:0;
  $("progress-fill").style.width=`${percentage}%`;$("progress-bar").setAttribute("aria-valuenow",String(percentage));
  $("stop-match").hidden=!view.can_stop;$("stop-match").disabled=requesting;
  const workers = Math.min(view.protocol.workers || 1, view.target);
  const fast = view.protocol.backend === "batched_cpp";
  const protocol = `${view.protocol.device === "cuda"?"GPU":"CPU"} · 동시 ${workers}판 · ${fast?"빠른 배치 탐색":"기존 탐색"} · 매 수 ${number(view.protocol.simulations || 0)}회 · 시드 ${view.protocol.seed??42} · 전술 검사 ${view.protocol.tactical_checks?"적용":"미적용"}${fast?` · 배치 ${view.protocol.leaf_batch_size} · 트리 재사용 ${view.protocol.reuse_tree?"켜짐":"꺼짐"}`:""}`;
  const active = view.progress?.active_games;
  const progress = view.progress && view.status === "running" ? (active ? `진행 중 ${active.length}판 · ` : `${view.progress.index}판 ${view.progress.plies}수 · `) : "";
  $("live-description").textContent=view.status==="loading"?`두 모델을 준비하고 있습니다. · 동시 ${workers}판`:view.status==="stopping"?"각 판의 현재 탐색이 끝나면 중단합니다.":progress+protocol;
  $("elapsed").textContent=`경과 ${duration(view.elapsed_seconds)}${view.eta_seconds!=null?` · 예상 남은 시간 약 ${duration(view.eta_seconds)}`:""}`;
  $("throughput").textContent=view.games_per_minute ? `평균 분당 ${view.games_per_minute.toFixed(1)}판 · 예상 시간은 완료한 대국의 평균 속도를 기준으로 갱신합니다.` : "완료한 대국이 쌓이면 처리 속도를 표시합니다.";
  $("run-error").hidden=!view.error;$("run-error").textContent=view.error || "";
  $("storage-path").textContent=`자동 저장 · ${view.file}`;
  for (const format of ["csv","jsonl"]) {
    const anchor=$(`download-${format}`);anchor.href=`/api/matches/${view.id}/${format}`;
    anchor.setAttribute("aria-disabled",String(!view.can_download));anchor.tabIndex=view.can_download?0:-1;
  }
  const nextKey=`${view.id}:${view.completed}:${view.status}`;
  if (visualKey!==nextKey) {visualKey=nextKey;drawDetails(view);}
  updateStart();
}
async function loadRows(force=false) {
  if (!selectedId || !current || current.id!==selectedId) return;
  const id=selectedId;
  const query=new URLSearchParams({page:String(page),winner:$("winner-filter").value,reason:$("reason-filter").value,color:$("color-filter").value});
  const key=`${id}:${current.completed}:${query}`;
  if (!force && rowsKey===key) return;
  const token=++rowsRequest;
  const data=await api(`/api/matches/${id}/games?${query}`);
  if (id!==selectedId || token!==rowsRequest) return;
  rowsKey=key;page=data.page;pages=data.pages;
  $("game-rows").replaceChildren();
  for (const game of data.items) {
    const row=node("tr"), winner=node("td"), replayCell=node("td"), replayButton=node("button","보기 ↗","replay-link");
    winner.append(node("span",game.winner.toUpperCase(),`letter ${game.winner}`));
    replayButton.setAttribute("aria-label",`${game.index}판 기보 보기`);replayButton.addEventListener("click",()=>openReplay(id,game.index));replayCell.append(replayButton);
    row.append(node("td",number(game.index)),node("td",game.model_a_color==="Black"?"흑 · 선공":"백 · 후공"),winner,
      node("td",reasons[game.reason]),node("td",`${game.plies}수`),node("td",`${game.territory.black} : ${game.territory.white}`),replayCell);
    $("game-rows").append(row);
  }
  if (!data.items.length) {const row=node("tr"), cell=node("td","표시할 완료 대국이 없습니다.");cell.colSpan=7;row.append(cell);$("game-rows").append(row);}
  $("filtered-count").textContent=`${number(data.total)}판`;
  $("page-label").textContent=`${page} / ${pages} 페이지`;
  $("page-prev").disabled=page<=1;$("page-next").disabled=page>=pages;
}
async function loadSelected() {
  if (!selectedId) return;
  const id=selectedId, view=await api(`/api/matches/${id}`);
  if (id!==selectedId) return;
  render(view);await loadRows();
}
async function openReplay(id,index) {
  const token=++replayRequest;
  replay=null;$("replay-title").textContent=`${index}판 기보`;
  $("replay-summary").textContent="저장된 수순을 불러오고 있습니다.";$("replay-board").replaceChildren();
  $("replay-step").textContent="";$("replay-slider").disabled=true;$("replay-prev").disabled=true;$("replay-next").disabled=true;
  if (!$("replay-dialog").open) $("replay-dialog").showModal();
  try {
    const data=await api(`/api/matches/${id}/replay/${index}`);
    if (token!==replayRequest || !$("replay-dialog").open) return;
    replay=data;const game=data.record;
    $("replay-summary").textContent=`모델 ${game.winner.toUpperCase()} 승리 · ${reasons[game.reason]} · A ${game.model_a_color==="Black"?"흑/선공":"백/후공"} · ${game.plies}수`;
    $("replay-slider").max=String(data.frames.length-1);$("replay-slider").value=$("replay-slider").max;$("replay-slider").disabled=false;
    drawReplay();
  } catch(error) {$("replay-summary").textContent=error.message;}
}
function drawReplay() {
  if (!replay) return;
  const step=Number($("replay-slider").value), frame=replay.frames[step], root=$("replay-board");root.replaceChildren();
  const grid=svg("svg",{viewBox:"0 0 1000 1000","aria-hidden":"true"});
  for (let i=0;i<9;i++) {
    const p=100+i*100;
    grid.append(svg("line",{x1:100,x2:900,y1:p,y2:p,stroke:"#ac8d5d","stroke-width":1.7}),svg("line",{x1:p,x2:p,y1:100,y2:900,stroke:"#ac8d5d","stroke-width":1.7}));
    grid.append(svg("text",{x:p,y:55,fill:"#9b7e50","font-size":25,"text-anchor":"middle"},i+1),svg("text",{x:48,y:p+8,fill:"#9b7e50","font-size":25,"text-anchor":"middle"},i+1));
  }
  root.append(grid);
  for (let action=0;action<81;action++) {
    let color=frame.cells[action], house=false;
    if (color==="Empty") {color=frame.ownership[action];house=true;}
    if (!["Black","White","Neutral"].includes(color)) continue;
    const stone=node("span",color==="Neutral"?"◇":"",`replay-stone ${color}${house?" house":""}${frame.action===action?" last":""}`);
    stone.style.left=`${10+action%9*10}%`;stone.style.top=`${10+Math.floor(action/9)*10}%`;root.append(stone);
  }
  const action=frame.action, description=action==null?"대국 시작":action===81?"패스":`${Math.floor(action/9)+1}행 ${action%9+1}열`;
  $("replay-step").textContent=`${step} / ${replay.frames.length-1}수 · ${description} · 집 흑 ${frame.score.black} : 백 ${frame.score.white}`;
  root.setAttribute("aria-label",`${step}수의 보드. ${description}, 흑 집 ${frame.score.black}칸, 백 집 ${frame.score.white}칸`);
  $("replay-prev").disabled=step===0;$("replay-next").disabled=step===replay.frames.length-1;
}
$("replay-slider").addEventListener("input",drawReplay);
for (const [id,delta] of [["replay-prev",-1],["replay-next",1]]) $(id).addEventListener("click",()=>{$("replay-slider").value=String(Number($("replay-slider").value)+delta);drawReplay();});
$("replay-close").addEventListener("click",()=>$("replay-dialog").close());
$("replay-dialog").addEventListener("close",()=>{replayRequest++;replay=null;});
$("run-select").addEventListener("change",async()=>{selectedId=$("run-select").value;current=null;page=1;visualKey="";rowsKey="";try{await loadSelected();}catch(error){notify(error.message);}});
for (const id of ["winner-filter","reason-filter","color-filter"]) $(id).addEventListener("change",async()=>{page=1;try{await loadRows(true);}catch(error){notify(error.message);}});
for (const [id,delta] of [["page-prev",-1],["page-next",1]]) $(id).addEventListener("click",async()=>{page=Math.max(1,Math.min(pages,page+delta));try{await loadRows(true);}catch(error){notify(error.message);}});
for (const format of ["csv","jsonl"]) $(`download-${format}`).addEventListener("click",event=>{if(event.currentTarget.getAttribute("aria-disabled")==="true") event.preventDefault();});
$("match-form").addEventListener("submit",async event=>{
  event.preventDefault();if(requesting) return;
  const settings={model_a:$("model-a").value,model_b:$("model-b").value,games:Number($("games").value),simulations:Number($("simulations").value),
    device:$("device").value,seed:Number($("seed").value),tactical_checks:$("tactical").checked,rating_a:Number($("rating-a").value),rating_b:Number($("rating-b").value),k:Number($("rating-k").value),
    workers:Number($("workers").value),backend:$("backend").value,leaf_batch_size:Number($("leaf-batch-size").value),reuse_tree:$("reuse-tree").checked};
  if(settings.model_a===settings.model_b){notify("서로 다른 모델 두 개를 선택해 주세요.");return;}
  requesting=true;updateStart();
  try {
    const view=await api("/api/matches/start",{settings});selectedId=view.id;page=1;rowsKey="";visualKey="";current=view;
    await loadCatalog();await loadSelected();notify(`${number(settings.games)}판 대결 · 동시 ${Math.min(settings.workers,settings.games)}판으로 시작했습니다.`);
  } catch(error){notify(error.message);}
  finally{requesting=false;updateStart();}
});
$("workers").addEventListener("input",()=>{workersEdited=true;});
$("device").addEventListener("change",()=>{if(!workersEdited) $("workers").value=$("device").value==="cuda"?"12":"4";});
$("backend").addEventListener("change",()=>{
  const legacy=$("backend").value==="legacy";
  $("leaf-batch-size").disabled=legacy;$("reuse-tree").disabled=legacy;
});
$("stop-match").addEventListener("click",()=>{stopId=selectedId;$("stop-dialog").showModal();});
$("stop-cancel").addEventListener("click",()=>$("stop-dialog").close());
$("stop-confirm").addEventListener("click",async()=>{
  $("stop-dialog").close();requesting=true;updateStart();
  try {await api(`/api/matches/${stopId}/stop`,{});await loadCatalog();await loadSelected();}
  catch(error){notify(error.message);}finally{requesting=false;updateStart();}
});
async function poll() {
  if(!requesting) {
    try {await loadCatalog();await loadSelected();connected=true;updateStart();}
    catch(error){connected=false;updateStart();$("live-description").textContent="연결이 끊겼습니다. 게임 실행 창이 켜져 있는지 확인해 주세요.";$("stop-match").disabled=true;}
  }
  setTimeout(poll,document.hidden?4000:1500);
}
async function initialize() {
  try {
    const data=await api("/api/models");
    for (const [id,index] of [["model-a",0],["model-b",1]]) {
      for(const model of data.models){const option=node("option",model.name.replace(/^runs\//,""));option.value=model.id;$(id).append(option);}
      $(id).selectedIndex=Math.min(index,data.models.length-1);
    }
    await loadCatalog();await loadSelected();
  } catch(error){notify(error.message);connected=false;updateStart();}
  setTimeout(poll,1500);
}
initialize();
