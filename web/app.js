"use strict";

const STAGES = [
  ["flux", "FLUX images"], ["img_up", "Upscale ×2"], ["classify", "Classify"],
  ["vid1", "Wan clip 1"], ["lastframe", "Last frame"], ["lf_up", "Upscale ×4"],
  ["vid2", "Wan clip 2"], ["concat", "Concat"], ["final_up", "Final ×4"],
];

const $ = (id) => document.getElementById(id);
const dz = $("dropzone");

// build the pipeline strip once
$("pipeline").innerHTML = STAGES.map(([k, label]) =>
  `<div class="node" data-stage="${k}">
     <span class="ncount" id="c-${k}">0</span>
     <span class="nlabel">${label}</span>
   </div>`).join("");

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.textContent = ""), 4000);
}

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  return r.json();
}

// ---- ingest HTML files ----------------------------------------------------
async function ingestFiles(files) {
  let total = 0;
  for (const f of files) {
    if (!/\.html?$/i.test(f.name)) { toast(`skipped ${f.name} (not HTML)`); continue; }
    const html = await f.text();
    const res = await api("/api/ingest", { name: f.name, html });
    total += res.added || 0;
  }
  toast(`+${total} prompts queued`);
  refresh();
}

dz.addEventListener("click", () => $("file").click());
$("browse").addEventListener("click", (e) => { e.stopPropagation(); $("file").click(); });
$("file").addEventListener("change", (e) => ingestFiles(e.target.files));

["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => ingestFiles(e.dataTransfer.files));

// ---- controls -------------------------------------------------------------
async function runPipeline() {
  const r = await api("/api/run", {});
  toast(r.started ? "pipeline running" : "already running");
  refresh();
}
$("run").addEventListener("click", runPipeline);
$("cancel").addEventListener("click", async () => { await api("/api/cancel", {}); toast("pausing after current item…"); });
$("clear").addEventListener("click", async () => {
  if (!confirm("Clear the queued prompts? Rendered files are kept on disk.")) return;
  await api("/api/clear", {}); toast("queue cleared"); refresh();
});

// keyboard shortcuts: R = run/resume, C = cancel (ignored while typing)
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input,textarea") || e.metaKey || e.ctrlKey) return;
  if (e.key === "r" || e.key === "R") { if (!$("run").disabled) runPipeline(); }
  if (e.key === "c" || e.key === "C") { api("/api/cancel", {}); toast("pausing after current item…"); }
});

// ---- live status ----------------------------------------------------------
function setPill(id, ok, label) {
  const el = $(id);
  el.className = "pill " + (ok ? "ok" : "bad");
  el.textContent = label + (ok ? " ✓" : " ✕");
}

function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m + ":" + String(s).padStart(2, "0");
}

// run-clock + ETA state (client-side, so it ticks every second even between polls)
let runStart = 0;        // ms timestamp when the current run began
let lastCounts = {};     // for the count-bump animation

function elapsedSec() { return runStart ? (Date.now() - runStart) / 1000 : 0; }

async function refresh() {
  let s;
  try { s = await api("/api/status"); } catch { return; }

  // run-clock bookkeeping
  if (s.running && !runStart) runStart = Date.now();
  if (!s.running) runStart = 0;

  // stage label + state colouring
  const labelEl = $("stage-label");
  labelEl.textContent = s.label || "idle";
  labelEl.className = "stage-label" +
    (s.label === "paused" ? " paused" : s.label === "error" ? " error" : s.label === "done" ? " done" : "");

  $("stage-pct").textContent = (s.percent || 0) + "%";
  const fill = $("bar-fill");
  fill.style.width = (s.percent || 0) + "%";
  fill.classList.toggle("idle", !s.running);

  // detail line: prefer the live "rendering x/y…" note, else fall back
  let det;
  if (s.note) det = s.note;
  else if (s.stage_total) det = `${s.label} — ${s.stage_done}/${s.stage_total}`;
  else if (s.last_error) det = "error: " + s.last_error;
  else det = s.running ? "scanning…" : "idle";
  if (s.last_error && s.label === "error") det = "error: " + s.last_error;
  $("stage-detail").textContent = det + (s.totals ? `  ·  ${s.totals.prompts} prompts in set` : "");

  // elapsed timer + rough ETA from completed items this stage
  if (s.running) {
    const el = elapsedSec();
    let txt = "⏱ " + fmtDur(el);
    if (s.stage_done > 0 && s.stage_total > 0 && s.stage_done < s.stage_total) {
      const per = el / s.stage_done;
      txt += " · ~" + fmtDur(per * (s.stage_total - s.stage_done)) + " left (stage)";
    }
    $("stage-timer").textContent = txt;
  } else {
    $("stage-timer").textContent = "";
  }

  // pipeline node counts + active/done, with a little bump when a count grows
  const counts = s.counts || {};
  STAGES.forEach(([k]) => {
    const cell = $("c-" + k);
    const n = counts[k] ?? 0;
    if (lastCounts[k] !== undefined && n > lastCounts[k]) {
      cell.classList.remove("bump"); void cell.offsetWidth; cell.classList.add("bump");
    }
    lastCounts[k] = n;
    cell.textContent = n;
    const node = document.querySelector(`.node[data-stage="${k}"]`);
    node.classList.toggle("active", s.stage === k && s.running);
    node.classList.toggle("done", n > 0 && s.stage !== k);
  });

  // overall completion = finished masters / prompts in set
  const masters = counts.final_up || 0;
  const setn = (s.totals && s.totals.prompts) || 0;
  $("overall-fill").style.width = (setn ? Math.round(masters / setn * 100) : 0) + "%";
  $("overall-text").textContent = `${masters} / ${setn}`;

  // vram
  if (s.vram) {
    $("vram-fill").style.width = s.vram.pct + "%";
    $("vram-val").textContent = `${s.vram.used} / ${s.vram.total} MB (${s.vram.pct}%)`;
    $("vram-name").textContent = s.vram.name || "";
  } else {
    $("vram-val").textContent = "nvidia-smi unavailable";
  }

  // health
  setPill("pill-comfy", s.comfy, "ComfyUI");

  // queue
  const q = s.queue || [];
  $("queue-total").textContent = q.length ? `(${q.reduce((a, x) => a + x.prompts, 0)} prompts)` : "";
  $("queue").innerHTML = q.length
    ? q.map((x) => `<li><span>${x.src}</span><span class="qn">${x.prompts}</span></li>`).join("")
    : `<div class="empty">empty — drop an HTML file to begin</div>`;

  // run button state
  $("run").disabled = !!s.running;
  $("run").textContent = s.running ? "● running" : "▶ Run / Resume";
}

// ---- activity feed (tail of forge.log) ------------------------------------
function classifyLine(text) {
  const t = text.toLowerCase();
  if (/fatal|fail|error|unreachable/.test(t) && !/cancelled/.test(t)) return "bad";
  if (/cancel|paused|interrupt/.test(t)) return "warn";
  if (/finished|complete|\+\d+ prompts/.test(t)) return "ok";
  return "";
}
async function refreshLog() {
  let d;
  try { d = await api("/api/log"); } catch { return; }
  const lines = d.lines || [];
  if (!lines.length) return;
  const ul = $("log");
  ul.innerHTML = lines.slice().reverse().map((ln) => {
    const m = ln.match(/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s(.*)$/);
    const cls = classifyLine(ln);
    return m
      ? `<li class="${cls}"><span class="lt">${m[1]}</span>  ${esc(m[2])}</li>`
      : `<li class="${cls}">${esc(ln)}</li>`;
  }).join("");
}
function esc(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

async function health() {
  try {
    const h = await api("/api/health");
    setPill("pill-comfy", h.comfy, "ComfyUI");
    setPill("pill-ffmpeg", h.ffmpeg, "ffmpeg");
  } catch {}
}

// tick the elapsed clock every second without hammering the API
setInterval(() => {
  if (runStart) {
    const cur = $("stage-timer").textContent;
    if (cur.startsWith("⏱")) $("stage-timer").textContent = "⏱ " + fmtDur(elapsedSec()) + cur.replace(/^⏱ [0-9:]+/, "");
  }
}, 1000);

health();
refresh();
refreshLog();
setInterval(refresh, 1000);
setInterval(refreshLog, 2500);
