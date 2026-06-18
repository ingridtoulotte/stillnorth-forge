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

function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0");
}
function esc(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

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
async function cancelPipeline() { await api("/api/cancel", {}); toast("pausing after current item…"); }

$("run").addEventListener("click", runPipeline);
$("cancel").addEventListener("click", cancelPipeline);
$("clear").addEventListener("click", async () => {
  if (!confirm("Clear the queued prompts? Rendered files are KEPT on disk (use Purge to delete them).")) return;
  await api("/api/clear", {}); toast("queue cleared"); refresh();
});
$("purge").addEventListener("click", async () => {
  if (!confirm("PURGE: permanently DELETE every rendered image/clip in the output workspace and reset to a clean slate.\n\nThis cannot be undone. Continue?")) return;
  toast("purging outputs…");
  const r = await api("/api/purge", {});
  toast(`purged — removed ${r.removed ?? 0} stage folders`);
  lastCounts = {};
  refresh(); refreshLog();
});

// keyboard shortcuts: R = run/resume, C = cancel (ignored while typing)
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input,textarea") || e.metaKey || e.ctrlKey) return;
  if (e.key === "r" || e.key === "R") { if (!$("run").disabled) runPipeline(); }
  if (e.key === "c" || e.key === "C") cancelPipeline();
});

// ---- live status ----------------------------------------------------------
function setPill(id, ok, label) {
  const el = $(id);
  el.className = "pill " + (ok ? "ok" : "bad");
  el.textContent = label + (ok ? " ✓" : " ✕");
}

let runStart = 0;        // ms timestamp the current run began (client-side clock)
let lastCounts = {};     // for the count-bump animation
let lastS = null;        // last status snapshot (so the 1s clock can repaint)

function elapsedSec() { return runStart ? (Date.now() - runStart) / 1000 : 0; }

// repaint everything time-derived; called on each poll AND every second
function paintClock() {
  const s = lastS;
  if (!s) return;
  if (s.running) {
    const el = elapsedSec();
    // main stage timer + ETA
    let txt = "⏱ " + fmtDur(el);
    let eta = "—", avg = "—";
    if (s.stage_done > 0 && s.stage_total > 0) {
      const per = el / s.stage_done;
      avg = fmtDur(per) + " · " + s.stage;
      if (s.stage_done < s.stage_total) {
        eta = "~" + fmtDur(per * (s.stage_total - s.stage_done));
        txt += " · " + eta + " left";
      }
    }
    $("stage-timer").textContent = txt;
    $("stat-elapsed").textContent = fmtDur(el);
    $("stat-avg").textContent = avg;
    $("stat-eta").textContent = eta;
  } else {
    $("stage-timer").textContent = "";
    $("stat-elapsed").textContent = "—";
    $("stat-avg").textContent = "—";
    $("stat-eta").textContent = "—";
  }
}

async function refresh() {
  let s;
  try { s = await api("/api/status"); } catch { return; }
  lastS = s;

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

  // detail line: prefer live "rendering x/y…" note
  let det;
  if (s.label === "error" && s.last_error) det = "error: " + s.last_error;
  else if (s.note) det = s.note;
  else if (s.stage_total) det = `${s.label} — ${s.stage_done}/${s.stage_total}`;
  else det = s.running ? "scanning…" : "idle";
  $("stage-detail").textContent = det + (s.totals ? `  ·  ${s.totals.prompts} prompts in set` : "");

  // pipeline node counts + active/done with a bump when a count grows
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

  // overall + sidebar stats
  const masters = counts.final_up || 0;
  const setn = (s.totals && s.totals.prompts) || 0;
  $("overall-fill").style.width = (setn ? Math.round(masters / setn * 100) : 0) + "%";
  $("overall-text").textContent = `${masters} / ${setn}`;
  $("stat-prompts").textContent = setn;
  $("stat-masters").textContent = masters;

  // vram (sidebar)
  if (s.vram) {
    $("vram-fill").style.width = s.vram.pct + "%";
    $("vram-val").textContent = `${s.vram.used} / ${s.vram.total} MB (${s.vram.pct}%)`;
    $("vram-name").textContent = s.vram.name || "";
  } else {
    $("vram-val").textContent = "nvidia-smi unavailable";
  }

  setPill("pill-comfy", s.comfy, "ComfyUI");

  // queue
  const q = s.queue || [];
  $("queue-total").textContent = q.length ? `(${q.reduce((a, x) => a + x.prompts, 0)} prompts)` : "";
  $("queue").innerHTML = q.length
    ? q.map((x) => `<li><span>${esc(x.src)}</span><span class="qn">${x.prompts}</span></li>`).join("")
    : `<div class="empty">empty — drop an HTML file to begin</div>`;

  // run button
  $("run").disabled = !!s.running;
  $("run").textContent = s.running ? "● running" : "▶ Run / Resume";

  paintClock();
}

// ---- activity feed (tail of forge.log) ------------------------------------
function classifyLine(text) {
  const t = text.toLowerCase();
  if (/fatal|fail|error|unreachable/.test(t) && !/cancelled/.test(t)) return "bad";
  if (/cancel|paused|interrupt|purge/.test(t)) return "warn";
  if (/finished|complete|\+\d+ prompts/.test(t)) return "ok";
  return "";
}
async function refreshLog() {
  let d;
  try { d = await api("/api/log"); } catch { return; }
  const lines = d.lines || [];
  const ul = $("log");
  if (!lines.length) { ul.innerHTML = `<li class="empty">no activity yet</li>`; return; }
  ul.innerHTML = lines.slice().reverse().map((ln) => {
    const m = ln.match(/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s(.*)$/);
    const cls = classifyLine(ln);
    return m
      ? `<li class="${cls}"><span class="lt">${m[1]}</span>  ${esc(m[2])}</li>`
      : `<li class="${cls}">${esc(ln)}</li>`;
  }).join("");
}

// ---- environment / workspace ----------------------------------------------
async function health() {
  try {
    const h = await api("/api/health");
    setPill("pill-comfy", h.comfy, "ComfyUI");
    setPill("pill-ffmpeg", h.ffmpeg, "ffmpeg");
    if (h.workspace) { $("ws-path").textContent = h.workspace; $("ws-path").dataset.path = h.workspace; }
  } catch {}
}
$("ws-copy").addEventListener("click", async () => {
  const p = $("ws-path").dataset.path || $("ws-path").textContent;
  try { await navigator.clipboard.writeText(p); toast("output path copied"); }
  catch { toast(p); }
});

// boot
health();
refresh();
refreshLog();
setInterval(refresh, 1000);
setInterval(refreshLog, 2500);
setInterval(paintClock, 1000);
