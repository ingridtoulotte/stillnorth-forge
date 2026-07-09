"use strict";

const STAGES = [
  ["flux", "FLUX images"], ["img_up", "Upscale ×2"], ["classify", "Classify"],
  ["vid1", "Wan clip 1"], ["lastframe", "Last frame"], ["lf_up", "Upscale ×4"],
  ["vid2", "Wan clip 2"], ["concat", "Concat"], ["final_up", "Final ×4"],
];
const STAGE_LABEL = Object.fromEntries(STAGES);

const $ = (id) => document.getElementById(id);
const dz = $("dropzone");

$("pipeline").innerHTML = STAGES.map(([k, label]) =>
  `<div class="node" data-stage="${k}">
     <span class="ncount" id="c-${k}">0</span>
     <span class="nlabel">${label}</span>
     <span class="nmeta" id="m-${k}"></span>
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
function fmtSize(b) { return b > 1e6 ? (b / 1e6).toFixed(1) + "MB" : Math.round(b / 1e3) + "KB"; }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---- theme ----------------------------------------------------------------
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  localStorage.setItem("snf-theme", t);
  $("theme").textContent = t === "dark" ? "◐" : "☀";
}
function toggleTheme() { applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"); }
applyTheme(localStorage.getItem("snf-theme") || "dark");
$("theme").addEventListener("click", toggleTheme);

// ---- ingest ---------------------------------------------------------------
async function ingestFiles(files) {
  let total = 0;
  for (const f of files) {
    if (!/\.html?$/i.test(f.name)) { toast(`skipped ${f.name} (not HTML)`); continue; }
    const res = await api("/api/ingest", { name: f.name, html: await f.text() });
    total += res.added || 0;
  }
  toast(`+${total} prompts queued`);
  refresh();
}
dz.addEventListener("click", () => $("file").click());
$("browse").addEventListener("click", (e) => { e.stopPropagation(); $("file").click(); });
$("file").addEventListener("change", (e) => ingestFiles(e.target.files));
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => ingestFiles(e.dataTransfer.files));

// ---- controls -------------------------------------------------------------
async function runPipeline() {
  const target = parseInt($("mode-target").value, 10) || null;
  const minutes = target ? null : (parseFloat($("mode-minutes").value) || null);
  const r = await api("/api/run", { target, minutes });
  const mode = target ? `until ${target} accepted vids` : minutes ? `for ${minutes} min` : "full set";
  toast(r.started ? `pipeline running — ${mode}` : "already running");
  refresh();
}
// the two mode inputs are alternatives: filling one clears the other
$("mode-target").addEventListener("input", () => { if ($("mode-target").value) $("mode-minutes").value = ""; });
$("mode-minutes").addEventListener("input", () => { if ($("mode-minutes").value) $("mode-target").value = ""; });
async function cancelPipeline() { await api("/api/cancel", {}); toast("pausing after current item…"); }
async function clearQueue() {
  if (!confirm("Clear the queued prompts? Rendered files are KEPT on disk (use Purge to delete them).")) return;
  await api("/api/clear", {}); toast("queue cleared"); refresh();
}
async function purgeOutputs() {
  if (!confirm("PURGE: permanently DELETE every rendered image/clip in the output workspace and reset to a clean slate.\n\nThis cannot be undone. Continue?")) return;
  toast("purging outputs…");
  const r = await api("/api/purge", {});
  toast(`purged — removed ${r.removed ?? 0} stage folders`);
  lastCounts = {}; refresh(); refreshLog(); loadGallery();
}
$("run").addEventListener("click", runPipeline);
$("cancel").addEventListener("click", cancelPipeline);
$("clear").addEventListener("click", clearQueue);
$("purge").addEventListener("click", purgeOutputs);

// ---- live status ----------------------------------------------------------
function setPill(id, ok, label) { const el = $(id); el.className = "pill " + (ok ? "ok" : "bad"); el.textContent = label + (ok ? " ✓" : " ✕"); }

let runStart = 0, lastCounts = {}, lastS = null, queueData = [], wsPath = "";
function elapsedSec() { return runStart ? (Date.now() - runStart) / 1000 : 0; }

function paintClock() {
  const s = lastS; if (!s) return;
  if (s.running) {
    const el = elapsedSec();
    let txt = "⏱ " + fmtDur(el), eta = "—", avg = "—";
    if (s.stage_done > 0 && s.stage_total > 0) {
      const per = el / s.stage_done;
      avg = fmtDur(per) + " · " + s.stage;
      if (s.stage_done < s.stage_total) { eta = "~" + fmtDur(per * (s.stage_total - s.stage_done)); txt += " · " + eta + " left"; }
    }
    $("stage-timer").textContent = txt;
    $("stat-elapsed").textContent = fmtDur(el);
    $("stat-avg").textContent = avg; $("stat-eta").textContent = eta;
  } else {
    $("stage-timer").textContent = "";
    $("stat-elapsed").textContent = "—"; $("stat-avg").textContent = "—"; $("stat-eta").textContent = "—";
  }
}

function renderQueue() {
  const f = $("queue-search").value.trim().toLowerCase();
  const q = queueData.filter((x) => !f || x.src.toLowerCase().includes(f));
  $("queue-total").textContent = queueData.length ? `(${queueData.reduce((a, x) => a + x.prompts, 0)} prompts)` : "";
  $("queue").innerHTML = q.length
    ? q.map((x) => `<li><span>${esc(x.src)}</span><span class="qn">${x.prompts}</span></li>`).join("")
    : `<div class="empty">${queueData.length ? "no match" : "empty — drop an HTML file to begin"}</div>`;
}
$("queue-search").addEventListener("input", renderQueue);

async function refresh() {
  let s; try { s = await api("/api/status"); } catch { return; }
  lastS = s;
  if (s.running && !runStart) runStart = Date.now();
  if (!s.running) runStart = 0;

  const labelEl = $("stage-label");
  labelEl.textContent = s.label || "idle";
  labelEl.className = "stage-label" + (s.label === "paused" ? " paused" : s.label === "error" ? " error" : s.label === "done" ? " done" : "");

  $("stage-pct").textContent = (s.percent || 0) + "%";
  const fill = $("bar-fill"); fill.style.width = (s.percent || 0) + "%"; fill.classList.toggle("idle", !s.running);

  let det;
  if (s.label === "error" && s.last_error) det = "error: " + s.last_error;
  else if (s.note) det = s.note;
  else if (s.stage_total) det = `${s.label} — ${s.stage_done}/${s.stage_total}`;
  else det = s.running ? "scanning…" : "idle";
  $("stage-detail").textContent = det + (s.totals ? `  ·  ${s.totals.prompts} prompts in set` : "");

  const counts = s.counts || {}, metrics = s.metrics || {};
  let fails = 0;
  STAGES.forEach(([k]) => {
    const cell = $("c-" + k), n = counts[k] ?? 0;
    if (lastCounts[k] !== undefined && n > lastCounts[k]) { cell.classList.remove("bump"); void cell.offsetWidth; cell.classList.add("bump"); }
    lastCounts[k] = n; cell.textContent = n;
    const node = document.querySelector(`.node[data-stage="${k}"]`);
    node.classList.toggle("active", s.stage === k && s.running);
    node.classList.toggle("done", n > 0 && s.stage !== k);
    const m = metrics[k];
    if (m) { fails += m.fail || 0; node.classList.toggle("hasfail", (m.fail || 0) > 0); }
    $("m-" + k).textContent = m && m.avg != null ? `~${fmtDur(m.avg)}${m.fail ? " · " + m.fail + "✕" : ""}` : "";
  });

  // finished masters = everything in the library (09_final_up4 + used buckets),
  // not just what is still sitting in final_up before the assembler moves it
  const masters = (s.library && s.library.total != null) ? s.library.total : (counts.final_up || 0);
  const setn = (s.totals && s.totals.prompts) || 0;
  $("overall-fill").style.width = (setn ? Math.round(Math.min(masters, setn) / setn * 100) : 0) + "%";
  $("overall-text").textContent = `${masters} / ${setn}`;
  $("stat-prompts").textContent = setn; $("stat-masters").textContent = masters; $("stat-fails").textContent = fails;

  if (s.vram) {
    $("vram-fill").style.width = s.vram.pct + "%";
    $("vram-val").textContent = `${s.vram.used} / ${s.vram.total} MB (${s.vram.pct}%)`;
    $("vram-name").textContent = s.vram.name || "";
  } else { $("vram-val").textContent = "nvidia-smi unavailable"; }
  setPill("pill-comfy", s.comfy, "ComfyUI");

  queueData = s.queue || []; renderQueue();
  $("run").disabled = !!s.running;
  $("run").textContent = s.running ? "● running" : "▶ Run / Resume";

  // mode / judge status line
  const md = s.mode || {};
  let mtxt = "";
  if (md.kind === "target") mtxt = `🎯 ${md.accepted ?? 0} / ${md.target} accepted`;
  else if (md.kind === "time" && s.running) mtxt = `⏲ ${fmtDur(md.seconds_left ?? 0)} left · ${md.accepted ?? 0} accepted`;
  if (md.review) mtxt += `${mtxt ? " · " : ""}⚠ ${md.review} in review`;
  if (md.judge === false) mtxt += `${mtxt ? " · " : ""}judge off`;
  $("mode-status").textContent = mtxt;
  paintClock();
}

// ---- activity feed --------------------------------------------------------
let logLines = [], logPaused = false;
function classifyLine(t) {
  t = t.toLowerCase();
  if (/fatal|fail|error|unreach|offline/.test(t) && !/cancelled/.test(t)) return "bad";
  if (/cancel|paused|interrupt|purge|retry|backoff/.test(t)) return "warn";
  if (/finished|complete|\+\d+ prompts/.test(t)) return "ok";
  return "";
}
function renderLog() {
  const f = $("log-search").value.trim().toLowerCase();
  const ul = $("log");
  let lines = logLines.slice().reverse();
  if (f) lines = lines.filter((l) => l.toLowerCase().includes(f));
  if (!lines.length) { ul.innerHTML = `<li class="empty">${logLines.length ? "no match" : "no activity yet"}</li>`; return; }
  ul.innerHTML = lines.map((ln) => {
    const m = ln.match(/^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s(.*)$/), cls = classifyLine(ln);
    return m ? `<li class="${cls}"><span class="lt">${m[1]}</span>  ${esc(m[2])}</li>` : `<li class="${cls}">${esc(ln)}</li>`;
  }).join("");
}
async function refreshLog() {
  if (logPaused) return;
  let d; try { d = await api("/api/log"); } catch { return; }
  logLines = d.lines || []; renderLog();
}
$("log-search").addEventListener("input", renderLog);
$("log-pause").addEventListener("click", () => {
  logPaused = !logPaused;
  $("log-pause").classList.toggle("on", logPaused);
  $("log-pause").textContent = logPaused ? "▶" : "⏸";
  $("log-pause").title = logPaused ? "resume autoscroll" : "pause autoscroll";
  if (!logPaused) refreshLog();
});
$("log-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(logLines.join("\n")); toast("log copied"); } catch { toast("copy failed"); }
});

// ---- output gallery -------------------------------------------------------
let galData = {}, galStage = localStorage.getItem("snf-gal") || "final_up";
const GAL_STAGES = STAGES.concat([["review", "⚠ Review"]]);
$("gal-tabs").innerHTML = GAL_STAGES.map(([k, label]) =>
  `<button class="gal-tab" data-stage="${k}">${label}</button>`).join("");
$("gal-tabs").addEventListener("click", (e) => {
  const b = e.target.closest(".gal-tab"); if (!b) return;
  galStage = b.dataset.stage; localStorage.setItem("snf-gal", galStage); renderGallery();
});
$("gal-refresh").addEventListener("click", loadGallery);

function pathSep() { return wsPath.includes("\\") ? "\\" : "/"; }
function absPath(rel) { const sep = pathSep(); return wsPath ? wsPath + sep + rel.replace(/\//g, sep) : rel; }

async function loadGallery() { try { galData = await api("/api/outputs"); } catch { return; } renderGallery(); }
function renderGallery() {
  $("gal-tabs").querySelectorAll(".gal-tab").forEach((b) => b.classList.toggle("sel", b.dataset.stage === galStage));
  const bucket = galData[galStage] || { files: [], total: 0 };
  const g = $("gallery");
  if (!bucket.files.length) { g.innerHTML = `<div class="empty">no files in ${STAGE_LABEL[galStage]} yet</div>`; return; }
  g.innerHTML = bucket.files.map((f) => {
    const src = "/api/file?path=" + encodeURIComponent(f.rel);
    const media = f.kind === "video"
      ? `<video src="${src}" preload="metadata" muted playsinline></video>`
      : `<img src="${src}" loading="lazy" alt="${esc(f.name)}" />`;
    return `<div class="tile" data-rel="${esc(f.rel)}" data-kind="${f.kind}" data-name="${esc(f.name)}">
       ${media}
       <span class="tkind">${f.kind === "video" ? "▶" : "img"}</span>
       <button class="mini-btn tcopy" title="copy file path">⧉</button>
       <div class="tcap">${esc(f.name)} · ${fmtSize(f.size)}</div>
     </div>`;
  }).join("") + (bucket.total > bucket.files.length ? `<div class="empty">+${bucket.total - bucket.files.length} more…</div>` : "");
  // seek videos to a frame so the tile isn't black
  g.querySelectorAll("video").forEach((v) => v.addEventListener("loadedmetadata", () => { try { v.currentTime = Math.min(0.1, v.duration || 0.1); } catch {} }, { once: true }));
}
$("gallery").addEventListener("click", (e) => {
  const tile = e.target.closest(".tile"); if (!tile) return;
  if (e.target.closest(".tcopy")) {
    const p = absPath(tile.dataset.rel);
    navigator.clipboard.writeText(p).then(() => toast("path copied"), () => toast(p));
    return;
  }
  openLightbox(tile.dataset.rel, tile.dataset.kind, tile.dataset.name);
});

// ---- lightbox -------------------------------------------------------------
function openLightbox(rel, kind, name) {
  const src = "/api/file?path=" + encodeURIComponent(rel);
  $("lb-body").innerHTML = kind === "video"
    ? `<video src="${src}" controls autoplay loop playsinline></video>`
    : `<img src="${src}" alt="${esc(name)}" />`;
  $("lb-caption").textContent = absPath(rel);
  $("lightbox").hidden = false;
}
function closeLightbox() { $("lightbox").hidden = true; $("lb-body").innerHTML = ""; }
$("lb-close").addEventListener("click", closeLightbox);
$("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") closeLightbox(); });

// ---- command palette ------------------------------------------------------
const COMMANDS = [
  { name: "Run / Resume pipeline", keys: "R", run: runPipeline },
  { name: "Cancel (pause)", keys: "C", run: cancelPipeline },
  { name: "Clear queue", keys: "", run: clearQueue },
  { name: "Purge outputs (delete all)", keys: "", run: purgeOutputs },
  { name: "Toggle light / dark theme", keys: "T", run: toggleTheme },
  { name: "Refresh output gallery", keys: "G", run: loadGallery },
  { name: "Build long video (compilation)", keys: "", run: () => { location.hash = "#sec-assembler"; buildLong(); } },
  { name: "Stop long-video build", keys: "", run: cancelLong },
  { name: "Clean aged intermediates now", keys: "", run: cleanupNow },
  { name: "Jump: Long video assembler", keys: "", run: () => location.hash = "#sec-assembler" },
  { name: "Copy output folder path", keys: "", run: () => navigator.clipboard.writeText(wsPath).then(() => toast("path copied")) },
  { name: "Copy activity log", keys: "", run: () => navigator.clipboard.writeText(logLines.join("\n")).then(() => toast("log copied")) },
  { name: "Jump: Input", keys: "", run: () => location.hash = "#sec-input" },
  { name: "Jump: Progress", keys: "", run: () => location.hash = "#sec-progress" },
  { name: "Jump: Outputs", keys: "", run: () => location.hash = "#sec-gallery" },
];
let palSel = 0, palItems = COMMANDS;
function openPalette() {
  $("palette").hidden = false; $("palette-input").value = ""; renderPalette(""); $("palette-input").focus();
}
function closePalette() { $("palette").hidden = true; }
function renderPalette(q) {
  q = q.trim().toLowerCase();
  palItems = COMMANDS.filter((c) => !q || c.name.toLowerCase().includes(q));
  palSel = 0;
  $("palette-list").innerHTML = palItems.map((c, i) =>
    `<li class="${i === 0 ? "sel" : ""}" data-i="${i}">${esc(c.name)}<span class="pk">${c.keys}</span></li>`).join("")
    || `<li class="empty">no command</li>`;
}
function palMove(d) {
  if (!palItems.length) return;
  palSel = (palSel + d + palItems.length) % palItems.length;
  $("palette-list").querySelectorAll("li").forEach((li, i) => li.classList.toggle("sel", i === palSel));
}
function palRun() { const c = palItems[palSel]; if (c) { closePalette(); c.run(); } }
$("palette-input").addEventListener("input", (e) => renderPalette(e.target.value));
$("palette-list").addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) { palSel = +li.dataset.i; palRun(); } });
$("open-palette").addEventListener("click", openPalette);

// ---- keyboard -------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); $("palette").hidden ? openPalette() : closePalette(); return; }
  if (e.key === "Escape") { if (!$("palette").hidden) closePalette(); if (!$("lightbox").hidden) closeLightbox(); return; }
  if (!$("palette").hidden) {
    if (e.key === "ArrowDown") { e.preventDefault(); palMove(1); }
    if (e.key === "ArrowUp") { e.preventDefault(); palMove(-1); }
    if (e.key === "Enter") { e.preventDefault(); palRun(); }
    return;
  }
  if (e.target.matches("input,textarea") || e.metaKey || e.ctrlKey) return;
  const k = e.key.toLowerCase();
  if (k === "r" && !$("run").disabled) runPipeline();
  else if (k === "c") cancelPipeline();
  else if (k === "t") toggleTheme();
  else if (k === "g") { loadGallery(); location.hash = "#sec-gallery"; }
});

// ---- preview ("last done only") -------------------------------------------
let prevRel = "";
async function refreshPreview() {
  let d; try { d = await api("/api/preview"); } catch { return; }
  const last = d.last, box = $("preview");
  if (!last) { if (prevRel) { box.innerHTML = `<div class="empty">nothing rendered yet</div>`; prevRel = ""; } $("prev-meta").textContent = ""; return; }
  $("prev-meta").textContent = `· ${STAGE_LABEL[last.stage] || last.stage}`;
  if (last.rel === prevRel) return;       // unchanged — don't reload the media
  prevRel = last.rel;
  const src = "/api/file?path=" + encodeURIComponent(last.rel);
  const m = last.kind === "video"
    ? `<video src="${src}" controls autoplay loop muted playsinline></video>`
    : `<img src="${src}" alt="${esc(last.name)}" />`;
  box.innerHTML = `${m}<div class="prev-cap">${esc(last.name)}</div>`;
}

// ---- long-video assembler -------------------------------------------------
const DUR_LABEL = { "15min": "15 min", "30min": "30 min", "1h": "1 hour", "2h": "2 h", "3h": "3 h", "4h": "4 h", "6h": "6 h", "8h": "8 h", "10h": "10 h", "12h": "12 h" };
const BUCKET_LABEL = ["Never used", "Used 1×", "Used 2×", "Used 3×", "Used 4×+"];
let asmDur = localStorage.getItem("snf-asm-dur") || "1h";
let asmWeights = null, asmDursBuilt = false, asmBusy = false;

function renderDurations(durs) {
  if (asmDursBuilt) return;
  $("asm-durations").innerHTML = durs.map((d) =>
    `<button class="dur-btn" data-key="${d.key}">${DUR_LABEL[d.key] || d.key}</button>`).join("");
  $("asm-durations").addEventListener("click", (e) => {
    const b = e.target.closest(".dur-btn"); if (!b) return;
    asmDur = b.dataset.key; localStorage.setItem("snf-asm-dur", asmDur); markDuration();
  });
  asmDursBuilt = true;
}
function markDuration() {
  $("asm-durations").querySelectorAll(".dur-btn").forEach((b) => b.classList.toggle("sel", b.dataset.key === asmDur));
}

function renderWeights(buckets) {
  const w = asmWeights || {};
  $("asm-weights").innerHTML = [0, 1, 2, 3, 4].map((lvl) => {
    const have = (buckets && buckets[lvl]) || 0;
    const pct = Math.round(+w[lvl] || 0);
    return `<div class="wrow">
      <span class="wlabel">${BUCKET_LABEL[lvl]}<span class="whave">${have} clip${have === 1 ? "" : "s"}</span></span>
      <span class="wctl">
        <button class="mini-btn wdec" data-lvl="${lvl}" title="less">◀</button>
        <span class="wval" id="wval-${lvl}">${pct}%</span>
        <button class="mini-btn winc" data-lvl="${lvl}" title="more">▶</button>
      </span>
    </div>`;
  }).join("");
}
function normalizeWeights() {
  const t = [0, 1, 2, 3, 4].reduce((a, l) => a + (+asmWeights[l] || 0), 0);
  if (t <= 0) { asmWeights = { 0: 100, 1: 0, 2: 0, 3: 0, 4: 0 }; return; }
  [0, 1, 2, 3, 4].forEach((l) => { asmWeights[l] = Math.round((+asmWeights[l] || 0) / t * 100); });
}
$("asm-weights").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b || !asmWeights) return;
  const lvl = +b.dataset.lvl;
  const step = b.classList.contains("winc") ? 5 : -5;
  asmWeights[lvl] = Math.max(0, (+asmWeights[lvl] || 0) + step);
  normalizeWeights();
  [0, 1, 2, 3, 4].forEach((l) => { const el = $("wval-" + l); if (el) el.textContent = Math.round(+asmWeights[l] || 0) + "%"; });
  clearTimeout(saveWeights._t);
  saveWeights._t = setTimeout(saveWeights, 600);
});
async function saveWeights() { try { await api("/api/weights", { weights: asmWeights }); } catch {} }

function renderCompilations(list) {
  const ul = $("asm-comp-list");
  if (!list || !list.length) { ul.innerHTML = `<li class="empty">none yet — build one above</li>`; return; }
  ul.innerHTML = list.map((c) =>
    `<li><span class="cname" title="${esc(c.path)}">${esc(c.name)}</span>
       <span class="cmeta">${fmtSize(c.size)}</span>
       <button class="mini-btn ccopy" data-path="${esc(c.path)}" title="copy path">⧉</button></li>`).join("");
}
$("asm-comp-list").addEventListener("click", (e) => {
  const b = e.target.closest(".ccopy"); if (!b) return;
  navigator.clipboard.writeText(b.dataset.path).then(() => toast("path copied"), () => toast(b.dataset.path));
});

async function loadAssembler() {
  let d; try { d = await api("/api/library"); } catch { return; }
  renderDurations(d.durations || []); markDuration();
  if (asmWeights === null) {
    const w = d.weights || {}; asmWeights = {};
    [0, 1, 2, 3, 4].forEach((l) => { asmWeights[l] = +w[l] != null ? +w[l] : (+w[String(l)] || 0); });
    normalizeWeights();
  }
  renderWeights(d.buckets);
  renderCompilations(d.compilations);
  const b = d.buckets || {};
  $("asm-libtotal").textContent = b.total != null ? `· ${b.total} masters available` : "";
  $("asm-retention-text").innerHTML = d.autodelete
    ? `Intermediate FLUX images, first clips, last frames & concats are auto-deleted <b>${d.retention_days} days</b> after a master is first used in a long video. Finished masters are always kept.`
    : `Auto-clean is <b>off</b>. Use the button to prune aged intermediates on demand.`;

  // build status
  const s = d.status || {};
  asmBusy = !!s.running;
  $("asm-phase").textContent = s.phase || "idle";
  $("asm-phase").className = "stage-label small" + (s.phase === "done" ? " done" : s.phase === "error" ? " error" : s.phase === "paused" ? " paused" : "");
  $("asm-pct").textContent = (s.percent || 0) + "%";
  const fill = $("asm-fill"); fill.style.width = (s.percent || 0) + "%"; fill.classList.toggle("idle", !s.running);
  let note = s.note || (s.running ? "working…" : "pick a length, set the mix, press Build");
  if (s.phase === "error" && s.last_error) note = "error: " + s.last_error;
  $("asm-note").textContent = note;
  $("asm-build").disabled = asmBusy;
  $("asm-build").textContent = asmBusy ? "● building…" : "🎞 Build long video";
}
async function buildLong() {
  if (asmBusy) { toast("a build is already running"); return; }
  const r = await api("/api/assemble", { duration: asmDur, weights: asmWeights });
  toast(r.ok ? `building ${DUR_LABEL[asmDur] || asmDur} compilation…` : (r.msg || "could not start"));
  loadAssembler();
}
async function cancelLong() { await api("/api/assemble/cancel", {}); toast("stopping build after current step…"); }
async function cleanupNow() {
  if (!confirm("Delete intermediate build files (FLUX images, 5s clips, last frames, concats) for masters first used over the retention period ago?\n\nFinished masters and compilations are kept. Continue?")) return;
  const r = await api("/api/cleanup", {});
  toast(`auto-clean removed ${r.removed ?? 0} intermediate files`);
  refresh();
}
$("asm-build").addEventListener("click", buildLong);
$("asm-cancel").addEventListener("click", cancelLong);
$("asm-cleanup").addEventListener("click", cleanupNow);

// ---- environment ----------------------------------------------------------
async function health() {
  try {
    const h = await api("/api/health");
    setPill("pill-comfy", h.comfy, "ComfyUI");
    setPill("pill-ffmpeg", h.ffmpeg, "ffmpeg");
    if (h.workspace) { wsPath = h.workspace; $("ws-path").textContent = h.workspace; }
  } catch {}
}
$("ws-copy").addEventListener("click", () => {
  navigator.clipboard.writeText(wsPath).then(() => toast("output path copied"), () => toast(wsPath));
});

// boot
health(); refresh(); refreshLog(); loadGallery(); loadAssembler(); refreshPreview();
setInterval(refresh, 1000);
setInterval(refreshLog, 2500);
setInterval(paintClock, 1000);
setInterval(loadGallery, 8000);
setInterval(loadAssembler, 2000);
setInterval(refreshPreview, 3000);
