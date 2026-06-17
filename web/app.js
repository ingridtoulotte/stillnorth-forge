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
$("run").addEventListener("click", async () => {
  const r = await api("/api/run", {});
  toast(r.started ? "pipeline running" : "already running");
  refresh();
});
$("cancel").addEventListener("click", async () => { await api("/api/cancel", {}); toast("cancelling…"); });
$("clear").addEventListener("click", async () => {
  if (!confirm("Clear the queued prompts? Rendered files are kept on disk.")) return;
  await api("/api/clear", {}); toast("queue cleared"); refresh();
});

// ---- live status ----------------------------------------------------------
function setPill(id, ok, label) {
  const el = $(id);
  el.className = "pill " + (ok ? "ok" : "bad");
  el.textContent = label + (ok ? " ✓" : " ✕");
}

async function refresh() {
  let s;
  try { s = await api("/api/status"); } catch { return; }

  $("stage-label").textContent = s.label || "idle";
  $("stage-pct").textContent = (s.percent || 0) + "%";
  $("bar-fill").style.width = (s.percent || 0) + "%";

  const det = s.stage_total
    ? `${s.label} — ${s.stage_done}/${s.stage_total}`
    : (s.running ? "scanning…" : (s.last_error ? "error: " + s.last_error : "idle"));
  $("stage-detail").textContent = det + (s.totals ? `  ·  ${s.totals.prompts} prompts in set` : "");

  // pipeline node counts + active/done
  const counts = s.counts || {};
  STAGES.forEach(([k]) => {
    $("c-" + k).textContent = counts[k] ?? 0;
    const node = document.querySelector(`.node[data-stage="${k}"]`);
    node.classList.toggle("active", s.stage === k && s.running);
    node.classList.toggle("done", (counts[k] || 0) > 0 && s.stage !== k);
  });

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

async function health() {
  try {
    const h = await api("/api/health");
    setPill("pill-comfy", h.comfy, "ComfyUI");
    setPill("pill-ffmpeg", h.ffmpeg, "ffmpeg");
  } catch {}
}

health();
refresh();
setInterval(refresh, 1000);
