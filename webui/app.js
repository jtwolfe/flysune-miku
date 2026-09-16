const AVOCADO =
  "Avocados grow on trees. The trees are tall, and the fruit is green. When avocados are ripe, people pick them. The fruit has a large seed inside.";

const STAGE_MS = { context: 180, kc: 420, vote: 500, speak: 320 };
const STAGES = ["context", "kc", "vote", "speak"];

const state = {
  trace: null,
  speed: 1,
  audition: true,
  skip: false,
  paused: false,
  revealing: false,
  audio: null,
  crumbAudio: null,
};

const $ = (id) => document.getElementById(id);

function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg || "";
  el.className = "status" + (kind ? " " + kind : "");
}

function flatten(trace) {
  const out = [];
  trace.tokens.forEach((tok, ti) => {
    tok.slots.forEach((_, si) => out.push({ token: ti, slot: si }));
  });
  return out;
}

function slotAt(trace, ref) {
  return trace.tokens[ref.token].slots[ref.slot];
}

function playB64(b64, onended) {
  if (state.audio) {
    state.audio.pause();
    state.audio = null;
  }
  const a = new Audio("data:audio/wav;base64," + b64);
  state.audio = a;
  a.onended = () => {
    if (onended) onended();
    if (state.audio === a) state.audio = null;
  };
  return a.play().then(() => a).catch(() => a);
}

async function wait(ms) {
  const end = performance.now() + ms;
  while (performance.now() < end) {
    if (state.skip) return;
    if (state.paused) {
      await new Promise((r) => setTimeout(r, 50));
      continue;
    }
    await new Promise((r) => setTimeout(r, 16));
  }
}

function setStage(name) {
  document.querySelectorAll(".stages li").forEach((li) => {
    li.classList.toggle("on", li.dataset.stage === name || (name === "done" && li.dataset.stage === "speak"));
  });
}

function drawWave(b64) {
  const canvas = $("wave");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#1b201d";
  ctx.fillRect(0, 0, w, h);
  if (!b64) return;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const sr = view.getUint32(24, true);
  const dataOff = 44;
  const n = Math.floor((bytes.length - dataOff) / 2);
  const buckets = w;
  const step = n / buckets;
  ctx.fillStyle = "rgba(197, 212, 204, 0.7)";
  for (let i = 0; i < buckets; i++) {
    const a = Math.floor(i * step);
    const b = Math.min(n, Math.floor((i + 1) * step));
    let m = 0;
    for (let j = a; j < b; j++) {
      m = Math.max(m, Math.abs(view.getInt16(dataOff + j * 2, true) / 32768));
    }
    const y = m * (h / 2 - 2);
    ctx.fillRect(i, h / 2 - y, 1, y * 2);
  }
  $("wav-meta").textContent = (n / sr).toFixed(2) + " s · " + sr + " Hz";
}

function drawKc(slot, progress) {
  const canvas = $("kc");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#1b201d";
  ctx.fillRect(0, 0, w, h);
  if (!slot) {
    $("kc-meta").textContent = "—";
    return;
  }
  const n = slot.nKc;
  const side = Math.ceil(Math.sqrt(n));
  const cell = Math.min(w, h) / side;
  const shown = Math.floor(slot.kcActive.length * Math.min(1, Math.max(0, progress)));
  const active = new Set(slot.kcActive.slice(0, shown));
  for (let i = 0; i < n; i++) {
    const x = (i % side) * cell;
    const y = Math.floor(i / side) * cell;
    if (active.has(i)) {
      ctx.fillStyle = "#6ee7b7";
      ctx.globalAlpha = 0.92;
    } else {
      ctx.fillStyle = "#2a322e";
      ctx.globalAlpha = 0.7;
    }
    ctx.fillRect(x + 0.4, y + 0.4, Math.max(0.6, cell - 0.8), Math.max(0.6, cell - 0.8));
  }
  ctx.globalAlpha = 1;
  $("kc-meta").textContent = slot.nKcActive + "/" + slot.nKc;
}

function renderLetters(slot) {
  const el = $("letters");
  if (!slot) {
    el.innerHTML = '<p class="muted">The 7-character window appears when a slot is live.</p>';
    $("letter-meta").textContent = "";
    $("prev-phone").textContent = "";
    return;
  }
  const chars = [...slot.letterContext];
  const center = Math.floor(chars.length / 2);
  el.innerHTML = chars
    .map(
      (c, i) =>
        `<span class="${i === center ? "focus" : ""}">${c === "_" ? "·" : c}</span>`,
    )
    .join("");
  $("letter-meta").textContent =
    "slot " + (slot.index + 1) + "/" + slot.nPhonemes + " · pos " + slot.phonemePos.toFixed(2);
  $("prev-phone").textContent =
    "Previous phone cue: " + (slot.previousPhone ?? "none (first slot)");
}

function renderVotes(slot, progress) {
  const ul = $("votes");
  if (!slot) {
    ul.innerHTML = "";
    $("vote-hint").textContent = "Waiting for a slot";
    return;
  }
  const nShow = Math.max(1, Math.round(slot.votes.length * Math.min(1, Math.max(0.08, progress))));
  const shown = slot.votes.slice(0, nShow);
  const maxAbs = Math.max(0.01, ...shown.map((v) => Math.abs(v.score)));
  ul.innerHTML = shown
    .map((v) => {
      const win = v.phoneme === slot.pickerPhone;
      const width = (Math.abs(v.score) / maxAbs) * 100;
      const klass = win ? "win" : v.isYes ? "yes" : "";
      const tag = (v.isYes ? "YES" : "NO") + (v.phoneme === slot.referencePhone ? " · ref" : "");
      return `<li><span style="color:${win ? "var(--kc)" : "var(--muted)"}">${v.phoneme}</span>
        <div class="bar"><i class="${klass}" style="width:${width}%"></i></div>
        <span>${tag}</span></li>`;
    })
    .join("");
  $("vote-hint").textContent =
    "Winner " + slot.pickerPhone + " · conf " + (slot.confidence * 100).toFixed(0) + "%";
}

function renderSpeaker(slot) {
  const params = $("params");
  if (!slot) {
    $("speaker-title").textContent = "—";
    $("speaker-desc").textContent =
      "After the choir votes, one speaker fly renders a 60–250 ms crumb from F0 / formants / noise.";
    params.innerHTML = "";
    $("crumb").disabled = true;
    return;
  }
  const s = slot.speaker;
  $("speaker-title").textContent = "/" + s.phoneme + "/";
  $("speaker-desc").textContent = "Speakers never see Kenyon cells. This crumb is phoneme-id only.";
  const rows = [
    ["F0", s.f0.toFixed(1) + " Hz"],
    ["F1×", s.f1Shift.toFixed(3)],
    ["F2×", s.f2Shift.toFixed(3)],
    ["Noise", s.noiseLevel.toFixed(3)],
    ["Dur", s.durationMs.toFixed(0) + " ms"],
    ["Rolloff", s.harmonicRolloff.toFixed(2)],
  ];
  params.innerHTML = rows
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join("");
  $("crumb").disabled = false;
  $("crumb").onclick = () => playB64(slot.crumbWavB64);
}

function showSlot(slot, stage) {
  renderLetters(slot);
  const kcP = stage === "context" ? 0.12 : stage === "kc" ? 0.7 : 1;
  const voteP =
    stage === "vote" || stage === "speak" || stage === "done"
      ? 1
      : stage === "kc"
        ? 0.2
        : 0.08;
  drawKc(slot, kcP);
  renderVotes(slot, voteP);
  renderSpeaker(slot);
}

function renderTimeline(trace, revealed, selected) {
  const el = $("timeline");
  if (!trace) {
    el.textContent =
      "Type a sentence and press Speak. The picker fills each CMUdict slot; speaker flies render crumbs.";
    return;
  }
  let flat = 0;
  el.innerHTML = trace.tokens
    .map((tok, ti) => {
      const phones = tok.slots
        .map((slot, si) => {
          const idx = flat++;
          const vis = idx < revealed;
          const ok = slot.pickerPhone === slot.referencePhone;
          const sel = selected && selected.token === ti && selected.slot === si;
          const cls = [
            "phone",
            vis ? "on" : "",
            vis ? (ok ? "ok" : "bad") : "",
            sel ? "sel" : "",
          ].join(" ");
          return `<button type="button" class="${cls}" data-ti="${ti}" data-si="${si}" ${
            vis ? "" : "disabled"
          }>${vis ? slot.pickerPhone : "·"}</button>`;
        })
        .join("");
      const pause =
        tok.pauseS > 0
          ? `<span class="pause-ms">${Math.round(tok.pauseS * 1000)} ms</span>`
          : "";
      return `<div class="t-row"><span class="t-word">${tok.word}</span><div class="phones">${phones}${pause}</div></div>`;
    })
    .join("");
  el.querySelectorAll(".phone.on").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ref = { token: Number(btn.dataset.ti), slot: Number(btn.dataset.si) };
      if (state.revealing) {
        state.paused = true;
        $("pause").textContent = "Resume";
      }
      showSlot(slotAt(trace, ref), "done");
      renderTimeline(trace, flatten(trace).length, ref);
    });
  });
}

function setControls(busy, hasTrace, revealing) {
  $("speak").disabled = busy;
  $("listen").disabled = !hasTrace;
  $("pause").disabled = !revealing;
  $("skip").disabled = !hasTrace || !revealing;
}

async function runReveal(trace) {
  const slots = flatten(trace);
  state.skip = false;
  state.paused = false;
  state.revealing = true;
  setControls(false, true, true);
  $("pause").textContent = "Pause";
  for (let i = 0; i < slots.length; i++) {
    if (state.skip) break;
    const ref = slots[i];
    const sl = slotAt(trace, ref);
    renderTimeline(trace, i, ref);
    for (const st of STAGES) {
      if (state.skip) break;
      setStage(st);
      showSlot(sl, st);
      await wait(STAGE_MS[st] / Math.max(0.25, state.speed));
    }
    renderTimeline(trace, i + 1, ref);
    if (!state.skip && state.audition && sl.crumbWavB64) {
      try {
        const a = await playB64(sl.crumbWavB64);
        const dur = (a.duration || 0.15) * 550;
        await wait(dur);
      } catch {
        /* autoplay */
      }
    }
  }
  state.revealing = false;
  setStage("done");
  renderTimeline(trace, slots.length, slots[0] || null);
  if (slots.length) showSlot(slotAt(trace, slots[slots.length - 1]), "done");
  setControls(false, true, false);
  $("pause").textContent = "Pause";
  $("playhead").style.left = "100%";
}

async function speak() {
  const text = $("utterance").value.trim();
  if (!text) return;
  if (state.audio) {
    state.audio.pause();
    state.audio = null;
  }
  setControls(true, false, false);
  setStatus("Flies are classifying…", "live");
  $("download").hidden = true;
  try {
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "speak failed");
    state.trace = data;
    const pct = data.nTotal ? Math.round((100 * data.nMatch) / data.nTotal) : 0;
    const match = $("match");
    match.textContent = data.nMatch + "/" + data.nTotal + " · " + pct + "%";
    match.className = "chip " + (pct < 80 ? "warn" : "ok");
    drawWave(data.wavB64);
    $("playhead").style.left = "0%";
    const blob = b64ToBlob(data.wavB64);
    const url = URL.createObjectURL(blob);
    const dl = $("download");
    dl.href = url;
    dl.download = "flysune.wav";
    dl.hidden = false;
    setStatus("");
    await runReveal(data);
  } catch (e) {
    setStatus(String(e.message || e), "err");
    setControls(false, !!state.trace, false);
  }
}

function b64ToBlob(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: "audio/wav" });
}

function listen() {
  if (!state.trace) return;
  if (state.audio) {
    state.audio.pause();
    state.audio = null;
    $("listen").textContent = "Listen";
    return;
  }
  $("listen").textContent = "Stop";
  const start = performance.now();
  const dur = (state.trace.durationS || 1) * 1000;
  const tick = () => {
    if (!state.audio) {
      $("playhead").style.left = "0%";
      return;
    }
    $("playhead").style.left = Math.min(100, ((performance.now() - start) / dur) * 100) + "%";
    requestAnimationFrame(tick);
  };
  playB64(state.trace.wavB64, () => {
    $("listen").textContent = "Listen";
    $("playhead").style.left = "100%";
  }).then(() => requestAnimationFrame(tick));
}

$("speak").addEventListener("click", () => void speak());
$("listen").addEventListener("click", listen);
$("pause").addEventListener("click", () => {
  if (!state.revealing) return;
  state.paused = !state.paused;
  $("pause").textContent = state.paused ? "Resume" : "Pause";
});
$("skip").addEventListener("click", () => {
  state.skip = true;
  state.paused = false;
});
$("audition").addEventListener("change", (e) => {
  state.audition = e.target.checked;
});
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("utterance").value = btn.dataset.preset === AVOCADO ? AVOCADO : btn.dataset.preset;
  });
});
document.querySelectorAll("#speeds button").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.speed = Number(btn.dataset.speed);
    document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("on", b === btn));
  });
});

drawKc(null, 0);
setStatus("");
