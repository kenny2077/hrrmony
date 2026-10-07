/* Villager Sound web app: pixel aurora + upload/convert flow. No dependencies. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------------- pixel aurora
     A dithered curtain of square cells over two dark ridges. It ignites once on load and
     afterwards only moves while the villager cover is playing (it follows the audio level). */
  const aurora = (() => {
    const cv = $("#aurora"), ctx = cv.getContext("2d");
    const CELL = 8;
    const CURTAIN = ["#7fe6b0", "#4fd89a", "#2fd9b0", "#39a8e0", "#8b7dff", "#c46bff"].map(hex);
    const SKY = [[0, "#020a09"], [.35, "#04160f"], [.6, "#061a2a"], [.85, "#05161a"], [1, "#030d0c"]];
    const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + .5) / 16);
    const STAR = hex("#8fb3aa");
    let cols = 0, rows = 0, level = 0, ignite = reduceMotion ? 1 : 0, phase = 0, raf = 0;

    function hex(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
    function lerp(a, b, t) { return a.map((v, i) => v + (b[i] - v) * t); }
    function skyAt(y) {
      for (let i = 1; i < SKY.length; i++) if (y <= SKY[i][0]) {
        const [y0, c0] = SKY[i - 1], [y1, c1] = SKY[i];
        return lerp(hex(c0), hex(c1), (y - y0) / (y1 - y0));
      }
      return hex(SKY[SKY.length - 1][1]);
    }
    function resize() {
      const r = cv.getBoundingClientRect();
      cols = Math.ceil(r.width / CELL); rows = Math.ceil(r.height / CELL);
      cv.width = cols; cv.height = rows;
      draw();
    }
    function ridge(x, base, amp, f, seed) {
      return base + amp * (Math.sin(x * f + seed) * .6 + Math.sin(x * f * 2.7 + seed * 3) * .25 +
                           Math.sin(x * f * 6.1 + seed * 7) * .15);
    }
    function draw() {
      if (!cols) return;
      const img = ctx.createImageData(cols, rows), d = img.data;
      const glow = (.8 + level * .9) * easeOut(ignite);
      for (let y = 0; y < rows; y++) {
        const ny = y / rows, sky = skyAt(ny);
        for (let x = 0; x < cols; x++) {
          const nx = x / cols, i = (y * cols + x) * 4;
          // curtain: folded bands that hang from the top third and fade towards the ridges
          const fold = Math.sin(nx * 7 + phase + Math.sin(nx * 3.1 - phase * .7) * 1.6);
          const band = Math.exp(-((ny - .28 - fold * .07) ** 2) / .018);
          const rays = .55 + .45 * Math.sin(nx * 64 + Math.sin(nx * 9 + phase) * 4);
          let a = band * rays * glow * (1 - ny * .9);
          const t = BAYER[(y & 3) * 4 + (x & 3)];
          const c = CURTAIN[Math.min(5, Math.floor((nx * .8 + fold * .12 + .1) * 6 + 6) % 6)];
          let px = a > t * .9 ? lerp(sky, c, Math.min(1, a)) : sky;
          // ridges
          if (ny > ridge(nx, .8, .05, 5.3, 1.7)) px = hex("#041311");
          if (ny > ridge(nx, .88, .04, 8.1, 4.2)) px = hex("#020908");
          if (ny < .32 && a < .05 && ((x * 73856093) ^ (y * 19349663)) % 997 === 0) px = STAR; // sparse stars
          d[i] = px[0]; d[i + 1] = px[1]; d[i + 2] = px[2]; d[i + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    }
    function easeOut(t) { return 1 - Math.pow(1 - Math.min(1, t), 3); }
    function igniteLoop(t0) {
      const step = (t) => {
        ignite = (t - t0) / 1400; draw();
        if (ignite < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    }
    let analyser = null, buf = null;
    function follow(audio) {
      if (reduceMotion) return;
      if (!analyser) {
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        const src = ac.createMediaElementSource(audio);
        analyser = ac.createAnalyser(); analyser.fftSize = 512;
        src.connect(analyser); analyser.connect(ac.destination);
        buf = new Uint8Array(analyser.fftSize);
        audio.addEventListener("play", () => ac.resume());
      }
      const tick = () => {
        if (audio.paused) { level = 0; draw(); return; }
        analyser.getByteTimeDomainData(buf);
        let s = 0; for (const v of buf) s += ((v - 128) / 128) ** 2;
        level = level * .7 + Math.min(1, Math.sqrt(s / buf.length) * 4) * .3;
        phase += .012 + level * .03; draw();
        requestAnimationFrame(tick);
      };
      audio.addEventListener("play", () => requestAnimationFrame(tick));
    }
    addEventListener("resize", () => { cancelAnimationFrame(raf); resize(); });
    resize();
    if (!reduceMotion) igniteLoop(performance.now());
    return { follow };
  })();

  /* ---------------------------------------------------------------- nav */
  const nav = $(".nav");
  const onScroll = () => nav.classList.toggle("solid", scrollY > 24);
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  /* ---------------------------------------------------------------- form */
  const form = $("#make"), fileIn = $("#file"), drop = $("#drop"), go = $("#go");
  let file = null;

  function pick(f) {
    if (!f) return;
    file = f;
    drop.classList.add("has-file");
    $("#drop-main").textContent = f.name;
    $("#drop-sub").textContent = `${(f.size / 1048576).toFixed(1)} MB. Choose another file to replace it`;
    go.disabled = false;
  }
  fileIn.addEventListener("change", () => pick(fileIn.files[0]));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("#device").textContent = h.device === "CPU"
      ? "Running on CPU. A 30-second cover takes a few minutes."
      : `Running on ${h.device}.`;
  }).catch(() => {});

  /* ---------------------------------------------------------------- job flow */
  const stage = $("#stage"), cells = $("#cells"), steps = [...document.querySelectorAll("#steps li")];
  const ORDER = ["analyze", "separate", "convert", "master", "done"];
  cells.innerHTML = "<span></span>".repeat(40);
  let timer = 0, lastPayload = null;

  form.addEventListener("submit", (e) => { e.preventDefault(); if (file) start(); });
  $("#retry").addEventListener("click", () => lastPayload && start());
  $("#again").addEventListener("click", () => { stage.hidden = true; form.scrollIntoView({ behavior: "smooth", block: "center" }); });

  async function start() {
    const mode = form.mode.value, shift = form.shift.value;
    lastPayload = true;
    const body = new FormData();
    body.append("file", file); body.append("mode", mode); body.append("shift", shift);
    reset(mode);
    stage.hidden = false;
    stage.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
    go.disabled = true;
    try {
      const r = await fetch("/api/jobs", { method: "POST", body });
      const job = await r.json();
      if (!r.ok) throw new Error(job.detail || `Upload failed (${r.status})`);
      poll(job.id);
    } catch (err) { fail(err.message); }
  }

  function reset(mode) {
    $("#stage-title").textContent = "Working on it";
    $("#stage-msg").textContent = "Uploading the song";
    steps[0].querySelector("span").textContent = mode === "full" ? "Read the song" : "Find the hook";
    steps.forEach((li) => li.classList.remove("active", "done"));
    cells.classList.remove("done");
    [...cells.children].forEach((c) => c.classList.remove("on"));
    $("#result").hidden = true; $("#error").hidden = true;
  }

  function poll(id) {
    clearTimeout(timer);
    fetch(`/api/jobs/${id}`).then((r) => r.json()).then((job) => {
      render(job);
      if (job.status === "done") done(job);
      else if (job.status === "error") fail(job.error);
      else timer = setTimeout(() => poll(id), 800);
    }).catch(() => { timer = setTimeout(() => poll(id), 2000); });
  }

  function render(job) {
    $("#stage-msg").textContent = job.message;
    const at = ORDER.indexOf(job.stage);
    steps.forEach((li, i) => {
      li.classList.toggle("done", at > i);
      li.classList.toggle("active", at === i);
    });
    const lit = Math.round(job.progress * cells.children.length);
    [...cells.children].forEach((c, i) => c.classList.toggle("on", i < lit));
  }

  function done(job) {
    go.disabled = false;
    cells.classList.add("done");
    const r = job.result || {};
    $("#stage-title").textContent = "Your villager cover";
    $("#stage-msg").textContent = job.mode === "full"
      ? `Whole song, made in ${r.seconds} s.`
      : `${fmt(r.start)} to ${fmt(r.start + r.duration)} of the song, made in ${r.seconds} s.`;
    const cover = $("#cover-audio"), orig = $("#orig-audio");
    cover.src = job.files.cover; orig.src = job.files.original || "";
    orig.closest(".player").hidden = !job.files.original;
    $("#dl-mp3").href = job.files.cover; $("#dl-wav").href = job.files.cover_wav;
    $("#result").hidden = false;
    aurora.follow(cover);
    [cover, orig].forEach((a) => a.addEventListener("play", () => [cover, orig].forEach((b) => b !== a && b.pause())));
  }

  function fail(msg) {
    go.disabled = !file;
    $("#stage-title").textContent = "Cover stopped";
    $("#stage-msg").textContent = "";
    $("#error-text").textContent = msg || "The server didn’t respond. Check that villager-sound serve is still running.";
    $("#error").hidden = false;
  }

  function fmt(s) { s = Math.max(0, Math.round(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

  /* highlight the nav tab of the section in view */
  const tabs = [...document.querySelectorAll(".tabs a")];
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) tabs.forEach((t) => t.classList.toggle("on", t.hash === `#${e.target.id}`));
  }), { threshold: .4 });
  ["make", "how"].forEach((id) => io.observe(document.getElementById(id)));
})();
