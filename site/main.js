/* Villager Sound product page: lamp-field sky, dot-matrix glyphs, scroll-lit statement,
   A/B demo player. No dependencies. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = matchMedia("(pointer: fine)").matches;
  const DPR = Math.min(2, devicePixelRatio || 1);

  /* ---------------------------------------------------------------- header */
  const header = $(".site-header");
  const onScroll = () => header.classList.toggle("is-scrolled", scrollY > 24);
  addEventListener("scroll", onScroll, { passive: true }); onScroll();
  const navLinks = $$(".nav-links a");
  const navIO = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) navLinks.forEach((a) => a.classList.toggle("is-current", a.hash === `#${e.target.id}`));
  }), { rootMargin: "-45% 0px -50% 0px" });
  ["listen", "features", "install"].forEach((id) => { const el = document.getElementById(id); if (el) navIO.observe(el); });

  /* ---------------------------------------------------------------- lamp field sky
     Square lamps with dark seams. Aurora bands drift slowly; the cursor warms lamps to amber. */
  const sky = $("#sky");
  if (sky) {
    const ctx = sky.getContext("2d");
    const CELL = 12, SEAM = 3;
    const C = {
      teal: [79, 227, 208], cyan: [76, 201, 240], blue: [91, 140, 255], violet: [155, 125, 255],
      amber: [255, 181, 71], base: [8, 18, 38],
    };
    let cols = 0, rows = 0, mx = -1e4, my = -1e4, heat = null, t0 = performance.now(), running = false, visible = true;
    const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

    function resize() {
      const r = sky.getBoundingClientRect();
      sky.width = Math.round(r.width * DPR); sky.height = Math.round(r.height * DPR);
      cols = Math.ceil(r.width / CELL); rows = Math.ceil(r.height / CELL);
      heat = new Float32Array(cols * rows);
      draw(performance.now());
    }
    function band(nx, ny, t, cy, amp, freq, speed, width) {
      const y = cy + amp * Math.sin(nx * freq + t * speed) + amp * .5 * Math.sin(nx * freq * 2.3 - t * speed * 1.4);
      return Math.exp(-((ny - y) ** 2) / width);
    }
    function draw(now) {
      const t = (now - t0) / 1000 + scrollY * .002;
      const ignite = reduce ? 1 : Math.min(1, (now - t0) / 1400);
      const ig = 1 - (1 - ignite) ** 3;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.fillStyle = "#02060e"; ctx.fillRect(0, 0, sky.width, sky.height);
      const cx = mx / CELL, cy = my / CELL;
      for (let y = 0; y < rows; y++) {
        const ny = y / rows;
        const fade = Math.max(0, 1 - ny * 1.5) ** 1.3;
        for (let x = 0; x < cols; x++) {
          const nx = x / cols, i = y * cols + x;
          const a = band(nx, ny, t * .08, .16, .06, 5.5, 1, .01);
          const b = band(nx, ny, t * .06, .3, .05, 3.8, -1.2, .016);
          const v = band(nx, ny, t * .05, .12, .04, 7.0, .7, .008);
          let col = mix(C.base, C.teal, Math.min(1, a * 1.1));
          col = mix(col, C.cyan, b * .55);
          col = mix(col, C.violet, v * .6 * nx);
          let lit = (a * .9 + b * .6 + v * .5) * fade * ig;
          // flicker: deterministic per lamp, slow
          lit *= .82 + .18 * Math.sin(i * 12.9898 + t * .9);
          if (fine && !reduce) {
            const d = Math.hypot(x - cx, y - cy);
            const target = d < 7 ? (1 - d / 7) : 0;
            heat[i] = Math.max(target, heat[i] * .94);
          }
          const h = heat[i] || 0;
          if (h > .01) { col = mix(col, C.amber, h); lit = Math.max(lit, h * .9); }
          const alpha = .07 + Math.min(1, lit) * .8;
          ctx.fillStyle = `rgba(${col[0] | 0},${col[1] | 0},${col[2] | 0},${alpha})`;
          ctx.fillRect(x * CELL, y * CELL, CELL - SEAM, CELL - SEAM);
        }
      }
    }
    function loop(now) {
      if (!running) return;
      draw(now);
      requestAnimationFrame(loop);
    }
    function setRunning(on) {
      const want = on && visible && !reduce;
      if (want && !running) { running = true; requestAnimationFrame(loop); }
      if (!want) running = false;
    }
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; setRunning(true); }).observe(sky);
    addEventListener("resize", resize);
    if (fine) addEventListener("pointermove", (e) => {
      const r = sky.getBoundingClientRect(); mx = e.clientX - r.left; my = e.clientY - r.top;
    }, { passive: true });
    resize();
    if (reduce) draw(performance.now()); else setRunning(true);
  }

  /* ---------------------------------------------------------------- dot-matrix glyphs (5x7) */
  const FONT = {
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    A: "01110 10001 10001 11111 10001 10001 10001", D: "11110 10001 10001 10001 10001 10001 11110",
    E: "11111 10000 10000 11110 10000 10000 11111", F: "11111 10000 10000 11110 10000 10000 10000",
    G: "01110 10001 10000 10111 10001 10001 01111", I: "01110 00100 00100 00100 00100 00100 01110",
    L: "10000 10000 10000 10000 10000 10000 11111", N: "10001 11001 10101 10011 10001 10001 10001",
    O: "01110 10001 10001 10001 10001 10001 01110", R: "11110 10001 10001 11110 10100 10010 10001",
    S: "01111 10000 10000 01110 00001 00001 11110", U: "10001 10001 10001 10001 10001 10001 01110",
    V: "10001 10001 10001 10001 10001 01010 00100", " ": "00000 00000 00000 00000 00000 00000 00000",
  };
  function drawGlyph(el) {
    const text = (el.dataset.glyph || "").toUpperCase();
    const span = $(".glyph-text", el);
    const color = getComputedStyle(span).color;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return;
    let cv = $("canvas", el);
    if (!cv) { cv = document.createElement("canvas"); cv.setAttribute("aria-hidden", "true"); el.appendChild(cv); }
    const gw = text.length * 6 - 1, gh = 7;
    const cell = Math.max(2, Math.floor(Math.min(r.width / gw, r.height / gh)));
    cv.width = Math.round(r.width * DPR); cv.height = Math.round(r.height * DPR);
    const ctx = cv.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    const ox = Math.floor((r.width - gw * cell) / 2), oy = Math.floor((r.height - gh * cell) / 2);
    const seam = Math.max(1, Math.round(cell * .22));
    [...text].forEach((ch, k) => {
      const rows = (FONT[ch] || FONT[" "]).split(" ");
      rows.forEach((row, y) => [...row].forEach((bit, x) => {
        ctx.fillStyle = bit === "1" ? color : "rgba(150, 205, 245, .06)";
        ctx.fillRect(ox + (k * 6 + x) * cell, oy + y * cell, cell - seam, cell - seam);
      }));
    });
    el.classList.add("is-drawn");
  }
  const glyphs = $$(".glyph");
  const drawAll = () => glyphs.forEach(drawGlyph);
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(drawAll);
  addEventListener("resize", drawAll);

  /* ---------------------------------------------------------------- statement: words light with scroll */
  const statement = $(".statement");
  if (statement) {
    const p = $("p", statement);
    const words = [];
    const wrap = (node, hot) => {
      [...node.childNodes].forEach((n) => {
        if (n.nodeType === 3) {
          const frag = document.createDocumentFragment();
          n.textContent.split(/(\s+)/).forEach((part) => {
            if (!part.trim()) { frag.appendChild(document.createTextNode(part)); return; }
            const s = document.createElement("span"); s.className = "w" + (hot ? " hot" : ""); s.textContent = part;
            words.push(s); frag.appendChild(s);
          });
          n.replaceWith(frag);
        } else if (n.nodeType === 1) wrap(n, hot || n.tagName === "EM");
      });
    };
    wrap(p, false);
    const update = () => {
      const r = statement.getBoundingClientRect();
      const span = Math.max(1, r.height - innerHeight);
      const prog = reduce ? 1 : Math.min(1, Math.max(0, (-r.top + innerHeight * .35) / span));
      const n = Math.round(prog * words.length);
      words.forEach((w, i) => w.classList.toggle("lit", i < n));
    };
    addEventListener("scroll", update, { passive: true }); update();
  }

  /* ---------------------------------------------------------------- A/B demo player */
  const player = $("#player");
  if (player) {
    const A = { villager: $("#a-villager"), original: $("#a-original") };
    const playBtn = $("#play"), seek = $("#seek"), fill = $("#seek-fill"), time = $("#time"), viz = $("#viz");
    let current = "villager", actx = null, analyser = null, gains = {}, raf = 0;
    const vctx = viz.getContext("2d");
    const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

    function setupAudio() {
      if (actx) return;
      actx = new (window.AudioContext || window.webkitAudioContext)();
      analyser = actx.createAnalyser(); analyser.fftSize = 1024; analyser.smoothingTimeConstant = .78;
      analyser.connect(actx.destination);
      for (const [k, el] of Object.entries(A)) {
        const src = actx.createMediaElementSource(el), g = actx.createGain();
        g.gain.value = k === current ? 1 : 0;
        src.connect(g); g.connect(analyser); gains[k] = g;
      }
    }
    function setVoice(v) {
      current = v;
      if (gains.villager) for (const k in gains) gains[k].gain.setTargetAtTime(k === v ? 1 : 0, actx.currentTime, .03);
      else for (const k in A) A[k].muted = k !== v;
      player.dataset.voice = v;
    }
    async function play() {
      setupAudio();
      await actx.resume();
      A.original.currentTime = A.villager.currentTime;
      // reflect the state immediately; playback may still be buffering
      player.classList.add("is-playing"); playBtn.setAttribute("aria-label", "Pause");
      tick();
      const results = await Promise.allSettled(Object.values(A).map((a) => a.play()));
      if (results.some((r) => r.status === "rejected")) pause();
    }
    function pause() {
      Object.values(A).forEach((a) => a.pause());
      player.classList.remove("is-playing"); playBtn.setAttribute("aria-label", "Play");
    }
    playBtn.addEventListener("click", () => (A.villager.paused ? play() : pause()));
    $$('input[name="voice"]', player).forEach((r) => r.addEventListener("change", () => setVoice(r.value)));
    A.villager.addEventListener("ended", () => { pause(); A.villager.currentTime = A.original.currentTime = 0; progress(); });
    function seekTo(frac) {
      const d = A.villager.duration || 0;
      Object.values(A).forEach((a) => { a.currentTime = frac * d; });
      progress();
    }
    seek.addEventListener("click", (e) => { const r = seek.getBoundingClientRect(); seekTo((e.clientX - r.left) / r.width); });
    seek.addEventListener("keydown", (e) => {
      const d = A.villager.duration || 1, step = 5 / d, cur = A.villager.currentTime / d;
      if (e.key === "ArrowRight") { seekTo(Math.min(1, cur + step)); e.preventDefault(); }
      if (e.key === "ArrowLeft") { seekTo(Math.max(0, cur - step)); e.preventDefault(); }
    });
    function progress() {
      const d = A.villager.duration || 0, t = A.villager.currentTime;
      const f = d ? t / d : 0;
      fill.style.width = `${f * 100}%`; seek.setAttribute("aria-valuenow", Math.round(f * 100));
      time.textContent = fmt(t);
      // keep the two takes locked together
      if (!A.villager.paused && Math.abs(A.original.currentTime - t) > .08) A.original.currentTime = t;
    }
    // lamp-column spectrum (idle: a quiet resting row)
    const bins = new Uint8Array(512);
    function drawViz() {
      const r = viz.getBoundingClientRect();
      if (viz.width !== Math.round(r.width * DPR)) { viz.width = Math.round(r.width * DPR); viz.height = Math.round(r.height * DPR); }
      vctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      vctx.clearRect(0, 0, r.width, r.height);
      const CELL = 12, SEAM = 3, cols = Math.floor(r.width / CELL), rows = Math.floor(r.height / CELL);
      if (analyser) analyser.getByteFrequencyData(bins);
      const hot = current === "villager";
      for (let x = 0; x < cols; x++) {
        const b = Math.floor(Math.pow(x / cols, 1.7) * 220) + 2;
        const v = analyser ? bins[b] / 255 : 0;
        const h = Math.max(1, Math.round(v * rows));
        for (let y = 0; y < rows; y++) {
          const on = rows - y <= h;
          const top = rows - y === h && v > .05;
          vctx.fillStyle = !on ? "rgba(150, 205, 245, .05)"
            : top ? (hot ? "#ffb547" : "#eaf2f8")
            : (hot ? `rgba(79, 227, 208, ${.35 + .65 * (rows - y) / rows})` : `rgba(147, 167, 187, ${.3 + .5 * (rows - y) / rows})`);
          vctx.fillRect(x * CELL, y * CELL, CELL - SEAM, CELL - SEAM);
        }
      }
    }
    function tick() {
      cancelAnimationFrame(raf);
      const step = () => { progress(); drawViz(); if (!A.villager.paused) raf = requestAnimationFrame(step); else drawViz(); };
      raf = requestAnimationFrame(step);
    }
    A.villager.addEventListener("loadedmetadata", progress);
    setVoice("villager");
    drawViz(); addEventListener("resize", drawViz);
  }

  /* ---------------------------------------------------------------- copy buttons */
  $$(".code .copy").forEach((btn) => btn.addEventListener("click", async () => {
    const text = $("code", btn.parentElement).textContent.split("\n").map((l) => l.replace(/\s+#.*$/, "")).join("\n");
    try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
    catch { btn.textContent = "Select and copy"; }
    setTimeout(() => { btn.textContent = "Copy"; }, 1600);
  }));
})();
