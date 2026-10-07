/* Villager Sound product page: lamp-field sky, dot-matrix glyphs, scroll-lit statement,
   A/B demo player. No dependencies. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
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

  /* ---------------------------------------------------------------- pixel Minecraft night
     A blocky village at night: banded sky, square moon, drifting flat clouds, hills, grass and
     dirt blocks, plank houses with lit windows, torches, an oak tree and a villager.
     The static world is painted once; only stars, clouds and light flicker move. */
  const BLOCK = 16, TP = 4; // one block = 4 x 4 texture pixels
  const P = {
    grass: [[86, 140, 56], [98, 156, 64], [72, 122, 46]],
    dirt: [[124, 88, 60], [138, 99, 68], [108, 75, 50], [94, 64, 43]],
    stone: [[118, 118, 118], [100, 100, 100], [132, 132, 132]],
    plank: [[166, 132, 80], [152, 120, 72], [180, 146, 90]],
    log: [[104, 80, 48], [88, 68, 40], [120, 92, 56]],
    roof: [[86, 58, 36], [72, 49, 30], [98, 67, 41]],
    cobble: [[122, 122, 122], [98, 98, 98], [142, 142, 142]],
    leaves: [[50, 98, 36], [60, 116, 44], [40, 82, 30]],
    door: [[92, 66, 40], [78, 55, 33]],
    robe: [[107, 74, 51], [127, 90, 63]], skin: [[189, 138, 109], [169, 118, 91]],
  };
  const hash = (x, y, s = 0) => { let h = (x * 374761393 + y * 668265263 + s * 1442695041) | 0; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };
  const pick = (pal, x, y, s) => pal[Math.floor(hash(x, y, s) * pal.length)];

  function makeWorld(cols, rows, opts = {}) {
    const base = opts.base ?? Math.floor(rows * .76);
    const heights = [];
    const vz = cols > 70 ? [Math.floor(cols * .53), Math.floor(cols * .72)] : [Math.floor(cols * .3), Math.floor(cols * .9)]; // flat village zone, clear of the headline and spec sheet
    for (let x = 0; x < cols; x++) {
      let h = Math.round(Math.sin(x * .19) * 1.4 + Math.sin(x * .07 + 2) * 1.6);
      if (x >= vz[0] - 2 && x <= vz[1] + 2) h = 0;
      heights.push(base + (opts.flat ? 0 : h));
    }
    return { cols, rows, base, heights, vz };
  }

  // paint a block (in block coords) with a texture; light = night dimming, warm = amber spill 0..1
  function paintBlock(g, bx, by, pal, light, warm, seed = 0, rowsTint) {
    for (let ty = 0; ty < 4; ty++) for (let tx = 0; tx < 4; tx++) {
      let c = pick(rowsTint && ty === 0 ? rowsTint : pal, bx * 4 + tx, by * 4 + ty, seed);
      const w = warm || 0;
      const r = c[0] * light + 255 * w * .55, gg = c[1] * light + 170 * w * .45, b = c[2] * light + 70 * w * .25;
      g.fillStyle = `rgb(${Math.min(255, r) | 0},${Math.min(255, gg) | 0},${Math.min(255, b) | 0})`;
      g.fillRect(bx * BLOCK + tx * TP, by * BLOCK + ty * TP, TP, TP);
    }
  }

  function drawWorld(g, world, W, H, withVillage = true, hills = true) {
    const { cols, rows, base, heights, vz } = world;
    const lights = [];
    const houses = [];
    if (withVillage) {
      const span = vz[1] - vz[0];
      if (span >= 14) houses.push({ x: vz[0], w: 6, h: 4 }, { x: vz[0] + 9, w: 5, h: 3 });
      else if (span > 8) houses.push({ x: vz[0], w: 6, h: 4 });
      houses.forEach((hs) => {
        lights.push({ x: (hs.x + Math.floor(hs.w / 2) + .5) * BLOCK, y: (base - 2.5) * BLOCK, r: 120, kind: "window" });
        lights.push({ x: (hs.x - .5) * BLOCK, y: (base - 1.6) * BLOCK, r: 70, kind: "torch" });
      });
    }
    const warmAt = (bx, by) => {
      let w = 0;
      for (const l of lights) {
        const d = Math.hypot((bx + .5) * BLOCK - l.x, (by + .5) * BLOCK - l.y);
        w = Math.max(w, Math.max(0, 1 - d / (l.r * 1.6)) ** 2 * (l.kind === "torch" ? .55 : .45));
      }
      return w;
    };
    // far and near hills (silhouettes)
    for (let x = 0; hills && x < cols; x++) {
      const far = base - 4 - Math.round(3 + 2.5 * Math.sin(x * .11 + 1) + 1.5 * Math.sin(x * .31));
      const near = base - 1 - Math.round(1.5 + 1.5 * Math.sin(x * .23 + 4));
      g.fillStyle = "#10203a"; g.fillRect(x * BLOCK, far * BLOCK, BLOCK, H);
      g.fillStyle = "#0c182c"; g.fillRect(x * BLOCK, near * BLOCK, BLOCK, H);
    }
    // terrain
    for (let x = 0; x < cols; x++) {
      const top = heights[x];
      for (let y = top; y < rows; y++) {
        const depth = y - top, w = warmAt(x, y), light = .52;
        if (depth === 0) paintBlock(g, x, y, P.dirt, light, w, 1, P.grass);
        else if (depth < 4) paintBlock(g, x, y, P.dirt, light * (1 - depth * .08), w * .6, 2);
        else paintBlock(g, x, y, P.stone, light * .7, 0, 3);
      }
      // grass tufts / flowers on top
      if (hash(x, 7) > .82 && (x < vz[0] - 1 || x > vz[1] + 1)) {
        g.fillStyle = hash(x, 9) > .6 ? "#c2493c" : "#4f8c34";
        g.fillRect(x * BLOCK + 6, (top - 1) * BLOCK + 8, 4, 8);
      }
    }
    if (!withVillage) return lights;
    // houses
    houses.forEach((hs, hi) => {
      const gy = base - 1; // block row just above the ground
      for (let i = 0; i < hs.w; i++) {
        for (let j = 0; j < hs.h; j++) {
          const bx = hs.x + i, by = gy - j;
          const corner = i === 0 || i === hs.w - 1;
          const pal = j === 0 && !corner ? P.cobble : corner ? P.log : P.plank;
          paintBlock(g, bx, by, pal, .58, warmAt(bx, by) * .8, 10 + hi);
        }
      }
      // roof: stepped
      const roofY = gy - hs.h;
      for (let k = 0; k <= Math.ceil(hs.w / 2); k++) {
        for (let i = k - 1; i <= hs.w - k; i++) {
          if (i < -1 || i > hs.w) continue;
          paintBlock(g, hs.x + i, roofY - k, P.roof, .5, 0, 20 + hi);
        }
      }
      // door (2 tall) and glowing window
      const dx = hs.x + 1;
      paintBlock(g, dx, gy, P.door, .5, .1, 30); paintBlock(g, dx, gy - 1, P.door, .5, .1, 31);
      const wx = hs.x + Math.floor(hs.w / 2) + (hs.w > 5 ? 1 : 0), wy = gy - 1;
      g.fillStyle = "#ffb547"; g.fillRect(wx * BLOCK + 2, wy * BLOCK + 2, BLOCK - 4, BLOCK - 4);
      g.fillStyle = "#c97a2a"; g.fillRect(wx * BLOCK + 7, wy * BLOCK + 2, 2, BLOCK - 4); g.fillRect(wx * BLOCK + 2, wy * BLOCK + 7, BLOCK - 4, 2);
      lights[hi * 2] = { x: (wx + .5) * BLOCK, y: (wy + .5) * BLOCK, r: 120, kind: "window" };
      // torch left of the house
      const tx = hs.x - 1, ty = gy;
      g.fillStyle = "#6e5030"; g.fillRect(tx * BLOCK + 7, ty * BLOCK + 6, 3, 10);
      g.fillStyle = "#ffd27a"; g.fillRect(tx * BLOCK + 6, ty * BLOCK + 2, 5, 5);
      lights[hi * 2 + 1] = { x: tx * BLOCK + 8.5, y: ty * BLOCK + 4, r: 70, kind: "torch" };
    });
    // oak tree at the right edge of the village
    const tX = vz[1], tY = base - 1;
    for (let j = 0; j < 4; j++) paintBlock(g, tX, tY - j, P.log, .5, 0, 40);
    for (let j = 0; j < 3; j++) for (let i = -2; i <= 2; i++) {
      if (j === 2 && Math.abs(i) === 2) continue;
      paintBlock(g, tX + i, tY - 3 - j, P.leaves, .5, 0, 41);
    }
    // the villager, between the houses
    const vx = houses.length > 1 ? houses[0].x + houses[0].w + 1 : vz[0] + 7, vy = base - 1;
    paintBlock(g, vx, vy, P.robe, .7, warmAt(vx, vy), 50);
    const hx = vx * BLOCK, hy = (vy - 1) * BLOCK;
    paintBlock(g, vx, vy - 1, P.skin, .78, warmAt(vx, vy - 1), 51);
    g.fillStyle = "#5e3d27"; g.fillRect(hx, hy, BLOCK, 4);              // hair
    g.fillStyle = "#3e271a"; g.fillRect(hx + 2, hy + 6, 12, 2);         // brow
    g.fillStyle = "#e9e6dc"; g.fillRect(hx + 2, hy + 8, 3, 2); g.fillRect(hx + 11, hy + 8, 3, 2);
    g.fillStyle = "#2f9e4f"; g.fillRect(hx + 5, hy + 8, 2, 2); g.fillRect(hx + 9, hy + 8, 2, 2);
    g.fillStyle = "#9a654c"; g.fillRect(hx + 6, hy + 8, 4, 10);          // the nose
    return lights;
  }

  function nightSky(g, W, H, horizonY) {
    const top = [8, 16, 38], bot = [34, 54, 98];
    const bands = Math.ceil(horizonY / BLOCK);
    for (let i = 0; i < bands; i++) {
      const t = i / Math.max(1, bands - 1), c = top.map((v, k) => v + (bot[k] - v) * t ** 1.6);
      g.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; g.fillRect(0, i * BLOCK, W, BLOCK);
    }
    g.fillStyle = "#0c182c"; g.fillRect(0, horizonY, W, H);
  }

  function moon(g, x, y, s) {
    g.fillStyle = "rgba(220, 230, 255, .08)"; g.fillRect(x - s * .6, y - s * .6, s * 2.2, s * 2.2);
    g.fillStyle = "rgba(220, 230, 255, .12)"; g.fillRect(x - s * .3, y - s * .3, s * 1.6, s * 1.6);
    g.fillStyle = "#e6ebe2"; g.fillRect(x, y, s, s);
    g.fillStyle = "#c4cbc2";
    [[.2, .2, .25], [.6, .55, .2], [.25, .65, .15], [.65, .15, .12]].forEach(([a, b, c]) => g.fillRect(x + a * s, y + b * s, c * s, c * s));
  }

  const sky = $("#sky");
  if (sky) {
    const ctx = sky.getContext("2d");
    const stat = document.createElement("canvas"), sctx = stat.getContext("2d");
    let W = 0, H = 0, world, lights = [], stars = [], clouds = [], running = false, visible = true, last = 0;
    const t0 = performance.now();

    function build() {
      const r = sky.getBoundingClientRect();
      W = Math.ceil(r.width); H = Math.ceil(r.height);
      for (const c of [sky, stat]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); }
      const cols = Math.ceil(W / BLOCK), rows = Math.ceil(H / BLOCK);
      world = makeWorld(cols, rows);
      sctx.setTransform(DPR, 0, 0, DPR, 0, 0); sctx.imageSmoothingEnabled = false;
      const horizon = (world.base - 9) * BLOCK;
      nightSky(sctx, W, H, (world.base - 2) * BLOCK);
      if (W > 700) moon(sctx, Math.round(W * .8 / BLOCK) * BLOCK, Math.round(H * .1 / BLOCK) * BLOCK + BLOCK, BLOCK * 3);
      else moon(sctx, W - BLOCK * 3.5, 76, BLOCK * 2);
      lights = drawWorld(sctx, world, W, H, true);
      stars = Array.from({ length: Math.round(W * H / 9000) }, (_, i) => ({
        x: Math.floor(hash(i, 1) * W / TP) * TP, y: Math.floor(hash(i, 2) * horizon * .9 / TP) * TP,
        p: hash(i, 3) * 6.28, s: hash(i, 4) > .85 ? TP : TP / 2,
      }));
      clouds = Array.from({ length: Math.max(3, Math.round(W / 260)) }, (_, i) => ({
        x: hash(i, 5) * W, y: Math.round((BLOCK * 2 + hash(i, 6) * H * .22) / BLOCK) * BLOCK,
        parts: Array.from({ length: 3 + Math.floor(hash(i, 7) * 3) }, (_, k) => [k * 2 * BLOCK - (hash(i, k) > .5 ? BLOCK : 0), hash(i, k + 9) > .5 ? 0 : BLOCK, (2 + Math.floor(hash(i, k + 3) * 3)) * BLOCK]),
        v: 4 + hash(i, 8) * 5,
      }));
      draw(performance.now());
    }
    function draw(now) {
      const t = (now - t0) / 1000;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(stat, 0, 0);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      for (const s of stars) {
        const a = reduce ? .7 : .45 + .45 * Math.sin(t * 1.3 + s.p);
        ctx.fillStyle = `rgba(232, 240, 255, ${a})`; ctx.fillRect(s.x, s.y, s.s, s.s);
      }
      ctx.fillStyle = "rgba(214, 226, 245, .2)";
      for (const c of clouds) {
        const span = W + 8 * BLOCK;
        const x = ((c.x + (reduce ? 0 : t * c.v)) % span) - 4 * BLOCK;
        for (const [dx, dy, w] of c.parts) ctx.fillRect(Math.round((x + dx) / TP) * TP, c.y + dy, w, BLOCK);
      }
      ctx.globalCompositeOperation = "lighter";
      lights.forEach((l, i) => {
        const f = reduce ? 1 : (l.kind === "torch" ? .85 + .15 * Math.sin(t * 9 + i * 2) * Math.sin(t * 5.3 + i) : .94 + .06 * Math.sin(t * 2.1 + i));
        const gr = ctx.createRadialGradient(l.x, l.y, 0, l.x, l.y, l.r * f);
        gr.addColorStop(0, `rgba(255, 170, 70, ${.32 * f})`); gr.addColorStop(1, "rgba(255, 170, 70, 0)");
        ctx.fillStyle = gr; ctx.fillRect(l.x - l.r, l.y - l.r, l.r * 2, l.r * 2);
      });
      ctx.globalCompositeOperation = "source-over";
    }
    function loop(now) {
      if (!running) return;
      if (now - last > 40) { draw(now); last = now; } // ~25 fps is plenty for drifting clouds
      requestAnimationFrame(loop);
    }
    const setRunning = () => {
      const want = visible && !reduce;
      if (want && !running) { running = true; requestAnimationFrame(loop); }
      if (!want) running = false;
    };
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; setRunning(); }).observe(sky);
    let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(build, 120); });
    build(); setRunning();
  }

  /* deepslate block texture for the feature panels (generated, tiled by CSS) */
  {
    const tile = document.createElement("canvas"); tile.width = tile.height = 64;
    const tg = tile.getContext("2d");
    const slate = [[38, 42, 52], [32, 35, 44], [44, 48, 60], [28, 31, 39]];
    for (let by = 0; by < 4; by++) for (let bx = 0; bx < 4; bx++) paintBlock(tg, bx, by, slate, .62, 0, 60);
    tg.fillStyle = "rgba(2, 6, 14, .55)";
    for (let i = 0; i < 4; i++) { tg.fillRect(i * BLOCK, 0, 1, 64); tg.fillRect(0, i * BLOCK, 64, 1); }
    document.documentElement.style.setProperty("--slate", `url(${tile.toDataURL()})`);
  }

  /* a strip of grass and dirt along the bottom of the closing section */
  const ground = $("#ground");
  if (ground) {
    const paint = () => {
      const r = ground.getBoundingClientRect();
      ground.width = Math.round(r.width * DPR); ground.height = Math.round(r.height * DPR);
      const g = ground.getContext("2d"); g.setTransform(DPR, 0, 0, DPR, 0, 0);
      const cols = Math.ceil(r.width / BLOCK), rows = Math.ceil(r.height / BLOCK);
      const w = makeWorld(cols, rows, { base: rows - 3, flat: true });
      w.vz = [-10, -10];
      drawWorld(g, w, r.width, r.height, false, false);
      // a villager wandering on the grass
      const vx = Math.floor(cols * .78), vy = rows - 4;
      paintBlock(g, vx, vy, P.robe, .7, 0, 50); paintBlock(g, vx, vy - 1, P.skin, .78, 0, 51);
      const hx = vx * BLOCK, hy = (vy - 1) * BLOCK;
      g.fillStyle = "#5e3d27"; g.fillRect(hx, hy, BLOCK, 4);
      g.fillStyle = "#3e271a"; g.fillRect(hx + 2, hy + 6, 12, 2);
      g.fillStyle = "#e9e6dc"; g.fillRect(hx + 2, hy + 8, 3, 2); g.fillRect(hx + 11, hy + 8, 3, 2);
      g.fillStyle = "#2f9e4f"; g.fillRect(hx + 5, hy + 8, 2, 2); g.fillRect(hx + 9, hy + 8, 2, 2);
      g.fillStyle = "#9a654c"; g.fillRect(hx + 6, hy + 8, 4, 10);
    };
    paint(); addEventListener("resize", paint);
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
    V: "10001 10001 10001 10001 10001 01010 00100", " ": "000 000 000 000 000 000 000",
    // lowercase s: four columns wide and only five rows tall, so "30s" can't be read as "305"
    s: "0000 0000 0111 1000 0110 0001 1110",
  };
  function drawGlyph(el) {
    const text = [...(el.dataset.glyph || "")].map((c) => (FONT[c] ? c : c.toUpperCase())).join("");
    const span = $(".glyph-text", el);
    const color = getComputedStyle(span).color;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return;
    let cv = $("canvas.glyph-cv", el);
    if (!cv) { cv = document.createElement("canvas"); cv.className = "glyph-cv"; cv.setAttribute("aria-hidden", "true"); el.prepend(cv); }
    const rowsOf = (ch) => (FONT[ch] || FONT[" "]).split(" ");
    const widths = [...text].map((ch) => rowsOf(ch)[0].length);
    const gw = widths.reduce((a, b) => a + b, 0) + widths.length - 1, gh = 7;
    const cell = Math.max(2, Math.floor(Math.min(r.width / gw, r.height / gh)));
    cv.width = Math.round(r.width * DPR); cv.height = Math.round(r.height * DPR);
    const ctx = cv.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    const ox = Math.floor((r.width - gw * cell) / 2), oy = Math.floor((r.height - gh * cell) / 2);
    const seam = Math.max(1, Math.round(cell * .22));
    let col = 0;
    [...text].forEach((ch, k) => {
      rowsOf(ch).forEach((row, y) => [...row].forEach((bit, x) => {
        ctx.fillStyle = bit === "1" ? color : "rgba(150, 205, 245, .06)";
        ctx.fillRect(ox + (col + x) * cell, oy + y * cell, cell - seam, cell - seam);
      }));
      col += widths[k] + 1;
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

  /* ---------------------------------------------------------------- cursor: block heat + flashlight
     As on auroraforgelab.com: on fine pointers the cursor warms the pixel blocks it passes over
     (five quantised amber steps, so the trail stays pixelated) and lights a soft flashlight on
     cards. Off on touch screens and under reduced motion. */
  const FINE = matchMedia("(hover: hover) and (pointer: fine)").matches;
  const HEAT = [[255, 122, 42], [255, 196, 107]]; // ember to glow, like a torch
  const layers = [];
  function heatLayer(host, { cell = 16, radius = 3, decay = .88, seam = 3 } = {}) {
    const cv = document.createElement("canvas");
    cv.className = "heat"; cv.setAttribute("aria-hidden", "true");
    host.appendChild(cv);
    const L = { host, cv, ctx: cv.getContext("2d"), cell, radius, decay, seam, cols: 0, rows: 0, heat: new Float32Array(0), hot: false };
    L.resize = () => {
      const r = host.getBoundingClientRect();
      cv.width = Math.round(r.width * DPR); cv.height = Math.round(r.height * DPR);
      L.cols = Math.ceil(r.width / cell); L.rows = Math.ceil(r.height / cell);
      L.heat = new Float32Array(L.cols * L.rows);
    };
    L.stamp = (e) => {
      const r = host.getBoundingClientRect();
      const cx = (e.clientX - r.left) / cell, cy = (e.clientY - r.top) / cell;
      if (cx < -radius || cy < -radius || cx > L.cols + radius || cy > L.rows + radius) return;
      for (let y = Math.max(0, Math.floor(cy - radius)); y <= Math.min(L.rows - 1, Math.ceil(cy + radius)); y++) {
        for (let x = Math.max(0, Math.floor(cx - radius)); x <= Math.min(L.cols - 1, Math.ceil(cx + radius)); x++) {
          const d = Math.hypot(x + .5 - cx, y + .5 - cy) / radius;
          if (d >= 1) continue;
          const i = y * L.cols + x;
          L.heat[i] = Math.max(L.heat[i], 1 - d * d * .9);
        }
      }
      L.hot = true; wake();
    };
    L.frame = () => {
      const { ctx, heat } = L;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.clearRect(0, 0, cv.width, cv.height);
      let hot = false;
      for (let i = 0; i < heat.length; i++) {
        const h = heat[i];
        if (h <= 0) continue;
        heat[i] = h < .08 ? 0 : h * decay; hot = true;
        if (h <= .38) continue;
        const q = Math.ceil(h * 5) / 5, k = .55 + q * .4;
        const c = HEAT[0].map((v, j) => v + (HEAT[1][j] - v) * q);
        ctx.fillStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${k * .78})`;
        const x = i % L.cols, y = (i / L.cols) | 0;
        ctx.fillRect(x * cell, y * cell, cell - seam, cell - seam);
      }
      L.hot = hot;
      return hot;
    };
    L.resize(); layers.push(L);
    return L;
  }
  let heatRaf = 0;
  function wake() {
    if (heatRaf) return;
    const step = () => {
      let busy = false;
      for (const L of layers) if (L.hot) busy = L.frame() || busy;
      heatRaf = busy ? requestAnimationFrame(step) : 0;
    };
    heatRaf = requestAnimationFrame(step);
  }
  if (FINE && !reduce) {
    const hero = $(".hero");
    if (hero) heatLayer(hero, { cell: 16, radius: 3, decay: .88, seam: 2 });
    $$(".card-media").forEach((m) => heatLayer(m, { cell: 16, radius: 2.4, decay: .9, seam: 1 }));
    const player = $(".player"); if (player) heatLayer(player, { cell: 12, radius: 2.2, decay: .9 });
    const cta = $(".cta"); if (cta) heatLayer(cta, { cell: 16, radius: 2.6, decay: .9, seam: 2 });
    const mark = $(".glyph-wordmark"); if (mark) heatLayer(mark, { cell: 12, radius: 2.2, decay: .9 });
    document.addEventListener("pointermove", (e) => {
      for (const L of layers) L.stamp(e);
      const surface = e.target.closest?.(".glow");
      if (!surface) return;
      const r = surface.getBoundingClientRect();
      surface.style.setProperty("--mx", `${e.clientX - r.left}px`);
      surface.style.setProperty("--my", `${e.clientY - r.top}px`);
    }, { passive: true });
    let rh; addEventListener("resize", () => { clearTimeout(rh); rh = setTimeout(() => layers.forEach((L) => L.resize()), 150); });
  }

  /* ---------------------------------------------------------------- copy buttons */
  $$(".code .copy").forEach((btn) => btn.addEventListener("click", async () => {
    const text = $("code", btn.parentElement).textContent.split("\n").map((l) => l.replace(/\s+#.*$/, "")).join("\n");
    try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
    catch { btn.textContent = "Select and copy"; }
    setTimeout(() => { btn.textContent = "Copy"; }, 1600);
  }));
})();
