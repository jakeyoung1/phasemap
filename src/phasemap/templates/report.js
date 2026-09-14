(() => {
  "use strict";
  const raw = JSON.parse(document.getElementById("phase-data").textContent);
  const bundles = Array.isArray(raw) ? raw : [raw];
  const TEAM = { home: "Home", away: "Away" };
  const STATUS = { verified: "backed", partial: "partly backed", unsupported: "not backed" };
  const SPEEDS = [1, 2, 0.5];
  const SCALE = 20; // canvas pixels per metre
  const MARGIN = 4; // metres of run-off drawn around the pitch

  const $ = (id) => document.getElementById(id);
  const other = (team) => (team === "home" ? "away" : "home");
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  const swatch = (team) => el("i", `swatch ${team}`);

  /* The phase on screen */
  let data, meta, players, track, ballTrack, fps, lastFrame, duration, facts, checks, clockBase;

  function use(bundle) {
    data = bundle;
    meta = bundle.meta;
    players = bundle.players;
    track = bundle.track.xy;
    ballTrack = bundle.track.ball;
    fps = meta.fps;
    lastFrame = track.length - 1;
    duration = lastFrame / fps;
    facts = new Map(bundle.facts.map((f) => [f.id, f]));
    checks = new Map(((bundle.verification || {}).checks || []).map((c) => [c.path, c]));
    const [m, s] = meta.clock_start.split(":").map(Number);
    clockBase = m * 60 + s;
  }

  function matchClock(t) {
    const total = Math.max(0, Math.round(clockBase + t - meta.won_t));
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
  }

  /* Replay board */
  const canvas = $("pitch");
  const ctx = canvas.getContext("2d");
  const layer = document.createElement("canvas");
  layer.width = canvas.width;
  layer.height = canvas.height;
  let palette = {};
  let frame = 0;
  let playing = false;
  let speedIndex = 0;
  let lastTime = null;
  let carry = 0;
  let focus = new Set();

  const X = (x) => (x + MARGIN) * SCALE;
  const Y = (y) => (68 - y + MARGIN) * SCALE;

  function readPalette() {
    const style = getComputedStyle(document.documentElement);
    const v = (name) => style.getPropertyValue(name).trim();
    palette = {
      turf: v("--turf"), stripe: v("--turf-stripe"), chalk: v("--chalk"),
      home: v("--home"), away: v("--away"), marker: v("--marker"),
    };
  }

  function paintPitch() {
    const g = layer.getContext("2d");
    g.fillStyle = palette.turf;
    g.fillRect(0, 0, layer.width, layer.height);
    g.fillStyle = palette.stripe;
    const band = 105 / 14;
    for (let k = 0; k < 14; k += 2) g.fillRect(X(k * band), Y(68), band * SCALE, 68 * SCALE);
    g.strokeStyle = palette.chalk;
    g.lineWidth = 0.14 * SCALE;
    const box = (x, y, w, h) => g.strokeRect(X(x), Y(y + h), w * SCALE, h * SCALE);
    const dot = (x, y) => {
      g.beginPath();
      g.arc(X(x), Y(y), 0.3 * SCALE, 0, Math.PI * 2);
      g.fillStyle = palette.chalk;
      g.fill();
    };
    box(0, 0, 105, 68);
    g.beginPath();
    g.moveTo(X(52.5), Y(0));
    g.lineTo(X(52.5), Y(68));
    g.stroke();
    g.beginPath();
    g.arc(X(52.5), Y(34), 9.15 * SCALE, 0, Math.PI * 2);
    g.stroke();
    dot(52.5, 34);
    const spread = Math.acos(5.5 / 9.15); // where the penalty arc meets the box edge
    for (const right of [false, true]) {
      const goalX = right ? 105 : 0;
      const inward = right ? -1 : 1;
      box(right ? 105 - 16.5 : 0, 34 - 20.16, 16.5, 40.32);
      box(right ? 105 - 5.5 : 0, 34 - 9.16, 5.5, 18.32);
      box(right ? 105 : -1.8, 34 - 3.66, 1.8, 7.32);
      dot(goalX + inward * 11, 34);
      const start = right ? Math.PI - spread : -spread;
      g.beginPath();
      g.arc(X(goalX + inward * 11), Y(34), 9.15 * SCALE, start, start + 2 * spread);
      g.stroke();
    }
    g.fillStyle = palette.chalk;
    g.font = `500 ${1.4 * SCALE}px "IBM Plex Mono", ui-monospace, monospace`;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(`${TEAM[meta.team].toUpperCase()} ATTACK →`, X(52.5), Y(68 + MARGIN / 2));
  }

  function arrow(from, to, color, alpha, dashed) {
    const ax = X(from[0]), ay = Y(from[1]), bx = X(to[0]), by = Y(to[1]);
    const angle = Math.atan2(by - ay, bx - ax);
    const head = 1.4 * SCALE;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 0.22 * SCALE;
    ctx.setLineDash(dashed ? [0.9 * SCALE, 0.6 * SCALE] : []);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - head * Math.cos(angle - 0.45), by - head * Math.sin(angle - 0.45));
    ctx.lineTo(bx - head * Math.cos(angle + 0.45), by - head * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function position(k, f) {
    const row = track[f];
    const x = row[2 * k];
    const y = row[2 * k + 1];
    return x === null || y === null ? null : [x, y];
  }

  function draw() {
    ctx.drawImage(layer, 0, 0);
    const t = frame / fps;
    const fade = 1.2;
    for (const ev of data.events) {
      if ((ev.type !== "PASS" && ev.type !== "SHOT") || !ev.start || !ev.end) continue;
      if (t < ev.t || t > ev.t_end + fade) continue;
      const alpha = t <= ev.t_end ? 0.95 : 0.95 * (1 - (t - ev.t_end) / fade);
      arrow(ev.start, ev.end, palette.chalk, alpha, ev.type === "PASS");
    }
    players.forEach((p, k) => {
      if (!focus.has(p.id)) return;
      ctx.save();
      ctx.strokeStyle = palette.marker;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 0.3 * SCALE;
      ctx.beginPath();
      let drawing = false;
      for (let f = Math.max(0, frame - 38); f <= frame; f++) {
        const pos = position(k, f);
        if (!pos) { drawing = false; continue; }
        if (drawing) ctx.lineTo(X(pos[0]), Y(pos[1]));
        else { ctx.moveTo(X(pos[0]), Y(pos[1])); drawing = true; }
      }
      ctx.stroke();
      ctx.restore();
    });
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `700 ${1.15 * SCALE}px "IBM Plex Mono", ui-monospace, monospace`;
    players.forEach((p, k) => {
      const pos = position(k, frame);
      if (!pos) return;
      const cx = X(pos[0]), cy = Y(pos[1]);
      if (focus.has(p.id)) {
        ctx.beginPath();
        ctx.arc(cx, cy, 2.0 * SCALE, 0, Math.PI * 2);
        ctx.strokeStyle = palette.marker;
        ctx.lineWidth = 0.4 * SCALE;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(cx, cy, 1.1 * SCALE, 0, Math.PI * 2);
      ctx.fillStyle = p.team === "home" ? palette.home : palette.away;
      ctx.fill();
      ctx.lineWidth = 0.18 * SCALE;
      ctx.strokeStyle = p.keeper ? "#f5f7f5" : "rgba(10, 16, 12, 0.55)";
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(p.id.slice(1), cx, cy + 0.05 * SCALE);
    });
    const ball = ballTrack[frame];
    if (ball) {
      ctx.beginPath();
      ctx.arc(X(ball[0]), Y(ball[1]), 0.6 * SCALE, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 0.16 * SCALE;
      ctx.strokeStyle = "#101512";
      ctx.stroke();
    }
  }

  const scrub = $("scrub");

  function setFrame(f) {
    frame = Math.max(0, Math.min(lastFrame, Math.round(f)));
    scrub.value = String(frame);
    const t = frame / fps;
    $("clock").textContent = `t ${t.toFixed(1)}s · ${matchClock(t)}`;
    draw();
  }

  function setPlaying(on) {
    playing = on;
    $("play").textContent = on ? "Pause" : "Play";
    $("play").setAttribute("aria-pressed", String(on));
    lastTime = null;
    carry = 0;
    if (on) {
      if (frame >= lastFrame) setFrame(0);
      requestAnimationFrame(advance);
    }
  }

  function advance(now) {
    if (!playing) return;
    if (lastTime !== null) {
      carry += ((now - lastTime) / 1000) * fps * SPEEDS[speedIndex];
      const whole = Math.floor(carry);
      if (whole > 0) {
        carry -= whole;
        setFrame(frame + whole);
      }
    }
    lastTime = now;
    if (frame >= lastFrame) {
      setPlaying(false);
      return;
    }
    requestAnimationFrame(advance);
  }

  $("play").addEventListener("click", () => setPlaying(!playing));
  canvas.addEventListener("click", () => setPlaying(!playing));
  $("speed").addEventListener("click", () => {
    speedIndex = (speedIndex + 1) % SPEEDS.length;
    $("speed").textContent = `${SPEEDS[speedIndex]}x`;
  });
  scrub.addEventListener("input", () => {
    setPlaying(false);
    setFrame(Number(scrub.value));
  });
  document.addEventListener("keydown", (e) => {
    if (e.target !== document.body) return;
    if (e.key === " ") {
      e.preventDefault();
      setPlaying(!playing);
    } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      setPlaying(false);
      setFrame(frame + (e.key === "ArrowRight" ? 5 : -5));
    }
  });

  function showFacts(ids, t) {
    setPlaying(false);
    const found = ids.map((id) => facts.get(id)).filter(Boolean);
    focus = new Set(found.flatMap((f) => f.players));
    const timed = found.find((f) => f.frame !== null);
    const when = typeof t === "number" ? t : timed ? timed.t : null;
    const panel = $("selected");
    panel.textContent = "";
    found.slice(0, 2).forEach((f, n) => {
      if (n) panel.append(" ");
      panel.append(el("b", null, f.id), ` ${f.text}`);
    });
    if (found.length > 2) panel.append(` +${found.length - 2} more cited`);
    document.querySelectorAll("[data-fact]").forEach((node) => {
      node.classList.toggle("active", ids.includes(node.dataset.fact));
    });
    if (when !== null) setFrame(when * fps);
    else draw();
  }

  const pct = (t) => `${Math.max(0, Math.min(100, (t / duration) * 100))}%`;

  function tickButton(cls, label, t, text, onClick) {
    const b = el("button", cls, text);
    b.type = "button";
    b.style.left = pct(t);
    b.title = label;
    b.setAttribute("aria-label", label);
    b.addEventListener("click", onClick);
    return b;
  }

  function buildTicks(analysis) {
    const box = $("ticks");
    const span = el("div", "span-mark");
    span.style.left = pct(meta.won_t);
    span.style.width = `calc(${pct(meta.final_t)} - ${pct(meta.won_t)})`;
    box.append(span);
    for (const ev of data.events) {
      if (ev.type !== "PASS" && ev.type !== "SHOT") continue;
      const label = `${ev.player || ""} ${ev.label}${ev.receiver ? ` to ${ev.receiver}` : ""}, t ${ev.t.toFixed(1)}s`;
      box.append(
        tickButton(ev.type === "SHOT" ? "tick shot" : "tick", label, ev.t, null, () => {
          setPlaying(false);
          focus = new Set([ev.player, ev.receiver].filter(Boolean));
          setFrame(ev.t * fps);
        })
      );
    }
    if (!analysis) return;
    const km = analysis.key_moment;
    const kmT = km && typeof km.t === "number" ? km.t : null;
    let kmShown = false;
    analysis.causal_chain.forEach((step, k) => {
      if (typeof step.t !== "number") return;
      const isKey = kmT !== null && Math.abs(step.t - kmT) < 0.3;
      kmShown = kmShown || isKey;
      const label = `Step ${k + 1}: ${step.label}${isKey ? " (key moment)" : ""}`;
      box.append(tickButton(isKey ? "tick step moment" : "tick step", label, step.t, String(k + 1), () => showFacts(step.evidence, step.t)));
    });
    if (kmT !== null && !kmShown) {
      box.append(tickButton("tick step moment", `Key moment: ${km.label}`, kmT, "K", () => showFacts(km.evidence, kmT)));
    }
  }

  /* Analysis notes */
  function section(title, eyebrow) {
    const s = el("section");
    if (eyebrow) s.append(el("p", "eyebrow", eyebrow));
    if (title) s.append(el("h2", null, title));
    return s;
  }

  function factChip(id) {
    const c = el("button", "chip", id);
    c.type = "button";
    c.dataset.fact = id;
    c.title = facts.has(id) ? facts.get(id).text : "This ID is not in the evidence ledger";
    c.addEventListener("click", () => showFacts([id]));
    return c;
  }

  // Turn inline citations such as "(F2, F9)" into the same clickable chips used elsewhere.
  function citedText(tag, cls, text) {
    const node = el(tag, cls);
    let last = 0;
    for (const match of text.matchAll(/\s*\((F\d+(?:,\s*F\d+)*)\)/g)) {
      node.append(text.slice(last, match.index), " ");
      const group = el("span", "cites");
      group.append(...match[1].split(/,\s*/).map((id) => factChip(id)));
      node.append(group);
      last = match.index + match[0].length;
    }
    node.append(text.slice(last));
    return node;
  }

  function evidenceRow(path, ids, t) {
    const row = el("div", "row");
    const check = checks.get(path);
    if (check) {
      const s = el("span", `status ${check.status}`, STATUS[check.status]);
      s.title = check.problems.length ? check.problems.join("; ") : "Every number and player here matches the cited measurements.";
      row.append(s);
    }
    if (typeof t === "number") {
      const c = el("button", "chip time", `t ${t.toFixed(1)}s`);
      c.type = "button";
      c.addEventListener("click", () => showFacts(ids || [], t));
      row.append(c);
    }
    for (const id of ids || []) row.append(factChip(id));
    return row;
  }

  function item(path, obj, extra) {
    const div = el("div", "item");
    if (obj.player) div.append(el("div", "who", obj.player));
    div.append(citedText("p", null, obj.detail));
    if (extra) div.append(extra);
    div.append(evidenceRow(path, obj.evidence));
    return div;
  }

  function chainSection(a) {
    if (!a.causal_chain.length) return null;
    const s = section("How it unfolded");
    const list = el("ol", "chain");
    a.causal_chain.forEach((step, k) => {
      const body = el("div");
      body.append(el("span", "label", step.label), citedText("p", null, step.detail), evidenceRow(`causal_chain[${k}]`, step.evidence, step.t));
      const li = el("li");
      li.append(el("span", "n", String(k + 1)), body);
      list.append(li);
    });
    s.append(list);
    return s;
  }

  function keySection(a) {
    const km = a.key_moment;
    if (!km) return null;
    const s = section("Key moment");
    const card = el("div", "key");
    const why = el("p");
    why.append(el("strong", null, "Why it mattered: "), km.why_it_mattered);
    card.append(el("span", "label", km.label), citedText("p", null, km.detail), why, evidenceRow("key_moment", km.evidence, km.t));
    s.append(card);
    return s;
  }

  function sidesSection(a) {
    const s = section("Both sides");
    const grid = el("div", "split");
    const column = (team, heading, items, path, advice, advicePath) => {
      const col = el("div");
      const h = el("h3");
      h.append(swatch(team), `${TEAM[team]}: ${heading}`);
      col.append(h, ...items.map((it, k) => item(`${path}[${k}]`, it)));
      const sub = el("h3", null, "Coaching points");
      sub.style.marginTop = "18px";
      col.append(sub, ...advice.map((it, k) => item(`${advicePath}[${k}]`, it)));
      return col;
    };
    grid.append(
      column(meta.team, "what worked", a.attack.what_worked, "attack.what_worked", a.attack.recommendations, "attack.recommendations"),
      column(other(meta.team), "what went wrong", a.defence.what_went_wrong, "defence.what_went_wrong", a.defence.recommendations, "defence.recommendations")
    );
    s.append(grid);
    return s;
  }

  function alternativesSection(a) {
    if (!a.alternatives.length) return null;
    const s = section("Other options at the key decision");
    a.alternatives.forEach((alt, k) => {
      const trade = el("p");
      trade.append(el("strong", null, "Trade-off: "), alt.tradeoff);
      s.append(item(`alternatives[${k}]`, alt, trade));
    });
    return s;
  }

  function limitsSection(a) {
    if (!a.intent_caveats.length && !a.data_limits.length) return null;
    const s = section("What the data can't tell us");
    a.intent_caveats.forEach((c, k) => s.append(item(`intent_caveats[${k}]`, c)));
    if (a.data_limits.length) {
      const list = el("ul", "limits");
      a.data_limits.forEach((d) => list.append(el("li", null, d)));
      s.append(list);
    }
    return s;
  }

  function groundingSection() {
    const v = data.verification;
    if (!v) return null;
    const s = section("Grounding check");
    const top = el("div", "grounding");
    const counts = el("div", "row");
    for (const key of ["verified", "partial", "unsupported"]) {
      counts.append(el("span", `status ${key}`, `${v.counts[key]} ${STATUS[key]}`));
    }
    top.append(el("span", "pct", `${Math.round(v.score * 100)}%`), counts);
    s.append(
      top,
      el("p", null, "Each claim cites fact IDs. PhaseMap checks that every cited fact exists and that every number and player in the claim appears in it. That tests grounding, not tactical judgement. Partly backed claims count half.")
    );
    for (const c of v.checks.filter((x) => x.status !== "verified")) {
      const p = el("p", "flag");
      p.append(el("code", null, c.path), ` ${c.problems.join("; ")}`);
      s.append(p);
    }
    return s;
  }

  function ledgerSection() {
    const s = section("Evidence ledger", `${data.facts.length} measured facts`);
    const filters = el("div", "row filters");
    const wrap = el("div", "ledger-wrap");
    const table = el("table", "ledger");
    const head = el("tr");
    for (const h of ["ID", "Time", "Kind", "Measurement"]) head.append(el("th", null, h));
    const thead = el("thead");
    thead.append(head);
    const tbody = el("tbody");
    for (const f of data.facts) {
      const tr = el("tr");
      tr.dataset.fact = f.id;
      tr.dataset.kind = f.kind;
      tr.tabIndex = 0;
      if (f.frame !== null) tr.dataset.seek = "1";
      tr.append(el("td", "id", f.id), el("td", "t", f.frame !== null ? `${f.t.toFixed(1)}s` : "match"), el("td", "kind", f.kind), el("td", null, f.text));
      tr.addEventListener("click", () => showFacts([f.id]));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter") showFacts([f.id]);
      });
      tbody.append(tr);
    }
    table.append(thead, tbody);
    wrap.append(table);
    const setFilter = (kind) => {
      filters.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.kind === kind)));
      tbody.querySelectorAll("tr").forEach((tr) => {
        tr.hidden = kind !== "all" && tr.dataset.kind !== kind;
      });
    };
    for (const kind of ["all", ...new Set(data.facts.map((f) => f.kind))]) {
      const c = el("button", "chip", kind);
      c.type = "button";
      c.dataset.kind = kind;
      c.addEventListener("click", () => setFilter(kind));
      filters.append(c);
    }
    s.append(filters, wrap);
    setFilter("all");
    return s;
  }

  function renderGoals(active) {
    const nav = $("goals");
    nav.hidden = bundles.length < 2;
    bundles.forEach((b, k) => {
      const m = b.meta;
      const btn = el("button");
      btn.type = "button";
      btn.setAttribute("aria-current", String(k === active));
      btn.setAttribute("aria-label", `${TEAM[m.team]} ${m.outcome} at ${m.clock_end}`);
      btn.append(swatch(m.team), el("span", null, m.clock_end), el("span", "tally", `${m.score_after.home}-${m.score_after.away}`));
      btn.addEventListener("click", () => {
        if (k === active) return;
        show(k);
        window.scrollTo(0, 0);
      });
      nav.append(btn);
    });
  }

  function renderScorebar() {
    const before = meta.score_before;
    const after = meta.score_after;
    const score = el("span", "score");
    score.append(swatch("home"), "Home ", el("b", null, `${before.home}-${before.away}`), " Away ", swatch("away"));
    const changed = after.home !== before.home || after.away !== before.away;
    const half = meta.period === 1 ? "1st half" : meta.period === 2 ? "2nd half" : `period ${meta.period}`;
    $("scorebar").append(
      score,
      el("span", "meta", `${meta.clock_start}-${meta.clock_end} · ${half}`),
      el("span", "outcome", changed ? `${cap(meta.outcome)} · ${after.home}-${after.away}` : cap(meta.outcome))
    );
  }

  function renderNotes() {
    const a = data.analysis;
    const head = el("header");
    const headline = el("h1", "headline", a ? a.headline : `${TEAM[meta.team]} ${meta.outcome}, ${meta.clock_start} to ${meta.clock_end}`);
    if (headline.textContent.length > 80) headline.classList.add("long");
    head.append(el("p", "eyebrow", `${meta.match} · ${TEAM[meta.team]} possession · ${cap(meta.outcome)}`), headline);
    if (a && a.summary) head.append(citedText("p", "summary", a.summary));
    const parts = [head];
    if (a) {
      parts.push(chainSection(a), keySection(a), sidesSection(a), alternativesSection(a), limitsSection(a), groundingSection());
    } else {
      const note = el("div", "notice");
      note.append(el("p", null, "The written breakdown for this phase has not been generated yet. The replay and every measured fact below are complete, and each fact is clickable."));
      parts.push(note);
    }
    parts.push(ledgerSection(), el("p", "source", `Data: ${meta.source}. Positions are shown from ${TEAM[meta.team]}'s attacking view. Generated ${meta.generated}.`));
    $("notes").append(...parts.filter(Boolean));
  }

  function openingFocus() {
    const km = data.analysis && data.analysis.key_moment;
    const shotFact = data.facts.find((f) => f.kind === "shot");
    if (km && typeof km.t === "number") showFacts(km.evidence || [], km.t);
    else if (shotFact) showFacts([shotFact.id]);
    else setFrame(meta.final_t * fps);
  }

  function show(index) {
    setPlaying(false);
    use(bundles[index]);
    focus = new Set();
    for (const id of ["goals", "scorebar", "ticks", "selected", "notes"]) $(id).textContent = "";
    scrub.max = String(lastFrame);
    renderGoals(index);
    renderScorebar();
    renderNotes();
    buildTicks(data.analysis);
    paintPitch();
    openingFocus();
    try {
      history.replaceState(null, "", `#phase-${meta.phase}`);
    } catch (err) {
      // Sandboxed hosts can refuse history changes; the page works without them.
    }
  }

  function repaint() {
    readPalette();
    paintPitch();
    draw();
  }

  readPalette();
  const wanted = /^#phase-(\d+)$/.exec(location.hash);
  show(wanted ? Math.max(0, bundles.findIndex((b) => String(b.meta.phase) === wanted[1])) : 0);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", repaint);
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  if (document.fonts) document.fonts.ready.then(repaint);
})();
