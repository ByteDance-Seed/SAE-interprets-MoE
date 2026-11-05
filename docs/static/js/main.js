(function () {
  "use strict";

  const DATA = window.PAPER_DATA;
  const N = 128;
  const LAYERS = [4, 30, 46];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const DOMAIN_COLORS = {
    "Computing": "#3b6ef5",
    "Text form": "#e5484d",
    "Mathematics and reasoning": "#12a594",
    "Applied science": "#f17a24",
    "Language": "#8e4ec6",
    "Culture and recreation": "#8d9a2b",
    "Entities and identifiers": "#a26a45",
    "Safety and advice": "#6b7280",
    "Commerce": "#0aa2c0",
    "Society and law": "#d6409f",
    "Time and place": "#5aa312",
  };

  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const experts = DATA.experts;
  const domainOrder = DATA.domains.map((d) => d.name);
  const rankByDegree = (() => {
    const order = experts.map((e) => e.id).sort((a, b) => experts[b].degree - experts[a].degree);
    const r = new Array(N);
    order.forEach((id, i) => (r[id] = i + 1));
    return r;
  })();
  const maxDegree = Math.max(...experts.map((e) => e.degree));

  /* ---------------- nav, progress, reveal ---------------- */

  function initNav() {
    const nav = $("#nav");
    const bar = $("#progress");
    const hero = $(".hero");
    const onScroll = () => {
      const h = document.documentElement;
      const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      bar.style.width = (p * 100).toFixed(2) + "%";
      nav.classList.toggle("show", h.scrollTop > hero.offsetHeight * 0.55);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    const links = $$("#nav-links a");
    const targets = links.map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
    const spy = () => {
      const y = window.scrollY + 120;
      let current = null;
      for (const t of targets) if (t.offsetTop <= y) current = t.id;
      links.forEach((a) => {
        const on = a.getAttribute("href") === "#" + current;
        a.classList.toggle("active", on);
        if (on && nav.classList.contains("show")) a.scrollIntoView({ block: "nearest", inline: "nearest" });
      });
    };
    let ticking = false;
    window.addEventListener("scroll", () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(() => { spy(); ticking = false; });
      }
    }, { passive: true });
  }

  function initReveal() {
    const els = $$(".reveal");
    if (reduceMotion || !("IntersectionObserver" in window)) {
      els.forEach((e) => e.classList.add("in"));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          en.target.classList.add("in");
          io.unobserve(en.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.04 });
    els.forEach((e) => io.observe(e));
  }

  /* ---------------- hero background ---------------- */

  function initHeroCanvas() {
    const canvas = $("#hero-canvas");
    const ctx = canvas.getContext("2d");
    let w, h, dpr, nodes;
    const COLORS = ["44,91,240", "122,76,243"];

    function setup() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.round(Math.min(80, (w * h) / 22000));
      nodes = Array.from({ length: count }, (_, i) => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.18,
        vy: (Math.random() - 0.5) * 0.18,
        r: 1.4 + Math.random() * 2.2,
        c: COLORS[i % 2],
      }));
    }

    function frame() {
      ctx.clearRect(0, 0, w, h);
      const maxD = 150;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d = Math.hypot(dx, dy);
          if (d < maxD) {
            ctx.strokeStyle = `rgba(${a.c},${0.16 * (1 - d / maxD)})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      for (const n of nodes) {
        ctx.fillStyle = `rgba(${n.c},0.35)`;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < -10 || n.x > w + 10) n.vx *= -1;
        if (n.y < -10 || n.y > h + 10) n.vy *= -1;
      }
    }

    setup();
    window.addEventListener("resize", setup);
    if (reduceMotion) { frame(); return; }

    let visible = true;
    new IntersectionObserver((en) => { visible = en[0].isIntersecting; }).observe(canvas);
    const loop = () => {
      if (visible) frame();
      requestAnimationFrame(loop);
    };
    loop();
  }

  /* ---------------- charts ---------------- */

  const RANK = {
    4: {
      g: [0.228385, 0.170646, 0.139857, 0.117821, 0.101261, 0.088947, 0.079887, 0.073193],
      c: [0.567694, 0.193636, 0.109129, 0.06529, 0.037442, 0.018584, 0.006854, 0.001371],
    },
    30: {
      g: [0.230709, 0.16931, 0.137491, 0.116342, 0.101124, 0.089777, 0.081058, 0.074186],
      c: [0.512017, 0.185317, 0.115389, 0.078591, 0.05342, 0.033443, 0.016858, 0.004964],
    },
    46: {
      g: [0.221303, 0.167379, 0.140202, 0.119784, 0.104272, 0.091635, 0.081666, 0.073756],
      c: [0.626773, 0.158527, 0.089251, 0.05637, 0.035613, 0.020709, 0.009856, 0.002901],
    },
  };

  const LATENT_DEGREE = {
    4: [7.03, 57.37, 32.51, 3.08, 0.009],
    30: [13.61, 62.1, 23.04, 1.25, 0.002],
    46: [3.69, 56.65, 36.02, 3.63, 0.014],
  };

  const LAYER_STYLE = {
    4: { color: "#2c5bf0", fill: "rgba(44,91,240,0.22)" },
    30: { color: "#c0262d", fill: "rgba(192,38,45,0.22)" },
    46: { color: "#3f4656", fill: "rgba(63,70,86,0.2)" },
  };

  function initCharts() {
    if (!window.Chart) return;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = "#5b6579";
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(13,20,36,0.94)";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.titleFont = { weight: "600" };
    const grid = { color: "rgba(15,23,42,0.06)" };

    LAYERS.forEach((L, idx) => {
      new Chart($("#rank-" + L), {
        type: "line",
        data: {
          labels: [0, 1, 2, 3, 4, 5, 6, 7],
          datasets: [
            {
              label: "gating weight", data: RANK[L].g, borderColor: "#5f6675", backgroundColor: "#5f6675",
              pointRadius: 3.5, pointHoverRadius: 6, borderWidth: 2, tension: 0.15,
            },
            {
              label: "contribution", data: RANK[L].c, borderColor: "#c0262d", backgroundColor: "#c0262d",
              pointStyle: "rect", pointRadius: 4, pointHoverRadius: 7, borderWidth: 2, borderDash: [6, 4], tension: 0.15,
              fill: { target: 0, above: "rgba(192,38,45,0.08)", below: "rgba(95,102,117,0.06)" },
            },
          ],
        },
        options: {
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => `Layer ${L} · rank r = ${items[0].label}`,
                label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(3)}`,
              },
            },
          },
          scales: {
            x: { grid, title: { display: true, text: "rank r" } },
            y: {
              grid, min: 0, max: 0.7, ticks: { stepSize: 0.1, display: idx === 0 },
              title: { display: idx === 0, text: "mean share" },
            },
          },
        },
      });
    });

    new Chart($("#latent-degree"), {
      type: "bar",
      data: {
        labels: ["0", "1", "2", "3", "4"],
        datasets: LAYERS.map((L) => ({
          label: "Layer " + L, data: LATENT_DEGREE[L],
          backgroundColor: LAYER_STYLE[L].fill, borderColor: LAYER_STYLE[L].color, borderWidth: 1.5, borderRadius: 4,
        })),
      },
      options: {
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top", labels: { boxWidth: 12, boxHeight: 12, usePointStyle: false } },
          tooltip: { callbacks: { title: (it) => `${it[0].label} dominator(s)`, label: (it) => ` ${it.dataset.label}: ${it.parsed.y}%` } },
        },
        scales: {
          x: { grid: { display: false }, title: { display: true, text: "number of dominators" } },
          y: { grid, min: 0, max: 70, title: { display: true, text: "share of latents (%)" } },
        },
      },
    });

    const valueLabels = {
      id: "valueLabels",
      afterDatasetsDraw(chart) {
        const { ctx } = chart;
        const meta = chart.getDatasetMeta(0);
        ctx.save();
        ctx.font = "600 12px " + Chart.defaults.font.family;
        ctx.fillStyle = "#283247";
        ctx.textBaseline = "middle";
        meta.data.forEach((bar, i) => {
          const v = chart.data.datasets[0].data[i];
          ctx.fillText(`${v}  ·  ${((v / N) * 100).toFixed(1)}%`, bar.x + 8, bar.y);
        });
        ctx.restore();
      },
    };

    new Chart($("#domain-chart"), {
      type: "bar",
      plugins: [valueLabels],
      data: {
        labels: domainOrder,
        datasets: [{
          data: DATA.domains.map((d) => d.count),
          backgroundColor: domainOrder.map((d) => DOMAIN_COLORS[d] + "38"),
          borderColor: domainOrder.map((d) => DOMAIN_COLORS[d]),
          hoverBackgroundColor: domainOrder.map((d) => DOMAIN_COLORS[d] + "70"),
          borderWidth: 1.6, borderRadius: 6, barPercentage: 0.78, categoryPercentage: 0.9,
        }],
      },
      options: {
        indexAxis: "y",
        maintainAspectRatio: false,
        layout: { padding: { right: 90 } },
        onHover: (e, el) => { e.native.target.style.cursor = el.length ? "pointer" : "default"; },
        onClick: (e, el) => {
          if (!el.length) return;
          explorer.setDomain(domainOrder[el[0].index]);
          document.getElementById("explorer").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (it) => ` ${it.parsed.x} experts (${((it.parsed.x / N) * 100).toFixed(1)}% of the layer)`,
              afterLabel: (it) => {
                const ids = experts.filter((e) => e.domain === domainOrder[it.dataIndex]).map((e) => "E" + e.id);
                const lines = [];
                for (let i = 0; i < ids.length; i += 8) lines.push(" " + ids.slice(i, i + 8).join(" "));
                return lines;
              },
            },
          },
        },
        scales: {
          x: { grid, min: 0, max: 30, title: { display: true, text: "number of experts" } },
          y: { grid: { display: false }, ticks: { color: "#283247", font: { weight: "600" } } },
        },
      },
    });
  }

  /* ---------------- jaccard heatmap ---------------- */

  const MAGMA = [
    [0, 0, 0, 4], [0.13, 28, 16, 68], [0.25, 79, 18, 123], [0.38, 129, 37, 129], [0.5, 181, 54, 122],
    [0.63, 229, 89, 100], [0.75, 251, 135, 97], [0.88, 254, 194, 135], [1, 252, 253, 191],
  ];
  function magma(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < MAGMA.length; i++) {
      if (t <= MAGMA[i][0]) {
        const a = MAGMA[i - 1], b = MAGMA[i];
        const u = (t - a[0]) / (b[0] - a[0]);
        return [a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3])].map(Math.round);
      }
    }
    return MAGMA[MAGMA.length - 1].slice(1);
  }

  const triIndex = (i, j) => {
    if (i > j) [i, j] = [j, i];
    return (i * (2 * N - i - 1)) / 2 + (j - i - 1);
  };

  function initHeatmap() {
    const canvas = $("#heat-canvas");
    const ctx = canvas.getContext("2d");
    const tip = $("#heat-tip");
    const stage = $("#heat-stage");
    const list = $("#heat-pairs");
    const S = canvas.width;
    const cell = S / N;
    const SAT = 0.05;

    const domainPerm = experts.slice().sort((a, b) =>
      domainOrder.indexOf(a.domain) - domainOrder.indexOf(b.domain) || a.id - b.id).map((e) => e.id);
    const indexPerm = Array.from({ length: N }, (_, i) => i);

    let layer = 30, order = "index", base = null, perm = indexPerm, pos = null;

    function J(a, b) { return a === b ? null : DATA.jaccard[layer][triIndex(a, b)]; }

    function draw() {
      perm = order === "domain" ? domainPerm : indexPerm;
      pos = new Array(N);
      perm.forEach((id, p) => (pos[id] = p));
      const img = ctx.createImageData(S, S);
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          const v = J(perm[r], perm[c]);
          const rgb = v === null ? [42, 47, 58] : magma(v / SAT);
          for (let y = Math.round(r * cell); y < Math.round((r + 1) * cell); y++) {
            for (let x = Math.round(c * cell); x < Math.round((c + 1) * cell); x++) {
              const k = (y * S + x) * 4;
              img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2]; img.data[k + 3] = 255;
            }
          }
        }
      }
      ctx.putImageData(img, 0, 0);
      if (order === "domain") {
        ctx.strokeStyle = "rgba(255,255,255,0.55)";
        ctx.lineWidth = 1.5;
        let acc = 0;
        DATA.domains.forEach((d) => {
          ctx.strokeRect(acc * cell + 0.75, acc * cell + 0.75, d.count * cell - 1.5, d.count * cell - 1.5);
          acc += d.count;
        });
      }
      base = ctx.getImageData(0, 0, S, S);
    }

    function highlight(a, b) {
      ctx.putImageData(base, 0, 0);
      if (a == null) return;
      const r = pos[a], c = pos[b];
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(0, r * cell, S, cell);
      ctx.fillRect(c * cell, 0, cell, S);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      [[r, c], [c, r]].forEach(([y, x]) => ctx.strokeRect(x * cell - 2, y * cell - 2, cell + 4, cell + 4));
    }

    function describe(a, b) {
      const v = J(a, b);
      let html = `<b>E${a} × E${b}</b> · layer ${layer}<br>J = ${v === null ? "—" : v.toFixed(4)}`;
      if (layer === 30) {
        const ea = experts[a], eb = experts[b];
        html += `<br><span class="muted">E${a}: ${esc(ea.label)} (${esc(ea.domain)})</span>`;
        html += `<br><span class="muted">E${b}: ${esc(eb.label)} (${esc(eb.domain)})</span>`;
      }
      return html;
    }

    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const c = Math.floor(((e.clientX - rect.left) / rect.width) * N);
      const r = Math.floor(((e.clientY - rect.top) / rect.height) * N);
      if (r < 0 || c < 0 || r >= N || c >= N) return;
      const a = perm[r], b = perm[c];
      highlight(a, b);
      tip.innerHTML = describe(a, b);
      tip.style.opacity = 1;
      const sr = stage.getBoundingClientRect();
      let x = e.clientX - sr.left + 14, y = e.clientY - sr.top + 14;
      if (x + 280 > sr.width) x = e.clientX - sr.left - 290;
      tip.style.left = Math.max(0, x) + "px";
      tip.style.top = y + "px";
    });
    canvas.addEventListener("mouseleave", () => { highlight(null); tip.style.opacity = 0; });

    function renderList() {
      const pairs = [];
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const v = DATA.jaccard[layer][triIndex(i, j)];
        if (v > SAT) pairs.push([i, j, v]);
      }
      pairs.sort((x, y) => y[2] - x[2]);
      let head = `${pairs.length} pairs with J &gt; 0.05 at layer ${layer}`;
      if (layer === 30) {
        const same = pairs.filter(([a, b]) => experts[a].domain === experts[b].domain).length;
        head += ` · ${same} within one domain`;
      }
      list.innerHTML = `<div class="pairs-head">${head}</div>` + pairs.map(([a, b, v], k) => {
        let lbl = "", cross = false;
        if (layer === 30) {
          cross = experts[a].domain !== experts[b].domain;
          lbl = `<div class="lbls">${esc(experts[a].label)}<br>${esc(experts[b].label)}</div>`;
        }
        return `<div class="pair${cross ? " cross" : ""}" data-a="${a}" data-b="${b}">
          <span class="rk">${k + 1}</span>
          <div><span class="ids">E${a} · E${b}</span>${cross ? '<span class="x">cross-domain</span>' : ""}${lbl}</div>
          <span class="j">${v.toFixed(3)}</span></div>`;
      }).join("");
      $$(".pair", list).forEach((el) => {
        el.addEventListener("mouseenter", () => highlight(+el.dataset.a, +el.dataset.b));
        el.addEventListener("mouseleave", () => highlight(null));
      });
    }

    $$("#heat-layer button").forEach((b) => b.addEventListener("click", () => {
      $$("#heat-layer button").forEach((x) => x.classList.toggle("on", x === b));
      layer = +b.dataset.layer;
      draw(); renderList();
    }));
    $$("#heat-order button").forEach((b) => b.addEventListener("click", () => {
      $$("#heat-order button").forEach((x) => x.classList.toggle("on", x === b));
      order = b.dataset.order;
      draw();
    }));

    draw();
    renderList();
  }

  /* ---------------- share bars ---------------- */

  function initShareBars() {
    const bars = $$(".sharebar");
    bars.forEach((el) => {
      const own = +el.dataset.own, rnd = +el.dataset.rnd;
      el.innerHTML = `
        <div class="row"><span>Share</span><div class="track"><div class="fill own" data-w="${own}"></div></div><span class="v">${own.toFixed(1)}%</span></div>
        <div class="row"><span>Random</span><div class="track"><div class="fill rnd" data-w="${rnd}"></div></div><span class="v">${rnd.toFixed(1)}%</span></div>`;
    });
    const fill = (el) => $$(".fill", el).forEach((f) => (f.style.width = f.dataset.w + "%"));
    if (reduceMotion || !("IntersectionObserver" in window)) { bars.forEach(fill); return; }
    const io = new IntersectionObserver((en) => en.forEach((x) => {
      if (x.isIntersecting) { fill(x.target); io.unobserve(x.target); }
    }), { threshold: 0.5 });
    bars.forEach((b) => io.observe(b));
  }

  /* ---------------- explorer ---------------- */

  const explorer = (() => {
    const grid = $("#ex-grid");
    const chips = $("#ex-domains");
    const search = $("#ex-search");
    const count = $("#ex-count");
    const empty = $("#ex-empty");
    const modal = $("#ex-modal");
    const allPanel = $("#ex-all");
    const moreBtn = $("#ex-more");
    const featured = (DATA.featured || []).map((f) => ({ ...experts[f.id], name: f.name }));
    const featuredIds = new Set(featured.map((e) => e.id));
    const rest = experts.filter((e) => !featuredIds.has(e.id));
    let state = { q: "", domain: null, sort: "domain" };
    let visible = [];
    let nav = [];
    let current = -1;

    function renderFeatured() {
      $("#ex-featured").innerHTML = featured.map((e, k) => `
        <button class="fcard" data-k="${k}" style="--dc:${DOMAIN_COLORS[e.domain]}">
          <img src="static/images/wordcloud/${e.id}f.webp" alt="Word cloud of expert ${e.id}" loading="lazy">
          <span class="fmeta"><span class="eid">E${e.id}</span><span class="dn">${esc(e.domain)}</span></span>
          <span class="fname">${esc(e.name)}</span>
        </button>`).join("");
      $$(".fcard").forEach((b) => b.addEventListener("click", () => open(featured, +b.dataset.k)));
    }

    function expand(on = true) {
      allPanel.hidden = !on;
      moreBtn.setAttribute("aria-expanded", String(on));
      moreBtn.classList.toggle("on", on);
      $("#ex-more-label").textContent = on ? "Show fewer" : `More experts (${rest.length})`;
    }

    function renderChips() {
      const counts = {};
      rest.forEach((e) => { counts[e.domain] = (counts[e.domain] || 0) + 1; });
      const all = `<button class="dchip${state.domain ? "" : " on"}" data-d="">All <span class="c">${rest.length}</span></button>`;
      chips.innerHTML = all + DATA.domains.filter((d) => counts[d.name]).map((d) =>
        `<button class="dchip${state.domain === d.name ? " on" : ""}" data-d="${esc(d.name)}"><i style="background:${DOMAIN_COLORS[d.name]}"></i>${esc(d.name)} <span class="c">${counts[d.name]}</span></button>`
      ).join("");
      $$(".dchip", chips).forEach((b) => b.addEventListener("click", () => setDomain(b.dataset.d || null)));
    }

    function highlightText(text, q) {
      if (!q) return esc(text);
      const i = text.toLowerCase().indexOf(q.toLowerCase());
      if (i < 0) return esc(text);
      return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + q.length)) + "</mark>" + esc(text.slice(i + q.length));
    }

    function render() {
      const q = state.q.trim();
      const ql = q.toLowerCase();
      const idq = /^e?(\d{1,3})$/i.exec(q);
      visible = rest.filter((e) => {
        if (state.domain && e.domain !== state.domain) return false;
        if (!q) return true;
        if (idq) return e.id === +idq[1];
        return e.label.toLowerCase().includes(ql) || e.domain.toLowerCase().includes(ql);
      });
      if (state.sort === "domain") {
        visible.sort((a, b) => domainOrder.indexOf(a.domain) - domainOrder.indexOf(b.domain) || a.id - b.id);
      } else if (state.sort === "degree") {
        visible.sort((a, b) => b.degree - a.degree);
      } else {
        visible.sort((a, b) => a.id - b.id);
      }
      grid.innerHTML = visible.map((e, k) => `
        <button class="ecard" data-k="${k}" style="--dc:${DOMAIN_COLORS[e.domain]}">
          <div class="top"><span class="eid">E${e.id}</span><span class="dn">${esc(e.domain)}</span></div>
          <div class="lb">${idq ? esc(e.label) : highlightText(e.label, q)}</div>
          <div class="deg"><span>${e.degree.toLocaleString()} latents</span><span class="t"><b style="width:${((e.degree / maxDegree) * 100).toFixed(1)}%"></b></span></div>
        </button>`).join("");
      $$(".ecard", grid).forEach((b) => b.addEventListener("click", () => open(visible, +b.dataset.k)));
      count.textContent = `${visible.length} of ${rest.length} experts`;
      empty.style.display = visible.length ? "none" : "block";
    }

    function setDomain(d) {
      if (d && !rest.some((e) => e.domain === d)) d = null;
      state.domain = d;
      expand(true);
      renderChips();
      render();
    }

    function show(e) {
      $("#ex-img").src = `static/images/wordcloud/${e.id}f.webp`;
      $("#ex-img").alt = `Explanation word cloud of expert ${e.id}`;
      $("#ex-id").textContent = "E" + e.id;
      const dom = $("#ex-dom");
      dom.textContent = e.domain;
      dom.style.background = DOMAIN_COLORS[e.domain];
      $("#ex-label").textContent = e.label;
      $("#ex-deg").textContent = e.degree.toLocaleString() + " latents";
      $("#ex-rank").textContent = `#${rankByDegree[e.id]} of 128`;
      const p = experts[e.partner];
      $("#ex-partner").innerHTML = `<a data-goto="${p.id}">E${p.id}</a> · J = ${e.partnerJ.toFixed(3)}`;
      $("#ex-partner a").addEventListener("click", () => {
        let k = nav.findIndex((x) => x.id === p.id);
        if (k >= 0) { open(nav, k); return; }
        k = featured.findIndex((x) => x.id === p.id);
        if (k >= 0) { open(featured, k); return; }
        k = visible.findIndex((x) => x.id === p.id);
        if (k < 0) { state = { q: "", domain: null, sort: state.sort }; search.value = ""; renderChips(); render(); k = visible.findIndex((x) => x.id === p.id); }
        open(visible, k);
      });
      const feats = e.topFeatures || [];
      const list = $("#ex-feats");
      const featExperts = DATA.featExperts || {};
      if (!feats.length) {
        list.innerHTML = `<li class="empty-feats">No feature explanations available for this expert.</li>`;
      } else {
        list.innerHTML = feats.map((f) => {
          const related = (featExperts[f.id] && featExperts[f.id].length)
            ? featExperts[f.id]
            : [e.id];
          const shared = related.length > 1;
          const tip = shared
            ? `In dominant-set samples of <b>${related.map((id) => "E" + id).join(", ")}</b>`
            : `In <b>E${related[0]}</b>'s dominant-set sample`;
          return `<li>
            <span class="fid${shared ? " shared" : ""}" tabindex="0">${esc(f.id)}<span class="ftip">${tip}</span></span>
            <span class="fexp">${esc(f.explanation)}</span>
          </li>`;
        }).join("");
      }
    }

    function open(list, k) {
      if (k < 0 || k >= list.length) return;
      nav = list;
      current = k;
      show(list[k]);
      $(".panel", modal).scrollTop = 0;
      modal.classList.add("open");
      document.body.style.overflow = "hidden";
    }
    function close() {
      modal.classList.remove("open");
      document.body.style.overflow = "";
    }
    function step(d) {
      if (!nav.length) return;
      open(nav, (current + d + nav.length) % nav.length);
    }

    function init() {
      renderFeatured();
      expand(false);
      moreBtn.addEventListener("click", () => expand(allPanel.hidden));
      renderChips();
      render();
      search.addEventListener("input", () => { state.q = search.value; render(); });
      $$("#ex-sort button").forEach((b) => b.addEventListener("click", () => {
        $$("#ex-sort button").forEach((x) => x.classList.toggle("on", x === b));
        state.sort = b.dataset.sort;
        render();
      }));
      $$("[data-close]", modal).forEach((el) => el.addEventListener("click", close));
      $("#ex-prev").addEventListener("click", () => step(-1));
      $("#ex-next").addEventListener("click", () => step(1));
      document.addEventListener("keydown", (e) => {
        if (!modal.classList.contains("open")) return;
        if (e.key === "Escape") close();
        else if (e.key === "ArrowLeft") step(-1);
        else if (e.key === "ArrowRight") step(1);
      });
    }

    return { init, setDomain };
  })();

  /* ---------------- misc ---------------- */

  function initLightbox() {
    const box = $("#lightbox");
    const img = $("#lightbox-img");
    const close = () => { box.classList.remove("open"); document.body.style.overflow = ""; };
    $$(".zoomable").forEach((el) => el.addEventListener("click", () => {
      img.src = el.currentSrc || el.src;
      img.alt = el.alt;
      box.classList.add("open");
      document.body.style.overflow = "hidden";
    }));
    $$("[data-close]", box).forEach((el) => el.addEventListener("click", close));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && box.classList.contains("open")) close(); });
  }

  function initCopy() {
    const btn = $("#copy-bib");
    btn.addEventListener("click", async () => {
      const text = $("#bibtex").textContent;
      try {
        await navigator.clipboard.writeText(text);
      } catch (err) {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  }

  function initMath() {
    if (!window.renderMathInElement) return;
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      throwOnError: false,
    });
  }

  initNav();
  initReveal();
  initHeroCanvas();
  initShareBars();
  explorer.init();
  initHeatmap();
  initLightbox();
  initCopy();
  initCharts();
  document.addEventListener("DOMContentLoaded", initMath);
})();
