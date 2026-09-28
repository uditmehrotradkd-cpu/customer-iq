import { api } from "./api.js";
import {
  card,
  dataTable,
  downloadBlob,
  errorBox,
  field,
  fmt,
  formatMetric,
  h,
  kpi,
  loading,
  prettify,
  select,
} from "./dom.js";
import {
  applyChartTheme,
  barChart,
  canvasBox,
  destroyChart,
  destroyCharts,
  lineChart,
  radarChart,
  rangeChart,
  scatterChart,
  withAlpha,
} from "./charts.js";
import { backgroundVideo, countUp, initMotionToggle, onCleanup, pageTransition, reducedMotion, reveal, runCleanups, tilt } from "./effects.js";
import { startBackdrop } from "./backdrop.js";
import { BAGS_VIDEO, CREDITS, EXPLORER_IMAGE, HERO_VIDEO, METHOD_IMAGE, persona } from "./media.js";
import { initThemeSwitch } from "./theme.js";
import { renderAgent } from "./agent-page.js";
import { fetchMe, renderLogin, signOut } from "./login.js";

const view = document.getElementById("view");
const state = { segments: [], explorer: null, user: null, workspace: null };
let renderToken = 0;

const isGeneric = () => state.workspace?.mode === "generic";
const words = () => state.workspace?.labels ?? { entity: "customers", entity_singular: "customer", value: "Revenue" };
const VALUE_FORMATS = {
  integer: fmt.integer,
  decimal3: (v) => (v == null ? "–" : Number(v).toFixed(3)),
  currency: fmt.currency,
  compactCurrency: fmt.compactCurrency,
  percent: fmt.percent,
  number: (v) => fmt.number(v, 2),
  compact: (v) => (v == null ? "–" : Math.abs(v) >= 10000 ? new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v) : fmt.number(v, 1)),
  text: (v) => v ?? "–",
};
const formatValue = (v, format) => (VALUE_FORMATS[format] ?? VALUE_FORMATS.number)(v);

const RADAR_FEATURES = [
  "Income", "Total_Spend", "Total_Purchases", "Avg_Order_Value", "Deal_Ratio",
  "NumWebVisitsMonth", "Web_Share", "Catalog_Share", "Campaigns_Accepted", "Recency",
];
const SPEND_FIELDS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"];
const CAMPAIGNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"];
const EDUCATION = [["Basic", "Basic"], ["Graduation", "Graduation"], ["2n Cycle", "2nd cycle"], ["Master", "Master"], ["PhD", "PhD"]];
const MARITAL = ["Married", "Together", "Single", "Divorced", "Widow", "Unknown"];
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

// ---------------------------------------------------------------- helpers
const segmentById = (id) => state.segments.find((s) => s.id === id);
const nameOf = (id) => segmentById(id)?.name ?? `Segment ${id}`;
const colorOf = (id) => segmentById(id)?.color ?? "#98a2b3";
const stillCurrent = (token) => token === renderToken;

function segmentLabel(id, name = nameOf(id)) {
  return h("span", { class: "bar-cell" }, h("span", { class: `dot dot-${id % 10}`, "aria-hidden": "true" }), name);
}

function banner({ title, text, image, video, badge, actions = [] }) {
  const media = [];
  if (video) {
    const { video: v, toggle } = backgroundVideo(video, "background video");
    media.push(v, toggle);
  } else if (image) {
    media.push(h("img", { class: "banner-img", src: image, alt: "", decoding: "async" }));
  }
  return h(
    "section",
    { class: `banner${media.length ? "" : " plain"}` },
    media,
    badge ? h("div", {}, h("span", { class: "badge" }, badge)) : null,
    h("h1", {}, title),
    text ? h("p", {}, text) : null,
    actions.length ? h("div", { class: "controls" }, actions) : null,
  );
}

function animatedKpi(label, target, format) {
  const el = kpi(label, format(0));
  const valueEl = el.querySelector(".value");
  requestAnimationFrame(() => countUp(valueEl, target, format));
  return el;
}

function segmentSlide(profile, rec, stats) {
  const id = profile.Segment;
  const media = persona(profile.Segment_Name, id);
  const stat = (label, value) => h("div", { class: "stat" }, h("div", { class: "k" }, label), h("div", { class: "v" }, value));
  return h(
    "article",
    { class: "slide", "aria-roledescription": "slide", "aria-label": profile.Segment_Name },
    h(
      "div",
      { class: "slide-media" },
      h("img", { src: media.image, alt: media.alt, loading: "lazy", decoding: "async" }),
      h("span", { class: `segment-stripe stripe-${id % 10}`, "aria-hidden": "true" }),
    ),
    h(
      "div",
      { class: "slide-body" },
      h("h3", {}, profile.Segment_Name),
      h("p", { class: "muted" }, rec?.summary ?? media.tagline),
      h(
        "div",
        { class: "slide-stats" },
        stats.map((s) => stat(s.label, formatValue(profile[s.key], s.format))),
      ),
      h("a", { class: "btn secondary", href: `#/segments/${id}` }, "View profile"),
    ),
  );
}

// 3D coverflow: the active slide faces the viewer, neighbours recede and rotate away.
function coverflow(profiles, recs, stats) {
  const slides = profiles.map((p) => segmentSlide(p, recs[p.Segment], stats));
  const n = slides.length;
  const track = h("div", { class: "coverflow-track" }, slides);
  const prev = h("button", { class: "cf-nav prev", type: "button", "aria-label": "Previous segment" }, "\u2039");
  const next = h("button", { class: "cf-nav next", type: "button", "aria-label": "Next segment" }, "\u203a");
  const dots = slides.map((_, i) => h("button", { class: "cf-dot", type: "button", "aria-label": `Show ${profiles[i].Segment_Name}` }));
  const root = h(
    "section",
    { class: "coverflow", "aria-roledescription": "carousel", "aria-label": "Customer segments", tabindex: "0" },
    track,
    prev,
    next,
    h("div", { class: "cf-dots" }, dots),
  );

  let index = 0;
  const layout = () => {
    const spread = root.clientWidth > 720 ? 64 : 58;
    slides.forEach((slide, i) => {
      let offset = i - index;
      if (offset > n / 2) offset -= n;
      if (offset < -n / 2) offset += n;
      const abs = Math.abs(offset);
      slide.style.transform = `translateX(calc(-50% + ${offset * spread}%)) translateZ(${-abs * 220}px) rotateY(${-offset * 42}deg)`;
      slide.style.opacity = abs > 1.5 ? "0" : String(1 - abs * 0.35);
      slide.style.zIndex = String(10 - abs);
      slide.classList.toggle("active", offset === 0);
      slide.setAttribute("aria-hidden", String(offset !== 0));
      slide.querySelector("a").tabIndex = offset === 0 ? 0 : -1;
    });
    dots.forEach((d, i) => d.setAttribute("aria-current", String(i === index)));
  };
  const go = (i) => {
    index = (i + n) % n;
    layout();
  };

  let timer = null;
  const stop = () => {
    clearInterval(timer);
    timer = null;
  };
  const start = () => {
    if (!reducedMotion && !timer) timer = setInterval(() => go(index + 1), 4500);
  };

  prev.addEventListener("click", () => (stop(), go(index - 1)));
  next.addEventListener("click", () => (stop(), go(index + 1)));
  dots.forEach((d, i) => d.addEventListener("click", () => (stop(), go(i))));
  slides.forEach((s, i) =>
    s.addEventListener("click", (e) => {
      if (i !== index) {
        e.preventDefault();
        stop();
        go(i);
      }
    }),
  );
  root.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") (stop(), go(index - 1));
    if (e.key === "ArrowRight") (stop(), go(index + 1));
  });
  let startX = null;
  track.addEventListener("pointerdown", (e) => (startX = e.clientX));
  track.addEventListener("pointerup", (e) => {
    if (startX === null) return;
    const dx = e.clientX - startX;
    startX = null;
    if (Math.abs(dx) > 40) (stop(), go(index + (dx < 0 ? 1 : -1)));
  });
  root.addEventListener("pointerenter", stop);
  root.addEventListener("focusin", stop);

  const resize = new ResizeObserver(layout);
  resize.observe(root);
  onCleanup(() => {
    stop();
    resize.disconnect();
  });
  requestAnimationFrame(() => {
    layout();
    start();
  });
  return root;
}

function shareBar(value) {
  return h("span", { class: "bar-cell" }, h("span", { class: "bar", style: { width: `${Math.max(2, value * 1.2)}px` } }), fmt.percent(value));
}

function heatColor(value) {
  const alpha = Math.min(Math.abs(value) / 2, 1) * 0.85;
  return value >= 0 ? `rgba(231, 111, 81, ${alpha})` : `rgba(69, 123, 157, ${alpha})`;
}

async function downloadFrom(path, filename, button) {
  button.disabled = true;
  try {
    downloadBlob(await api.download(path), filename);
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
  }
}

function profilesTable(profiles, columns) {
  const render = (format) => (format === "shareBar" ? { render: shareBar } : { num: format !== "text", format: (v) => formatValue(v, format) });
  return dataTable(
    [
      { key: "Segment_Name", label: "Segment", render: (v, row) => segmentLabel(row.Segment, v) },
      ...columns.map((c) => ({ key: c.key, label: c.label, ...render(c.format) })),
    ],
    profiles,
  );
}

// ---------------------------------------------------------------- overview
async function renderOverview(root, token) {
  const [overview, pca, universe] = await Promise.all([api.get("/overview"), api.get("/pca"), api.get("/pca3d")]);
  if (!stillCurrent(token)) return;
  const k = overview.kpis;
  const w = overview.labels ?? words();
  const generic = overview.mode === "generic";
  const profiles = overview.profiles;
  const recs = Object.fromEntries(overview.recommendations.map((r) => [r.segment, r]));
  const topTwo = [...profiles].sort((a, b) => b["Revenue_Share_%"] - a["Revenue_Share_%"]).slice(0, 2);
  const topCustomers = topTwo.reduce((s, p) => s + p["Customer_Share_%"], 0);
  const topRevenue = topTwo.reduce((s, p) => s + p["Revenue_Share_%"], 0);
  const valueName = w.value ? w.value.toLowerCase() : null;
  const lead = generic
    ? `${fmt.integer(k.customers)} ${w.entity} of ${w.dataset} grouped into ${k.segments} segments using every usable column.` +
      (valueName ? ` ${fmt.percent(topCustomers)} of ${w.entity} hold ${fmt.percent(topRevenue)} of total ${valueName}.` : "")
    : `${fmt.integer(k.customers)} customers in ${k.segments} behavioural segments. ${fmt.percent(topCustomers)} of customers drive ${fmt.percent(topRevenue)} of revenue.`;

  const { video, toggle } = backgroundVideo(HERO_VIDEO, "hero video");
  const hint = h("span", { class: "stage-hint" });
  const views = [
    { key: "skyline", label: "Skyline", hint: `Tower height = ${valueName ? `${valueName} share` : `share of ${w.entity}`} · tap a tower` },
    { key: "galaxy", label: "Galaxy", hint: `Each dot is a ${w.entity_singular ?? "row"} · drag to rotate` },
  ];
  const switchButtons = views.map((v) => h("button", { type: "button", "aria-pressed": "false" }, v.label));
  const canvasHost = h("div", { class: "stage-canvas" });
  const stage = h(
    "div",
    { class: "hero-stage" },
    h("div", { class: "stage-toolbar" }, h("div", { class: "view-switch", role: "group", "aria-label": "3D view" }, switchButtons), hint),
    canvasHost,
  );
  const hero = h(
    "section",
    { class: "hero" },
    video,
    h(
      "div",
      { class: "hero-copy" },
      h("span", { class: "eyebrow" }, generic ? `Your dataset · ${w.dataset}` : "Customer segmentation"),
      generic
        ? h("h1", {}, "Every row tells a story. ", h("span", { class: "accent-text" }, "Now you can see the patterns."))
        : h("h1", {}, "Every customer is different. ", h("span", { class: "accent-text" }, "Now you can see how.")),
      h("p", { class: "lead" }, lead),
      h(
        "div",
        { class: "hero-actions" },
        h("a", { class: "btn", href: "#/segments" }, "Explore segments"),
        h("a", { class: "btn ghost", href: "#/assign" }, generic ? "Assign a record" : "Assign a customer"),
      ),
      h("div", { class: "legend", "aria-hidden": "true" }, state.segments.map((s) => h("span", {}, h("span", { class: `dot dot-${s.id % 10}` }), s.name))),
      toggle,
    ),
    stage,
  );

  const sizes = canvasBox(valueName ? `Share of ${w.entity} versus share of ${valueName} by segment` : `Share of ${w.entity} by segment`);
  const map = canvasBox(`${w.entity} projected on two principal components, coloured by segment`, "tall");

  root.replaceChildren(
    hero,
    h("div", { class: "kpis" }, overview.kpi_list.map((item) => animatedKpi(item.label, item.value, (v) => formatValue(item.format === "integer" ? Math.round(v) : v, item.format)))),
    h("div", { class: "section-title" }, h("h2", {}, "Meet the segments"), h("p", {}, "Swipe or use the arrows to browse each segment.")),
    coverflow(profiles, recs, overview.slide_stats),
    h("div", { class: "section-title" }, h("h2", {}, valueName || !generic ? "Where the value is" : "How big each segment is"), h("p", {}, valueName ? `Share of ${w.entity} versus share of ${valueName}, and the 2D segment map.` : "Segment sizes and the 2D segment map.")),
    h(
      "div",
      { class: "grid grid-5-7" },
      card(
        valueName ? `${w.entity[0].toUpperCase()}${w.entity.slice(1)} vs. ${valueName}` : "Segment sizes",
        h("p", { class: "caption" }, generic ? (valueName ? `Does each segment hold more or less ${valueName} than its size suggests?` : `Share of all ${w.entity} in each segment.`) : "A minority of customers generates most of the revenue."),
        sizes.box,
      ),
      card(
        "Segment map",
        h("p", { class: "caption" }, `PCA projection: PC1 explains ${fmt.fraction(pca.variance[0])}, PC2 ${fmt.fraction(pca.variance[1])} of variance.`),
        map.box,
      ),
    ),
    h("div", { class: "stack" },
      card("Segment summary", profilesTable(profiles, overview.profile_columns)),
      h("div", { class: "callout" }, h("strong", {}, "How the number of segments was chosen: "), overview.rationale),
    ),
  );

  const entityLabel = `${w.entity[0].toUpperCase()}${w.entity.slice(1)} %`;
  barChart(sizes.canvas, {
    labels: profiles.map((p) => p.Segment_Name),
    datasets: [
      { label: entityLabel, data: profiles.map((p) => p["Customer_Share_%"]), backgroundColor: "#8da0cb", borderRadius: 6 },
      ...(valueName || !generic ? [{ label: `${w.value ?? "Revenue"} %`, data: profiles.map((p) => p["Revenue_Share_%"]), backgroundColor: "#e76f51", borderRadius: 6 }] : []),
    ],
    horizontal: true,
    tickFormat: (v) => `${v}%`,
  });
  scatterChart(map.canvas, {
    datasets: state.segments.map((s) => ({
      label: s.name,
      data: pca.points.filter((p) => p.segment === s.id),
      backgroundColor: withAlpha(s.color, 0.55),
      pointRadius: 2.5,
    })),
    xTitle: "PC1",
    yTitle: "PC2",
    tooltip: (ctx) => `${nameOf(ctx.raw.segment)} · ID ${ctx.raw.id}${ctx.raw.info ? ` · ${ctx.raw.info}` : ""}`,
  });

  // Three.js is only downloaded for the overview page.
  try {
    const [{ mountTowers }, { mountUniverse }] = await Promise.all([import("./towers.js"), import("./universe.js")]);
    if (!stillCurrent(token)) return;
    let dispose = () => {};
    const show = (key) => {
      dispose();
      views.forEach((v, i) => switchButtons[i].setAttribute("aria-pressed", String(v.key === key)));
      hint.textContent = views.find((v) => v.key === key).hint;
      dispose =
        key === "skyline"
          ? mountTowers(canvasHost, profiles, colorOf, (segment) => (location.hash = `#/segments/${segment}`))
          : mountUniverse(canvasHost, universe, colorOf);
    };
    views.forEach((v, i) => switchButtons[i].addEventListener("click", () => show(v.key)));
    onCleanup(() => dispose());
    show("skyline");
  } catch {
    canvasHost.classList.add("no-webgl");
  }
}

// ---------------------------------------------------------------- segment profiles
async function renderSegments(root, token, param) {
  const requested = Number.parseInt(param, 10);
  const selected = segmentById(requested) ? requested : state.segments[0].id;
  const [detail, fingerprint] = await Promise.all([api.get(`/segments/${selected}`), api.get("/fingerprint")]);
  if (!stillCurrent(token)) return;
  const rec = detail.recommendation;

  const tabs = h(
    "div",
    { class: "segment-tabs", role: "tablist", "aria-label": "Segments" },
    state.segments.map((s) =>
      h(
        "button",
        { class: "segment-tab", type: "button", role: "tab", "aria-selected": s.id === selected ? "true" : "false", onclick: () => (location.hash = `#/segments/${s.id}`) },
        h("span", { class: `dot dot-${s.id % 10}`, "aria-hidden": "true" }),
        s.name,
      ),
    ),
  );

  const metrics = h(
    "div",
    { class: "kpis" },
    detail.metrics.map((m) => {
      const delta = m.delta_pct == null ? null : `${m.delta_pct > 0 ? "+" : ""}${m.delta_pct.toFixed(0)}% vs. average`;
      return kpi(m.label, formatMetric(m.value, m.format), delta);
    }),
  );

  const charts = (detail.charts ?? []).map((c) => ({ ...c, ...canvasBox(c.caption ?? c.title, "short") }));
  const radar = canvasBox("Radar comparing all segments", "tall");
  const w = detail.labels ?? words();
  const exportBtn = h("button", { class: "btn secondary", type: "button" }, `Download ${w.entity} (CSV)`);
  exportBtn.addEventListener("click", () => downloadFrom(`/customers/export?segment=${selected}`, `customers_segment_${selected}.csv`, exportBtn));

  const heatmap = h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      { class: "heatmap" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Segment"), fingerprint.features.map((f) => h("th", { scope: "col" }, prettify(f))))),
      h(
        "tbody",
        {},
        fingerprint.rows.map((row) =>
          h(
            "tr",
            { class: row.segment === selected ? "highlight" : "" },
            h("td", {}, segmentLabel(row.segment, row.name)),
            row.values.map((v) => h("td", { class: "cell", style: { background: heatColor(v) }, title: `${v} standard deviations` }, v.toFixed(1))),
          ),
        ),
      ),
    ),
  );

  root.replaceChildren(
    banner({ title: detail.name, text: rec.summary, image: persona(detail.name, detail.id).image, badge: `Goal: ${rec.goal}`, actions: [exportBtn] }),
    tabs,
    h("p", { class: "muted" }, `${fmt.integer(detail.customers)} ${w.entity}` + (w.value ? ` · ${fmt.percent(detail.revenue_share)} of ${w.value.toLowerCase()}` : "")),
    metrics,
    h(
      "div",
      { class: "grid grid-3" },
      card(
        "What defines this segment",
        h("ul", { class: "traits" }, rec.defining_high_traits.map((t) => h("li", { class: "high" }, `▲ ${t}`)), rec.defining_low_traits.map((t) => h("li", { class: "low" }, `▼ ${t}`))),
        rec.preferred_channel ? h("p", { class: "muted" }, `Preferred channel: ${rec.preferred_channel} · Over-indexed categories: ${rec.over_indexed_categories.join(", ")}`) : null,
        !rec.preferred_channel && rec.over_indexed_categories?.length ? h("p", { class: "muted" }, `Stands out on: ${rec.over_indexed_categories.join(", ")}`) : null,
      ),
      charts.map((c) => card(c.title, c.box)),
    ),
    h(
      "div",
      { class: "grid grid-2" },
      card(
        "Recommended engagement actions",
        h("ol", { class: "actions" }, rec.actions.map((a) => h("li", {}, a))),
        h("p", {}, h("strong", {}, "KPIs to track: "), rec.kpis.join(", ")),
        h("div", { class: "guardrail", role: "note" }, h("strong", {}, "Responsible-use guardrail: "), rec.responsible_use),
      ),
      card("Segment radar", h("p", { class: "caption" }, `Standard deviations from the average ${w.entity_singular ?? "row"} (clipped to ±2).`), radar.box),
    ),
    card("Segment fingerprint", h("p", { class: "caption" }, "Red = above average, blue = below average (standard deviations)."), heatmap),
  );

  tabs.querySelector('[aria-selected="true"]')?.scrollIntoView({ inline: "center", block: "nearest" });
  const color = detail.color;
  charts.forEach((c) => {
    const entries = Object.entries(c.values).sort((a, b) => b[1] - a[1]);
    barChart(c.canvas, {
      labels: entries.map(([key]) => prettify(key)),
      datasets: [{ label: c.title, data: entries.map(([, v]) => v), backgroundColor: color }],
      horizontal: true,
      legend: false,
      tickFormat: (v) => `${v}${c.unit ?? ""}`,
    });
  });
  let idx = RADAR_FEATURES.map((f) => fingerprint.features.indexOf(f)).filter((i) => i >= 0);
  if (idx.length < 3) idx = fingerprint.features.slice(0, 10).map((_, i) => i);
  radarChart(radar.canvas, {
    labels: idx.map((i) => prettify(fingerprint.features[i])),
    datasets: fingerprint.rows.map((row) => {
      const isSelected = row.segment === selected;
      const c = colorOf(row.segment);
      return {
        label: row.name,
        data: idx.map((i) => Math.max(-2, Math.min(2, row.values[i]))),
        borderColor: isSelected ? c : withAlpha(c, 0.45),
        backgroundColor: isSelected ? withAlpha(c, 0.25) : "transparent",
        borderWidth: isSelected ? 3 : 1,
        pointRadius: isSelected ? 3 : 0,
      };
    }),
  });
}

// ---------------------------------------------------------------- explorer
async function renderExplorer(root, token) {
  const features = await api.get("/explorer/features");
  if (!stillCurrent(token)) return;
  const pick = (list, preferred, fallback = 0) => (list.includes(preferred) ? preferred : list[Math.min(fallback, list.length - 1)]);
  state.explorer ??= {
    segments: new Set(state.segments.map((s) => s.id)),
    feature: pick(features.numeric, "Total_Spend"),
    x: pick(features.numeric, "Income"),
    y: pick(features.numeric, "Total_Spend", 1),
    attribute: pick(features.categorical, "Age_Band"),
  };
  const hasCategories = features.categorical.length > 0;
  const ex = state.explorer;
  const charts = {};

  const segmentChecks = h(
    "div",
    { class: "checks", role: "group", "aria-label": "Segments to show" },
    state.segments.map((s) => {
      const box = h("input", { type: "checkbox", checked: ex.segments.has(s.id) });
      box.addEventListener("change", () => {
        box.checked ? ex.segments.add(s.id) : ex.segments.delete(s.id);
        if (ex.segments.size === 0) {
          ex.segments.add(s.id);
          box.checked = true;
        }
        refreshAll();
      });
      return h("label", {}, box, segmentLabel(s.id, s.name));
    }),
  );

  const featureSel = select(features.numeric, ex.feature);
  const xSel = select(features.numeric, ex.x);
  const ySel = select(features.numeric, ex.y);
  const attrSel = select(features.categorical, ex.attribute);
  const dist = canvasBox("Distribution of the selected feature by segment");
  const scat = canvasBox("Scatter plot of two selected features by segment");
  const demo = canvasBox("Demographic composition of each segment");
  const exportBtn = h("button", { class: "btn secondary", type: "button" }, "Download all (CSV)");
  exportBtn.addEventListener("click", () => downloadFrom("/customers/export", isGeneric() ? "rows_segmented.csv" : "customers_segmented.csv", exportBtn));

  root.replaceChildren(
    banner({ title: "Segment explorer", text: isGeneric() ? `Compare any column of ${words().dataset} across segments.` : "Compare any behavioural or descriptive feature across segments.", image: EXPLORER_IMAGE, badge: "Interactive analysis", actions: [exportBtn] }),
    card(null, h("h2", {}, "Segments"), segmentChecks),
    h(
      "div",
      { class: "grid grid-2" },
      card("Distribution", h("div", { class: "controls" }, field("Feature", featureSel)), dist.box),
      card("Relationship", h("div", { class: "controls" }, field("X axis", xSel), field("Y axis", ySel)), scat.box),
    ),
    hasCategories
      ? card(
          isGeneric() ? "Category mix" : "Descriptive demographics",
          h("p", { class: "caption" }, features.categorical_note),
          h("div", { class: "controls" }, field("Attribute", attrSel)),
          demo.box,
        )
      : null,
  );

  const visible = () => state.segments.filter((s) => ex.segments.has(s.id));

  async function drawDistribution() {
    const data = await api.get(`/explorer/distribution?feature=${encodeURIComponent(ex.feature)}`);
    if (!stillCurrent(token)) return;
    const stats = data.stats.filter((s) => ex.segments.has(s.segment));
    destroyChart(charts.dist);
    charts.dist = rangeChart(dist.canvas, { labels: stats.map((s) => nameOf(s.segment)), stats, colors: stats.map((s) => colorOf(s.segment)), title: prettify(ex.feature) });
  }

  async function drawScatter() {
    const params = new URLSearchParams({ x: ex.x, y: ex.y });
    [...ex.segments].sort().forEach((s) => params.append("segments", s));
    const data = await api.get(`/explorer/scatter?${params}`);
    if (!stillCurrent(token)) return;
    destroyChart(charts.scat);
    charts.scat = scatterChart(scat.canvas, {
      datasets: visible().map((s) => ({ label: s.name, data: data.points.filter((p) => p.segment === s.id), backgroundColor: withAlpha(s.color, 0.55), pointRadius: 2.5 })),
      xTitle: prettify(ex.x),
      yTitle: prettify(ex.y),
      tooltip: (ctx) => `${nameOf(ctx.raw.segment)} · ID ${ctx.raw.id} · (${fmt.number(ctx.raw.x)}, ${fmt.number(ctx.raw.y)})`,
    });
  }

  async function drawDemographics() {
    const data = await api.get(`/explorer/demographics?attribute=${encodeURIComponent(ex.attribute)}`);
    if (!stillCurrent(token)) return;
    const rows = data.rows.filter((r) => ex.segments.has(r.segment));
    const palette = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#b56576", "#6c63ff"];
    destroyChart(charts.demo);
    charts.demo = barChart(demo.canvas, {
      labels: rows.map((r) => nameOf(r.segment)),
      datasets: data.categories.map((c, i) => ({ label: c, data: rows.map((r) => r.values[i]), backgroundColor: palette[i % palette.length] })),
      horizontal: true,
      stacked: true,
      tickFormat: (v) => `${v}%`,
    });
  }

  function refreshAll() {
    Promise.all([drawDistribution(), drawScatter(), hasCategories ? drawDemographics() : null]).catch((err) => root.prepend(errorBox(err.message)));
  }

  featureSel.addEventListener("change", () => { ex.feature = featureSel.value; drawDistribution(); });
  xSel.addEventListener("change", () => { ex.x = xSel.value; drawScatter(); });
  ySel.addEventListener("change", () => { ex.y = ySel.value; drawScatter(); });
  attrSel.addEventListener("change", () => { ex.attribute = attrSel.value; drawDemographics(); });
  refreshAll();
}

// ---------------------------------------------------------------- assign
function numberInput(name, value, { min = 0, max, step = 1 } = {}) {
  return h("input", { type: "number", name, value, min, max, step, required: true, inputmode: "decimal" });
}

async function renderAssign(root, token) {
  const data = await api.get("/assign/defaults");
  if (!stillCurrent(token)) return;
  if (data.mode === "generic") return renderAssignGeneric(root, data);
  const { defaults, reference_date: referenceDate } = data;

  const inputs = {};
  const num = (name, label, opts) => field(label, (inputs[name] = numberInput(name, defaults[name], opts)));
  const incomeInput = (inputs.Income = h("input", { type: "number", name: "Income", value: defaults.Income, min: 0, max: 1000000, step: "any", placeholder: "Leave blank if unknown" }));
  inputs.Education = select(EDUCATION, "Graduation", { name: "Education" });
  inputs.Marital_Status = select(MARITAL.map((m) => [m, m]), "Married", { name: "Marital_Status" });
  inputs.Dt_Customer = h("input", { type: "date", name: "Dt_Customer", value: defaults.Dt_Customer, max: referenceDate, min: "2000-01-01", required: true });
  const campaignBoxes = CAMPAIGNS.map((c) => h("input", { type: "checkbox", name: c }));
  const complain = h("input", { type: "checkbox", name: "Complain" });

  const submit = h("button", { class: "btn", type: "submit" }, "Assign segment");
  const result = h("div", { class: "stack", "aria-live": "polite" });
  const form = h(
    "form",
    { novalidate: false },
    h(
      "div",
      { class: "grid grid-3" },
      h(
        "fieldset",
        {},
        h("legend", {}, "Profile"),
        num("Year_Birth", "Year of birth", { min: 1920, max: 2010 }),
        field("Annual household income ($)", incomeInput),
        num("Kidhome", "Children at home", { max: 10 }),
        num("Teenhome", "Teenagers at home", { max: 10 }),
        field("Education", inputs.Education),
        field("Marital status", inputs.Marital_Status),
        field("Customer since", inputs.Dt_Customer),
      ),
      h(
        "fieldset",
        {},
        h("legend", {}, "Spend in the last 2 years ($)"),
        SPEND_FIELDS.map((f) => num(f, prettify(f).replace("Products", "").replace("Prods", "").trim(), { max: 100000 })),
      ),
      h(
        "fieldset",
        {},
        h("legend", {}, "Activity"),
        num("NumWebPurchases", "Web purchases", { max: 1000 }),
        num("NumCatalogPurchases", "Catalog purchases", { max: 1000 }),
        num("NumStorePurchases", "Store purchases", { max: 1000 }),
        num("NumDealsPurchases", "Purchases with a discount", { max: 1000 }),
        num("NumWebVisitsMonth", "Web visits last month", { max: 500 }),
        num("Recency", "Days since last purchase", { max: 3650 }),
      ),
    ),
    h(
      "fieldset",
      {},
      h("legend", {}, "Engagement"),
      h("div", { class: "checks" }, campaignBoxes.map((b, i) => h("label", {}, b, i < 5 ? `Campaign ${i + 1}` : "Last campaign")), h("label", {}, complain, "Complained in last 2 years")),
    ),
    h("div", { class: "form-actions" }, submit),
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const payload = {};
    for (const [name, input] of Object.entries(inputs)) {
      if (input.type === "number") payload[name] = input.value === "" ? null : Number(input.value);
      else payload[name] = input.value;
    }
    campaignBoxes.forEach((b) => (payload[b.name] = b.checked));
    payload.Complain = complain.checked;
    submit.disabled = true;
    submit.textContent = "Assigning…";
    try {
      const res = await api.post("/assign", payload);
      showAssignment(result, res);
      result.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      result.replaceChildren(errorBox(err.message));
    } finally {
      submit.disabled = false;
      submit.textContent = "Assign segment";
    }
  });

  root.replaceChildren(
    banner({ title: "Assign a customer", text: "Enter a customer's profile and transactions to place them in the closest segment and get their next best actions.", video: BAGS_VIDEO, badge: "Real-time scoring" }),
    card(null, form),
    result,
    batchCard(),
  );
}

function renderAssignGeneric(root, data) {
  const w = data.labels ?? words();
  const inputs = data.fields.map((f) => {
    let control;
    if (f.type === "select") control = select(f.options.map((o) => [o, o]), f.value, { name: f.name });
    else if (f.type === "date") control = h("input", { type: "date", name: f.name, value: f.value });
    else control = h("input", { type: "number", name: f.name, value: f.value, step: f.step, inputmode: "decimal", placeholder: `e.g. ${f.value}` });
    return { spec: f, control };
  });
  const submit = h("button", { class: "btn", type: "submit" }, "Assign segment");
  const result = h("div", { class: "stack", "aria-live": "polite" });
  const form = h(
    "form",
    {},
    h(
      "fieldset",
      { class: "generic-fields" },
      h("legend", {}, `Columns of ${w.dataset}`),
      inputs.map(({ spec, control }) => field(spec.label, control)),
    ),
    h("p", { class: "caption" }, "Fields start at the typical (median / most common) value. Leave a number empty if it is unknown."),
    h("div", { class: "form-actions" }, submit),
  );
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = {};
    inputs.forEach(({ spec, control }) => {
      values[spec.name] = spec.type === "number" ? (control.value === "" ? null : Number(control.value)) : control.value || null;
    });
    submit.disabled = true;
    submit.textContent = "Assigning…";
    try {
      const res = await api.post("/assign/generic", { values });
      showAssignment(result, res);
      result.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      result.replaceChildren(errorBox(err.message));
    } finally {
      submit.disabled = false;
      submit.textContent = "Assign segment";
    }
  });
  root.replaceChildren(
    banner({ title: "Assign a record", text: `Enter the values of one ${w.entity_singular} to see which segment of ${w.dataset} it belongs to.`, video: BAGS_VIDEO, badge: "Real-time scoring" }),
    card(null, form),
    result,
    batchCard(),
  );
}

let assignChart = null;
function showAssignment(container, res) {
  const rec = res.recommendation;
  const distances = canvasBox("Distance from the customer to each segment centroid", "short");
  const comparison = Object.keys(res.customer_features).map((k) => ({ feature: k, customer: res.customer_features[k], segment: res.segment_average[k] }));
  container.replaceChildren(
    card(
      null,
      h("div", { class: "page-header" }, h("div", {}, h("h2", {}, segmentLabel(res.segment, `Assigned segment: ${res.segment_name}`)), h("p", {}, rec.summary)), h("span", { class: `badge ${res.fit}` }, `${res.fit} fit · margin ${fmt.fraction(res.assignment_margin)}`)),
      res.warnings.map((w) => h("div", { class: "warning", role: "status" }, w)),
    ),
    h(
      "div",
      { class: "grid grid-2" },
      card("Distance to each segment", h("p", { class: "caption" }, "Lower is closer. The margin compares the nearest and second-nearest segment."), distances.box),
      card(
        "This customer vs. segment average",
        dataTable(
          [
            { key: "feature", label: "Feature", format: prettify },
            { key: "customer", label: "Customer", num: true, format: (v) => fmt.number(v) },
            { key: "segment", label: "Segment avg", num: true, format: (v) => fmt.number(v) },
          ],
          comparison,
        ),
      ),
    ),
    card(
      "Next best actions",
      h("ol", { class: "actions" }, rec.actions.map((a) => h("li", {}, a))),
      h("p", {}, rec.preferred_channel ? [h("strong", {}, "Preferred channel: "), rec.preferred_channel, " · "] : null, h("strong", {}, "KPIs: "), rec.kpis.join(", ")),
      h("div", { class: "guardrail", role: "note" }, h("strong", {}, "Guardrail: "), rec.responsible_use),
    ),
  );
  const entries = Object.entries(res.distances).sort((a, b) => a[1] - b[1]);
  destroyChart(assignChart);
  assignChart = barChart(distances.canvas, {
    labels: entries.map(([name]) => name),
    datasets: [{ label: "Distance", data: entries.map(([, d]) => d), backgroundColor: entries.map(([name]) => colorOf(state.segments.find((s) => s.name === name)?.id)) }],
    horizontal: true,
    legend: false,
  });
}

function batchCard() {
  const fileInput = h("input", { type: "file", accept: ".csv,.tsv,.txt,.xlsx,.xlsm,.xls,.json,.jsonl,.ndjson", "aria-label": "Customer file (CSV, Excel or JSON)" });
  const button = h("button", { class: "btn", type: "button" }, "Download report with charts");
  const csvButton = h("button", { class: "btn secondary", type: "button" }, "Download CSV only");
  const status = h("p", { class: "muted", role: "status" });
  const score = async (format, btn) => {
    const file = fileInput.files[0];
    if (!file) return (status.textContent = "Choose a file first.");
    if (file.size > MAX_UPLOAD_BYTES) return (status.textContent = "File is larger than 5 MB.");
    btn.disabled = true;
    status.textContent = format === "xlsx" ? "Scoring and building your report…" : "Scoring…";
    try {
      const name = format === "xlsx" ? "segment_report.xlsx" : "scored_customers.csv";
      downloadBlob(await api.upload(`/score/batch?format=${format}`, file), name);
      status.textContent = format === "xlsx" ? "Done. Open the Excel report for charts and a summary of every segment." : "Done. The scored file has been downloaded.";
    } catch (err) {
      status.textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  };
  button.addEventListener("click", () => score("xlsx", button));
  csvButton.addEventListener("click", () => score("csv", csvButton));
  return card(
    "Batch scoring",
    h("p", { class: "caption" }, isGeneric()
      ? `Upload a CSV, Excel or JSON file with the same columns as ${words().dataset} (max 5 MB) to assign every row to a segment. Missing columns are filled with typical values.`
      : "Upload a CSV, Excel (.xlsx/.xls) or JSON file in the raw or one-hot-encoded training schema (max 5 MB) to assign segments in bulk. The report opens in Excel with charts, segment profiles, next steps and every scored row."),
    h("div", { class: "controls" }, fileInput, button, csvButton),
    status,
  );
}

// ---------------------------------------------------------------- diagnostics
async function renderDiagnostics(root, token) {
  const d = await api.get("/diagnostics");
  if (!stillCurrent(token)) return;
  const metrics = d.k_metrics;
  const ks = metrics.map((m) => String(m.k));
  const selectedIdx = metrics.findIndex((m) => m.recommended);
  const panels = [
    ["inertia", "Elbow method: within-cluster SSE"],
    ["silhouette", "Silhouette (higher is better)"],
    ["davies_bouldin", "Davies-Bouldin (lower is better)"],
    ["stability_ari", "Bootstrap stability ARI (higher is better)"],
  ]
    .filter(([key]) => metrics.some((m) => m[key] != null))
    .map(([key, title]) => ({ key, title, ...canvasBox(title, "short") }));
  const num = (digits) => (v) => (typeof v === "number" ? fmt.number(v, digits) : v == null ? "–" : String(v));
  const bool = (v) => (v ? "Yes" : "No");
  const has = (key) => metrics.some((m) => m[key] != null);
  const generic = d.mode === "generic";

  root.replaceChildren(
    banner({
      title: "Model & methodology",
      text: generic ? `How ${words().dataset} was prepared, how the number of segments was chosen and how the segments were validated.` : "How the data was cleaned, how the number of segments was chosen and how the model was validated.",
      image: METHOD_IMAGE,
      badge: "Transparent by design",
    }),
    h(
      "div",
      { class: "kpis" },
      kpi("Selected k", String(d.selected_k)),
      generic ? null : kpi("Elbow k", String(d.elbow_k)),
      kpi("Best raw silhouette k", String(d.best_silhouette_k)),
      kpi("Final silhouette", d.silhouette.toFixed(3)),
    ),
    h("div", { class: "callout" }, d.rationale),
    h("div", { class: "grid grid-2" }, panels.map((p) => card(p.title, p.box))),
    card(
      "Candidate k scores",
      dataTable(
        [
          { key: "k", label: "k", num: true },
          { key: "inertia", label: "Inertia", num: true, format: num(0) },
          { key: "silhouette", label: "Silhouette", num: true, format: num(3) },
          { key: "calinski_harabasz", label: "Calinski-Harabasz", num: true, format: num(0) },
          { key: "davies_bouldin", label: "Davies-Bouldin", num: true, format: num(3) },
          { key: "min_segment_share", label: "Smallest segment", num: true, format: fmt.fraction },
          { key: "stability_ari", label: "Stability ARI", num: true, format: num(3) },
          { key: "composite_score", label: "Composite", num: true, format: num(3) },
          { key: "eligible", label: "Eligible", format: bool },
        ].filter((c) => has(c.key)),
        metrics,
        { highlight: (row) => row.recommended },
      ),
    ),
    d.columns?.length
      ? card(
          "Columns of your file",
          h("p", { class: "caption" }, "How each column was used. Identifiers and personal data are never used to form segments."),
          dataTable(
            [
              { key: "column", label: "Column" },
              { key: "used_as", label: "Used as" },
              { key: "reason", label: "Why not used", format: (v) => v || "–" },
            ],
            d.columns,
            { highlight: (row) => row.used_as === "not used" },
          ),
        )
      : null,
    h(
      "div",
      { class: "grid grid-2" },
      d.algorithms.length
        ? card(
        "Algorithm comparison",
        h("p", { class: "caption" }, "K-Means is deployed: best separation and it can assign new customers."),
        dataTable(
          [
            { key: "algorithm", label: "Algorithm" },
            { key: "silhouette", label: "Silhouette", num: true, format: num(3) },
            { key: "calinski_harabasz", label: "CH", num: true, format: num(0) },
            { key: "davies_bouldin", label: "DB", num: true, format: num(3) },
            { key: "min_segment_share", label: "Smallest", num: true, format: fmt.fraction },
            { key: "assigns_new_customers", label: "Scores new", format: bool },
          ],
          d.algorithms,
          { highlight: (row) => row.algorithm === "K-Means" },
        ),
      )
        : null,
      card(
        "Skewness before / after transformation",
        dataTable(
          [
            { key: "feature", label: "Feature", format: prettify },
            { key: "skew_before", label: "Before", num: true, format: num(2) },
            { key: "skew_after", label: "After", num: true, format: num(2) },
            { key: "log_transformed", label: "log1p", format: bool },
          ],
          d.skewness,
        ),
      ),
    ),
    d.cleaning.length
      ? card(
      "Data cleaning audit",
      dataTable(
        [
          { key: "step", label: "Step" },
          { key: "rows_before", label: "Before", num: true, format: fmt.integer },
          { key: "rows_after", label: "After", num: true, format: fmt.integer },
          { key: "rows_removed", label: "Removed", num: true, format: fmt.integer },
          { key: "detail", label: "Detail" },
        ],
        d.cleaning,
      ),
    )
      : null,
    d.proxy_check.length
      ? card(
      "Responsible AI: demographic proxy check",
      h("p", { class: "caption" }, "Cramér's V between segment membership and demographics not used for clustering. V ≥ 0.3 needs human review before targeting."),
      dataTable(
        [
          { key: "attribute", label: "Attribute", format: prettify },
          { key: "cramers_v", label: "Cramér's V", num: true, format: num(3) },
          { key: "review_needed", label: "Review needed", format: bool },
        ],
        d.proxy_check,
        { highlight: (row) => row.review_needed },
      ),
    )
      : null,
    card(null, h("details", {}, h("summary", {}, "Model card"), h("pre", { class: "json" }, JSON.stringify(d.model_card, null, 2)))),
  );

  panels.forEach((p) => lineChart(p.canvas, { labels: ks, data: metrics.map((m) => m[p.key]), highlightIndex: selectedIdx, title: p.title }));
}

// ---------------------------------------------------------------- router
const routes = {
  overview: { title: "Overview", render: renderOverview },
  segments: { title: "Segment profiles", render: renderSegments },
  explorer: { title: "Explorer", render: renderExplorer },
  assign: { title: "Assign a customer", render: renderAssign },
  diagnostics: { title: "Model & methodology", render: renderDiagnostics },
  agent: {
    title: "Data agent",
    render: async (root) =>
      renderAgent(root, {
        user: state.user,
        onModelChanged: () => {
          state.segments = [];
          state.explorer = null;
          state.workspace = null;
          showStatus();
        },
        onSignedOut: showLogin,
      }),
  },
};

async function navigate() {
  if (!state.user) return;
  const [name, param] = location.hash.replace(/^#\/?/, "").split("/");
  const key = routes[name] ? name : "overview";
  document.querySelectorAll("[data-route]").forEach((a) => {
    const on = a.dataset.route === key;
    a.classList.toggle("active", on);
    on ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current");
  });
  document.getElementById("nav").classList.remove("open");
  document.getElementById("nav-toggle").setAttribute("aria-expanded", "false");
  document.title = `${routes[key].title} · Customer Segmentation`;

  const token = ++renderToken;
  runCleanups();
  destroyCharts();
  view.replaceChildren(loading());
  window.scrollTo({ top: 0 });
  try {
    if (!state.segments.length || !state.workspace) {
      [state.segments, state.workspace] = await Promise.all([api.get("/segments"), api.get("/health")]);
      document.querySelector('[data-route="assign"]').textContent = isGeneric() ? "Assign a record" : "Assign a customer";
    }
    document.title = `${key === "assign" && isGeneric() ? "Assign a record" : routes[key].title} · Customer Segmentation`;
    await routes[key].render(view, token, param);
    if (stillCurrent(token)) {
      pageTransition(view);
      view.querySelectorAll(".kpi").forEach((el) => tilt(el, { max: 6, scale: 1.01 }));
      reveal(view);
    }
  } catch (err) {
    if (stillCurrent(token)) view.replaceChildren(errorBox(`Could not load this page: ${err.message}`));
  }
}

function renderCredits() {
  const list = document.getElementById("credits-list");
  list.replaceChildren(
    ...CREDITS.map((c) => h("li", {}, `${c.item} — `, h("a", { href: c.url, target: "_blank", rel: "noopener noreferrer" }, c.author), " (Pexels)")),
  );
}

async function showStatus() {
  const status = document.getElementById("footer-status");
  try {
    const health = await api.fresh("/health");
    const trained = health.trained_at_utc ? new Date(health.trained_at_utc).toLocaleDateString() : "unknown";
    const source = health.mode === "generic" ? health.labels?.dataset ?? "your dataset" : health.model_source === "published" ? "your data" : "demo data";
    status.textContent = `Model v${health.version} · ${health.n_segments} segments · ${source} · trained ${trained}`;
  } catch {
    status.textContent = "Model unavailable";
  }
}

function resetSession() {
  api.clearCache();
  state.segments = [];
  state.explorer = null;
  state.workspace = null;
  runCleanups();
  destroyCharts();
}

function renderUserMenu() {
  const menu = document.getElementById("user-menu");
  const user = state.user;
  if (!user) {
    menu.classList.add("hidden");
    menu.replaceChildren();
    return;
  }
  const initials = user.name.split(" ").filter(Boolean).map((p) => p[0]).slice(0, 2).join("").toUpperCase() || "?";
  const details = h(
    "details",
    { class: "user-dropdown" },
    h("summary", { "aria-label": `Account menu for ${user.name}` }, h("span", { class: "avatar user" }, initials), h("span", { class: "user-name" }, user.name)),
    h(
      "div",
      { class: "user-panel" },
      h("strong", {}, user.name),
      h("span", { class: "muted" }, `@${user.username}`),
      h("a", { class: "btn secondary", href: "#/agent" }, "Upload my data"),
      h("button", { class: "btn", type: "button", onclick: async () => {
        await signOut();
        showLogin();
      } }, "Sign out"),
    ),
  );
  menu.replaceChildren(details);
  menu.classList.remove("hidden");
}

document.addEventListener("click", (e) => {
  const dropdown = document.querySelector(".user-dropdown[open]");
  if (dropdown && (!dropdown.contains(e.target) || e.target.closest("a"))) dropdown.removeAttribute("open");
});

function showLogin() {
  state.user = null;
  resetSession();
  renderUserMenu();
  document.body.classList.add("signed-out");
  document.title = "Sign in · Customer Segmentation";
  document.getElementById("footer-status").textContent = "Sign in to load your workspace.";
  renderLogin(view, (user) => {
    state.user = user;
    resetSession();
    startApp();
  });
}

function startApp() {
  document.body.classList.remove("signed-out");
  renderUserMenu();
  navigate();
  showStatus();
}

async function boot() {
  state.user = await fetchMe();
  state.user ? startApp() : showLogin();
}

document.getElementById("nav-toggle").addEventListener("click", (e) => {
  const nav = document.getElementById("nav");
  const open = nav.classList.toggle("open");
  e.currentTarget.setAttribute("aria-expanded", String(open));
});
window.addEventListener("hashchange", () => {
  navigate();
  document.getElementById("main").focus({ preventScroll: true });
});
initThemeSwitch(applyChartTheme);
initMotionToggle();
startBackdrop();
boot();
renderCredits();
