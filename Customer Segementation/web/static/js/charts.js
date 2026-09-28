// Thin wrappers around the vendored Chart.js (window.Chart) with consistent styling.
const active = new Set();

export function cssVar(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function applyThemeDefaults(Chart) {
  Chart.defaults.color = cssVar("--muted", "#6b7280");
  Chart.defaults.borderColor = cssVar("--chart-grid", "#eef0f4");
}

function chartLib() {
  if (!window.Chart) throw new Error("Chart library failed to load.");
  const Chart = window.Chart;
  const small = window.innerWidth < 640;
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = small ? 10 : 12;
  applyThemeDefaults(Chart);
  Chart.defaults.plugins.legend.labels.boxWidth = small ? 8 : 12;
  Chart.defaults.plugins.legend.labels.padding = small ? 8 : 12;
  Chart.defaults.maintainAspectRatio = false;
  return Chart;
}

// Chart.js re-resolves options on update(), so new defaults + scriptable colours pick up the theme.
export function applyChartTheme() {
  if (!window.Chart) return;
  applyThemeDefaults(window.Chart);
  active.forEach((chart) => chart.update("none"));
}

export function destroyCharts() {
  for (const chart of active) chart.destroy();
  active.clear();
}

export function destroyChart(chart) {
  if (chart && active.delete(chart)) chart.destroy();
}

export function canvasBox(label, size = "") {
  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", label);
  const box = document.createElement("div");
  box.className = `chart-box ${size}`.trim();
  box.append(canvas);
  return { box, canvas };
}

function make(canvas, config) {
  const Chart = chartLib();
  const chart = new Chart(canvas, config);
  active.add(chart);
  return chart;
}

export function withAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export function barChart(canvas, { labels, datasets, horizontal = false, stacked = false, xTitle, yTitle, tickFormat, legend = true }) {
  const valueAxis = horizontal ? "x" : "y";
  const scales = {
    x: { stacked, grid: { display: horizontal }, title: { display: !!xTitle, text: xTitle } },
    y: { stacked, grid: { display: !horizontal }, title: { display: !!yTitle, text: yTitle } },
  };
  if (tickFormat) scales[valueAxis].ticks = { callback: tickFormat };
  return make(canvas, {
    type: "bar",
    data: { labels, datasets },
    options: {
      indexAxis: horizontal ? "y" : "x",
      scales,
      plugins: {
        legend: { display: legend, position: "bottom" },
        tooltip: tickFormat ? { callbacks: { label: (ctx) => `${ctx.dataset.label ?? ""}: ${tickFormat(ctx.parsed[valueAxis])}` } } : {},
      },
    },
  });
}

export function scatterChart(canvas, { datasets, xTitle, yTitle, tooltip }) {
  return make(canvas, {
    type: "scatter",
    data: { datasets },
    options: {
      animation: false,
      parsing: false,
      scales: { x: { title: { display: true, text: xTitle } }, y: { title: { display: true, text: yTitle } } },
      plugins: { legend: { position: "bottom" }, tooltip: tooltip ? { callbacks: { label: tooltip } } : {} },
    },
  });
}

export function lineChart(canvas, { labels, data, highlightIndex, title }) {
  const line = () => cssVar("--primary", "#4f46e5");
  return make(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: title,
          data,
          borderColor: line,
          backgroundColor: line,
          pointRadius: labels.map((_, i) => (i === highlightIndex ? 7 : 3)),
          pointBackgroundColor: (ctx) => (ctx.dataIndex === highlightIndex ? "#e76f51" : line()),
          tension: 0.2,
        },
      ],
    },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Number of segments (k)" } } } },
  });
}

export function radarChart(canvas, { labels, datasets }) {
  return make(canvas, {
    type: "radar",
    data: { labels, datasets },
    options: {
      scales: {
        r: {
          suggestedMin: -2,
          suggestedMax: 2,
          ticks: { stepSize: 1, backdropColor: "transparent" },
          grid: { color: () => window.Chart.defaults.borderColor },
          angleLines: { color: () => window.Chart.defaults.borderColor },
          pointLabels: { font: { size: window.innerWidth < 640 ? 9 : 11 } },
        },
      },
      plugins: { legend: { position: "bottom" } },
      elements: { line: { borderWidth: 2 } },
    },
  });
}

export function rangeChart(canvas, { labels, stats, colors, title }) {
  // Floating bars show the inter-quartile range; points mark the median and the min/max whiskers.
  return make(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Middle 50% (Q1–Q3)",
          data: stats.map((s) => [s.q1, s.q3]),
          backgroundColor: colors.map((c) => withAlpha(c, 0.55)),
          borderColor: colors,
          borderWidth: 1,
          barPercentage: 0.5,
        },
        {
          type: "scatter",
          label: "Median",
          data: stats.map((s, i) => ({ x: labels[i], y: s.median })),
          backgroundColor: () => cssVar("--ink", "#111827"),
          pointStyle: "line",
          pointRadius: 18,
          borderWidth: 3,
          borderColor: () => cssVar("--ink", "#111827"),
        },
        {
          type: "scatter",
          label: "Min / max",
          data: stats.flatMap((s, i) => [{ x: labels[i], y: s.min }, { x: labels[i], y: s.max }]),
          backgroundColor: () => cssVar("--muted", "#98a2b3"),
          pointRadius: 3,
        },
      ],
    },
    options: {
      scales: { x: { type: "category" }, y: { title: { display: true, text: title } } },
      plugins: { legend: { position: "bottom" } },
    },
  });
}
