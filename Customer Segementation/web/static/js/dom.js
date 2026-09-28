// Safe DOM construction: all text goes through textContent / createTextNode (no innerHTML).
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style") Object.assign(el.style, value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const currency0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const currency2 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 0, maximumFractionDigits: 1 });
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export const fmt = {
  currency: (v) => (v == null ? "–" : Math.abs(v) >= 100 ? currency0.format(v) : currency2.format(v)),
  compactCurrency: (v) => (v == null ? "–" : `$${compact.format(v)}`),
  fraction: (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`),
  percent: (v) => (v == null ? "–" : `${Number(v).toFixed(1)}%`),
  number: (v, digits = 2) => (v == null ? "–" : Number(v).toLocaleString("en-US", { maximumFractionDigits: digits })),
  integer: (v) => (v == null ? "–" : Math.round(v).toLocaleString("en-US")),
};

export function formatMetric(value, kind) {
  if (kind === "currency") return fmt.currency(value);
  if (kind === "percent") return fmt.fraction(value);
  return fmt.number(value, 2);
}

const LABELS = {
  NumWebVisitsMonth: "Web visits / month",
  Total_Spend: "Total spend",
  Total_Purchases: "Total purchases",
  Avg_Order_Value: "Avg order value",
  Deal_Ratio: "Deal ratio",
  Web_Share: "Web share",
  Catalog_Share: "Catalog share",
  Store_Share: "Store share",
  Campaigns_Accepted: "Campaigns accepted",
  Tenure_Days: "Tenure (days)",
  Spend_To_Income: "Spend / income",
  Age_Band: "Age band",
  Is_Parent: "Parental status",
  Education_Level: "Education level",
  Marital_Status: "Marital status",
};

export function prettify(name) {
  return LABELS[name] ?? String(name).replace(/_/g, " ").replace(/^Mnt/, "").replace(/^Num/, "");
}

export function card(title, ...children) {
  return h("section", { class: "card" }, title ? h("h2", {}, title) : null, ...children);
}

export function kpi(label, value, delta = null, direction = null) {
  return h(
    "div",
    { class: "kpi" },
    h("div", { class: "label" }, label),
    h("div", { class: "value" }, value),
    delta ? h("div", { class: `delta ${direction ?? ""}` }, delta) : null,
  );
}

export function dataTable(columns, rows, { highlight } = {}) {
  const head = h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "", scope: "col" }, c.label)));
  const body = rows.map((row) =>
    h(
      "tr",
      { class: highlight && highlight(row) ? "highlight" : "" },
      columns.map((c) => {
        const raw = row[c.key];
        const content = c.render ? c.render(raw, row) : c.format ? c.format(raw) : raw ?? "–";
        return h("td", { class: c.num ? "num" : "" }, content);
      }),
    ),
  );
  return h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
}

export function field(label, control) {
  const id = control.id || `f-${Math.random().toString(36).slice(2, 9)}`;
  control.id = id;
  return h("div", { class: "field" }, h("label", { for: id }, label), control);
}

export function select(options, value, attrs = {}) {
  return h(
    "select",
    attrs,
    options.map((o) => {
      const [val, label] = Array.isArray(o) ? o : [o, prettify(o)];
      return h("option", { value: val, selected: String(val) === String(value) }, label);
    }),
  );
}

export function loading(text = "Loading…") {
  return h("div", { class: "loading", role: "status" }, text);
}

export function errorBox(message) {
  return h("div", { class: "error", role: "alert" }, message);
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
