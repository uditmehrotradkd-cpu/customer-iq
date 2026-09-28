export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function describe(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc || []).filter((p) => p !== "body").join(".")}: ${d.msg}`).join("; ");
  }
  return JSON.stringify(detail);
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`/api/v1${path}`, { credentials: "same-origin", ...options });
  } catch {
    throw new ApiError(0, "Network error: the server could not be reached.");
  }
  if (!response.ok) {
    let detail = "";
    try {
      detail = describe((await response.json()).detail);
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(response.status, detail || `Request failed (${response.status})`);
  }
  return response;
}

// Model outputs only change on redeploy, so GET responses are cached for the page session.
const cache = new Map();

export const api = {
  clearCache() {
    cache.clear();
  },
  async postForm(path, form) {
    const response = await request(path, { method: "POST", body: form });
    return response.json();
  },
  async postForBlob(path, body) {
    const response = await request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return response.blob();
  },
  async get(path) {
    if (!cache.has(path)) {
      const promise = request(path, { headers: { Accept: "application/json" } }).then((r) => r.json());
      cache.set(path, promise);
      promise.catch(() => cache.delete(path));
    }
    return cache.get(path);
  },
  async fresh(path) {
    const response = await request(path, { headers: { Accept: "application/json" }, cache: "no-store" });
    return response.json();
  },
  async put(path, body) {
    const response = await request(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
    return response.json();
  },
  async post(path, body) {
    const response = await request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
    return response.json();
  },
  async upload(path, file) {
    const form = new FormData();
    form.append("file", file);
    const response = await request(path, { method: "POST", body: form });
    return response.blob();
  },
  async download(path) {
    const response = await request(path);
    return response.blob();
  },
};
