const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000";

export function getToken() {
  return localStorage.getItem("smartinvest_token");
}

export function setToken(token) {
  localStorage.setItem("smartinvest_token", token);
}

export function clearToken() {
  localStorage.removeItem("smartinvest_token");
}

export function getUsername() {
  return localStorage.getItem("smartinvest_username");
}

export function setUsername(username) {
  localStorage.setItem("smartinvest_username", username);
}

async function request(path, { method = "GET", body, auth = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.error || data.message || `Request failed (${res.status})`);
  }
  return data;
}

export const api = {
  register: (username, password) =>
    request("/api/auth/register", { method: "POST", body: { username, password }, auth: false }),
  login: (username, password) =>
    request("/api/auth/login", { method: "POST", body: { username, password }, auth: false }),

  etfQuotes: () => request("/api/etf/quotes"),
  etfRecommend: (ticker) => request("/api/etf/recommend", { method: "POST", body: { ticker } }),
  etfTrack: (ticker) => request("/api/etf/track", { method: "POST", body: { ticker } }),
  etfNews: (ticker) => request(`/api/etf/news${ticker ? `?ticker=${ticker}` : ""}`),

  stockQuotes: () => request("/api/stocks/quotes"),
  stockRecommend: (ticker) => request("/api/stocks/recommend", { method: "POST", body: { ticker } }),
  stockTrack: (ticker) => request("/api/stocks/track", { method: "POST", body: { ticker } }),
  stockNews: (ticker) => request(`/api/stocks/news${ticker ? `?ticker=${ticker}` : ""}`),
};
