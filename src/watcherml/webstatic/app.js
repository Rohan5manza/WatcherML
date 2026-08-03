// WatcherML local UI. Vanilla JS, no build step -- keeps the local-mode
// install dependency-free (matches "no Docker, no bundler" for local mode).

const app = document.getElementById("app");
const routeProgress = document.getElementById("route-progress");

function notify(message, kind = "success", timeout = 3200) {
  const region = document.getElementById("toast-region");
  if (!region) return;
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.innerHTML = `<span class="toast-symbol">${kind === "error" ? "!" : "✓"}</span><span class="toast-message">${esc(message)}</span>`;
  region.appendChild(toast);
  window.setTimeout(() => {
    toast.classList.add("out");
    window.setTimeout(() => toast.remove(), 220);
  }, timeout);
}

function startRouteProgress() {
  if (!routeProgress) return;
  routeProgress.classList.remove("done");
  routeProgress.classList.add("active");
}

function finishRouteProgress() {
  if (!routeProgress) return;
  routeProgress.classList.remove("active");
  routeProgress.classList.add("done");
  window.setTimeout(() => routeProgress.classList.remove("done"), 260);
}

function setupGlobalUI() {
  const palette = document.getElementById("command-palette");
  const search = document.getElementById("command-search");
  const trigger = document.getElementById("command-trigger");
  const sidebar = document.getElementById("sidebar");
  const mobileButton = document.getElementById("mobile-menu-button");

  const closePalette = () => {
    if (!palette) return;
    palette.hidden = true;
    if (search) search.value = "";
    document.querySelectorAll("[data-command-item]").forEach((item) => { item.hidden = false; });
  };
  const openPalette = () => {
    if (!palette) return;
    palette.hidden = false;
    window.setTimeout(() => search?.focus(), 0);
  };

  trigger?.addEventListener("click", openPalette);
  document.querySelectorAll("[data-command-close]").forEach((el) => el.addEventListener("click", closePalette));
  document.querySelectorAll("[data-command-item]").forEach((el) => el.addEventListener("click", closePalette));
  search?.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    document.querySelectorAll("[data-command-item]").forEach((item) => {
      item.hidden = q && !item.textContent.toLowerCase().includes(q);
    });
  });

  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      palette?.hidden ? openPalette() : closePalette();
    }
    if (event.key === "Escape") {
      closePalette();
      sidebar?.classList.remove("open");
      mobileButton?.setAttribute("aria-expanded", "false");
    }
  });

  mobileButton?.addEventListener("click", () => {
    const isOpen = sidebar?.classList.toggle("open");
    mobileButton.setAttribute("aria-expanded", String(Boolean(isOpen)));
  });
  document.querySelectorAll("#sidebar-nav a").forEach((link) => link.addEventListener("click", () => {
    sidebar?.classList.remove("open");
    mobileButton?.setAttribute("aria-expanded", "false");
  }));

  document.addEventListener("click", async (event) => {
    const copyButton = event.target.closest("[data-copy]");
    if (!copyButton) return;
    try {
      await navigator.clipboard.writeText(copyButton.dataset.copy || "");
      notify(copyButton.dataset.copyLabel || "Copied to clipboard");
    } catch (_) {
      notify("Clipboard access was unavailable", "error");
    }
  });
}

function formatGpuTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "—";
  const total = Math.max(0, Number(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m`;
}

function firstFinite(...values) {
  for (const value of values) {
    const n = Number(value);
    if (value !== null && value !== undefined && Number.isFinite(n)) return n;
  }
  return null;
}

function formatPatch(patch) {
  if (!patch || typeof patch !== "object") return "No configuration change";
  const entries = Object.entries(patch);
  if (!entries.length) return "No configuration change";
  return entries.map(([key, value]) => `${key.replaceAll("_", " ")} → ${typeof value === "boolean" ? (value ? "enabled" : "disabled") : value}`).join(" · ");
}

async function api(path, options) {
  const res = await fetch("/api" + path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

function esc(v) {
  if (v === null || v === undefined) return "&mdash;";
  return String(v).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtNum(v, digits = 4) {
  if (v === null || v === undefined) return "&mdash;";
  if (typeof v !== "number") return esc(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "&mdash;";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

function fmtTimestamp(ts) {
  if (!ts) return "&mdash;";
  const d = new Date(ts * 1000);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function badge(status) {
  const cls = status === "success" ? "success" : status === "failed" ? "failed" : "running";
  return `<span class="badge ${cls}">${esc(status || "running")}</span>`;
}

function demoBadge(simulated) {
  return simulated ? `<span class="pill demo">Simulated OOM Scenario</span>` : "";
}

function resolvedBadge(resolved) {
  return resolved ? `<span class="pill resolved">Resolved</span>` : "";
}

function tagPills(tags) {
  if (!tags || !tags.length) return "";
  return tags.map((t) => `<span class="pill tag">${esc(t)}</span>`).join(" ");
}

function provenance(kind) {
  const labels = {
    "rule-based": "Rule-based", "calculated": "Calculated",
    "ollama": "Ollama-generated", "verified": "Verified outcome",
  };
  return `<span class="provenance ${kind}">${labels[kind] || kind}</span>`;
}

// -------------------- trace strip --------------------

function renderTrace(samples, containerLabel) {
  if (!samples || samples.length < 2) {
    return `<div class="trace"><div class="trace-label"><span>${containerLabel}</span></div>
      <div class="trace-empty">not enough samples recorded for a trace (very short run)</div></div>`;
  }
  const hasGpu = samples.some((s) => s.gpu_util_pct !== null && s.gpu_util_pct !== undefined);
  const key = hasGpu ? "gpu_util_pct" : "cpu_pct";
  const label = hasGpu ? "gpu utilization %" : "cpu utilization % (no gpu detected)";
  const values = samples.map((s) => (s[key] === null || s[key] === undefined ? 0 : s[key]));
  const t0 = samples[0].t;
  const tSpan = Math.max(1, samples[samples.length - 1].t - t0);
  const W = 1000, H = 64, PAD = 4;
  const points = samples.map((s, i) => {
    const x = PAD + ((s.t - t0) / tSpan) * (W - PAD * 2);
    const y = H - PAD - (Math.min(100, Math.max(0, values[i])) / 100) * (H - PAD * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const gridLines = [0.25, 0.5, 0.75].map((f) =>
    `<line x1="0" x2="${W}" y1="${H * f}" y2="${H * f}" stroke="var(--border-hairline)" stroke-width="1"/>`
  ).join("");
  return `<div class="trace">
    <div class="trace-label"><span>${containerLabel} -- ${label}</span><span>${samples.length} samples</span></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${gridLines}
      <polyline points="${points}" fill="none" stroke="var(--signal-mint)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
  </div>`;
}

function renderSparkline(values, color = "var(--signal-mint)") {
  if (!values || values.length < 2) return `<div class="trace-empty">not enough trials for a chart yet</div>`;
  const W = 600, H = 210, PAD_X = 10, PAD_Y = 22;
  const min = Math.min(...values), max = Math.max(...values);
  const margin = Math.max((max - min) * 0.28, Math.abs(max || 1) * 0.025);
  const low = min - margin, high = max + margin;
  const span = high - low || 1;
  const coords = values.map((v, i) => ({
    x: PAD_X + (i / (values.length - 1)) * (W - PAD_X * 2),
    y: H - PAD_Y - ((v - low) / span) * (H - PAD_Y * 2),
  }));
  const points = coords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const last = coords[coords.length - 1];
  const areaPoints = `${PAD_X},${H - PAD_Y} ${points} ${W - PAD_X},${H - PAD_Y}`;
  const grid = [0.25, 0.5, 0.75].map((f) =>
    `<line x1="${PAD_X}" x2="${W - PAD_X}" y1="${PAD_Y + (H - PAD_Y * 2) * f}" y2="${PAD_Y + (H - PAD_Y * 2) * f}" stroke="rgba(109,125,147,.20)" stroke-width="1"/>`
  ).join("");
  const gid = `objective-gradient-${Math.random().toString(36).slice(2)}`;
  return `<div class="objective-chart"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="Trial objective trend">
    <defs><linearGradient id="${gid}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".34"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>
    ${grid}
    <polygon points="${areaPoints}" fill="url(#${gid})"/>
    <polyline points="${points}" fill="none" stroke="${color}" stroke-width="3.6" vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${last.x}" cy="${last.y}" r="6" fill="${color}" stroke="#0d151d" stroke-width="3" vector-effect="non-scaling-stroke"/>
  </svg></div>`;
}

// -------------------- router --------------------

const routes = [
  [/^#\/$/, "overview", renderOverviewScreen],
  [/^#\/projects$/, "projects", renderProjectsScreen],
  [/^#\/runs$/, "runs", () => renderGlobalRunsScreen(new URLSearchParams(location.hash.split("?")[1]))],
  [/^#\/failures$/, "failures", renderFailuresScreen],
  [/^#\/campaigns$/, "campaigns", renderCampaignsScreen],
  [/^#\/memory$/, "memory", renderMemoryScreen],
  [/^#\/settings$/, "settings", renderSettingsScreen],
  [/^#\/project\/([^/]+)$/, "projects", (m) => renderProjectRunsScreen(decodeURIComponent(m[1]))],
  [/^#\/run\/([^/]+)$/, "runs", (m) => renderRunScreen(decodeURIComponent(m[1]))],
  [/^#\/failure\/([^/]+)$/, "failures", (m) => renderFailureScreen(decodeURIComponent(m[1]))],
  [/^#\/campaign\/([^/]+)$/, "campaigns", (m) => renderCampaignScreen(decodeURIComponent(m[1]))],
  [/^#\/compare$/, "runs", () => renderCompareScreen(new URLSearchParams(location.hash.split("?")[1]))],
];

function route() {
  startRouteProgress();
  const hash = location.hash || "#/";
  const path = hash.split("?")[0];
  for (const [pattern, navKey, handler] of routes) {
    const m = path.match(pattern);
    if (m) {
      updateActiveNav(navKey);
      Promise.resolve(handler(m)).finally(() => {
        finishRouteProgress();
        app.focus({ preventScroll: true });
        window.scrollTo({ top: 0, behavior: "instant" });
      });
      return;
    }
  }
  app.innerHTML = `<div class="empty-state"><p class="eyebrow">404</p><p>Unknown view.</p></div>`;
  finishRouteProgress();
}

function updateActiveNav(navKey) {
  document.querySelectorAll("#sidebar-nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === navKey);
  });
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  setupGlobalUI();
  route();
});

// -------------------- screen: overview --------------------

async function renderOverviewScreen() {
  app.innerHTML = `<p class="loading">loading overview&hellip;</p>`;
  let ov;
  try {
    ov = await api("/overview");
  } catch (e) { app.innerHTML = errorState(e); return; }

  if (ov.run_count === 0) {
    app.innerHTML = `<div class="empty-state">
      <p class="eyebrow">no runs yet</p>
      <p>Start one from Python: <code>with watcher.init(project="...", config={...}) as run:</code></p>
    </div>`;
    return;
  }

  const attention = ov.runs_needing_attention.map((r) => `
    <tr>
      <td><a href="#/run/${encodeURIComponent(r.run_id)}">${esc(r.display_name)}</a></td>
      <td>${esc(r.project)}</td>
      <td>${esc(r.failure_category)}</td>
      <td><a href="#/failure/${encodeURIComponent(r.run_id)}">investigate &rarr;</a></td>
    </tr>`).join("");

  const verified = ov.recent_verified_fixes.map((c) => `
    <tr>
      <td><a href="#/campaign/${encodeURIComponent(c.campaign_id)}">${esc(c.campaign_id)}</a></td>
      <td>${esc(c.project)}</td>
      <td><a href="#/run/${encodeURIComponent(c.best_run_id)}">${esc(c.best_run_id)}</a></td>
    </tr>`).join("");

  app.innerHTML = `
    <h1 class="page-title">Overview</h1>
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Projects</div><div class="stat-value">${ov.project_count}</div></div>
      <div class="stat-card"><div class="stat-label">Total runs</div><div class="stat-value">${ov.run_count}</div></div>
      <div class="stat-card"><div class="stat-label">Runs needing attention</div><div class="stat-value ${ov.runs_needing_attention.length ? 'red' : ''}">${ov.runs_needing_attention.length}</div></div>
      <div class="stat-card"><div class="stat-label">Active campaigns</div><div class="stat-value mint">${ov.active_campaign_count}</div></div>
      <div class="stat-card"><div class="stat-label">GPU</div><div class="stat-value" style="font-size:15px;">${ov.gpu_available ? esc(ov.gpu_name) : "not detected"}</div></div>
      <div class="stat-card"><div class="stat-label">Ollama</div><div class="stat-value ${ov.ollama_available ? 'mint' : ''}" style="font-size:15px;">${ov.ollama_available ? "available" : "not running"}</div></div>
    </div>

    <div class="panel">
      <h2 class="section-title">Runs needing attention</h2>
      ${attention ? `<table class="runs-table"><thead><tr><th>run</th><th>project</th><th>failure</th><th></th></tr></thead><tbody>${attention}</tbody></table>`
        : `<p class="ai-empty">Nothing unresolved right now.</p>`}
    </div>

    <div class="panel">
      <h2 class="section-title">Recent verified fixes</h2>
      ${verified ? `<table class="runs-table"><thead><tr><th>campaign</th><th>project</th><th>best run</th></tr></thead><tbody>${verified}</tbody></table>`
        : `<p class="ai-empty">No recovery campaigns have produced a verified fix yet.</p>`}
    </div>
  `;
}

// -------------------- screen: projects --------------------

async function renderProjectsScreen() {
  app.innerHTML = `<p class="loading">loading projects&hellip;</p>`;
  let projects;
  try { projects = await api("/projects"); } catch (e) { app.innerHTML = errorState(e); return; }
  if (!projects.length) {
    app.innerHTML = `<div class="empty-state"><p class="eyebrow">no runs yet</p></div>`;
    return;
  }
  app.innerHTML = `
    <p class="eyebrow">watcherml / projects</p>
    <h1 class="page-title">Projects</h1>
    <div class="card-grid">
      ${projects.map((p) => `
        <a class="card" href="#/project/${encodeURIComponent(p.name)}">
          <div class="card-title">${esc(p.name)}</div>
          <div class="card-meta">
            <span>${p.run_count} run${p.run_count === 1 ? "" : "s"}</span>
            <span class="${p.failure_count ? "fail-count" : ""}">${p.failure_count} failed</span>
          </div>
        </a>
      `).join("")}
    </div>
  `;
}

// -------------------- screen: project run list --------------------

async function renderProjectRunsScreen(project) {
  app.innerHTML = `<p class="loading">loading runs&hellip;</p>`;
  let runs;
  try { runs = await api(`/projects/${encodeURIComponent(project)}/runs`); }
  catch (e) { app.innerHTML = errorState(e); return; }
  renderRunsTable(runs, `<a href="#/">projects</a> / ${esc(project)}`, project);
}

async function renderGlobalRunsScreen(params) {
  app.innerHTML = `<p class="loading">loading runs&hellip;</p>`;
  const status = params.get("status") || "";
  let query = "";
  if (status) query = `?status=${encodeURIComponent(status)}`;
  let runs;
  try { runs = await api(`/runs${query}`); } catch (e) { app.innerHTML = errorState(e); return; }
  renderRunsTable(runs, "watcherml", null, status);
}

function renderRunsTable(runs, crumbHtml, project, currentStatusFilter) {
  const metricNames = [...new Set(runs.flatMap((r) => Object.keys(r.final_metrics || {})))].slice(0, 2);
  const filterBar = project ? "" : `
    <div class="filter-bar">
      <span>filter:</span>
      <a href="#/runs" style="${!currentStatusFilter ? 'color:var(--signal-mint)' : ''}">all</a>
      <a href="#/runs?status=failed" style="${currentStatusFilter === 'failed' ? 'color:var(--signal-mint)' : ''}">failed</a>
      <a href="#/runs?status=success" style="${currentStatusFilter === 'success' ? 'color:var(--signal-mint)' : ''}">success</a>
    </div>`;

  app.innerHTML = `
    <p class="eyebrow">${crumbHtml}</p>
    <h1 class="page-title">${project ? esc(project) : "Runs"}</h1>
    ${filterBar}
    ${project ? `<div class="compare-picker"><span>select two runs to compare:</span><button id="compare-btn" disabled>Compare selected</button></div>` : ""}
    <div class="panel">
      <table class="runs-table">
        <thead><tr>
          ${project ? "<th></th>" : ""}<th>run</th><th>status</th><th>started</th><th>duration</th>
          ${metricNames.map((m) => `<th>${esc(m)}</th>`).join("")}
          <th>hardware</th><th>warnings</th>
        </tr></thead>
        <tbody>
          ${runs.map((r) => `
            <tr>
              ${project ? `<td><input type="checkbox" class="compare-check" value="${esc(r.run_id)}" /></td>` : ""}
              <td>
                <a href="#/run/${encodeURIComponent(r.run_id)}">
                  <span class="run-name">${esc(r.display_name)}</span>
                  <span class="run-id-sub">${esc(r.run_id)}</span>
                </a>
                ${demoBadge(r.simulated)} ${resolvedBadge(r.resolved)}
              </td>
              <td>${badge(r.status)}</td>
              <td>${fmtTimestamp(r.started_at)}</td>
              <td>${fmtDuration(r.duration_seconds)}</td>
              ${metricNames.map((m) => `<td>${fmtNum(r.final_metrics[m])}</td>`).join("")}
              <td>${esc(r.hardware)}</td>
              <td>${r.warning_count > 0 ? `<span style="color:var(--signal-amber)">${r.warning_count}</span>` : "0"}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;

  if (project) {
    const checks = [...document.querySelectorAll(".compare-check")];
    const compareBtn = document.getElementById("compare-btn");
    checks.forEach((c) => c.addEventListener("change", () => {
      const checked = checks.filter((x) => x.checked);
      if (checked.length > 2) { c.checked = false; return; }
      compareBtn.disabled = checked.length !== 2;
    }));
    compareBtn.addEventListener("click", () => {
      const [a, b] = checks.filter((c) => c.checked).map((c) => c.value);
      location.hash = `#/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`;
    });
  }
}

// -------------------- screen: run detail --------------------

async function renderRunScreen(runId) {
  app.innerHTML = `<p class="loading">loading run&hellip;</p>`;
  let run, samples;
  try {
    [run, samples] = await Promise.all([
      api(`/runs/${encodeURIComponent(runId)}`),
      api(`/runs/${encodeURIComponent(runId)}/samples`),
    ]);
  } catch (e) { app.innerHTML = errorState(e); return; }

  const metricRows = Object.entries(run.final_metrics || {})
    .map(([k, v]) => `<div class="field"><span class="field-label">${esc(k)}</span><span class="field-value">${fmtNum(v)}</span></div>`)
    .join("") || `<p class="ai-empty">no metrics logged</p>`;

  const configRows = Object.entries(run.config || {})
    .filter(([k]) => k !== "_simulated")
    .map(([k, v]) => `<div class="field"><span class="field-label">${esc(k)}</span><span class="field-value">${esc(v)}</span></div>`)
    .join("") || `<p class="ai-empty">no config recorded</p>`;

  const failureLink = run.has_failure
    ? `<div class="failure-banner"><p class="eyebrow">this run failed</p>
        <a href="#/failure/${encodeURIComponent(runId)}"><strong>View failure capsule &rarr;</strong></a></div>`
    : "";

  const warningsHtml = (run.warnings || []).length
    ? `<div class="panel"><h2 class="section-title">Warnings</h2>
        <ul class="warning-list">${run.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>` : "";

  const timelineItems = buildTimeline(run);

  app.innerHTML = `
    <p class="eyebrow"><a href="#/">projects</a> / <a href="#/project/${encodeURIComponent(run.project)}">${esc(run.project)}</a> / run</p>
    <h1 class="page-title">
      <span id="run-title">${esc(run.display_name)}</span>
      ${badge(run.status)} ${demoBadge(run.simulated)} ${resolvedBadge(run.resolved)}
    </h1>
    <p class="page-subtitle">${esc(run.run_id)}</p>
    <div style="margin-bottom:16px;">${tagPills(run.tags)}</div>

    <div class="action-bar">
      <button id="rename-btn">Rename</button>
      <a href="/api/runs/${encodeURIComponent(runId)}/export"><button>Export capsule</button></a>
      <button id="resolve-btn">${run.resolved ? "Mark unresolved" : "Mark as resolved"}</button>
    </div>

    ${failureLink}
    ${renderTrace(samples, "system telemetry")}

    <div class="panel-grid">
      <div class="panel"><h2 class="section-title">Metrics ${provenance("calculated")}</h2>${metricRows}</div>
      <div class="panel"><h2 class="section-title">Config</h2>${configRows}</div>
      <div class="panel">
        <h2 class="section-title">Reproduction</h2>
        <div class="field"><span class="field-label">duration</span><span class="field-value">${fmtDuration(run.duration_seconds)}</span></div>
        <div class="field"><span class="field-label">git_state</span><span class="field-value">${run.git && run.git.available ? (run.git.dirty ? "dirty" : "clean") : "no_git_repo"}</span></div>
        <div class="field"><span class="field-label">dataset_fingerprint</span><span class="field-value">${esc(run.dataset_fingerprint)}</span></div>
        <div class="field"><span class="field-label">reproduction_score</span><span class="field-value">${run.reproduction_score !== null && run.reproduction_score !== undefined ? run.reproduction_score + "/10" : "&mdash;"}</span></div>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">Timeline</h2>
      <div class="timeline">${timelineItems}</div>
    </div>

    ${warningsHtml}
  `;

  document.getElementById("rename-btn").addEventListener("click", async () => {
    const next = prompt("Rename this run:", run.display_name === run.run_id ? "" : run.display_name);
    if (next === null) return;
    await api(`/runs/${encodeURIComponent(runId)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: next || null }),
    });
    notify("Run name updated");
    renderRunScreen(runId);
  });
  document.getElementById("resolve-btn").addEventListener("click", async () => {
    let note = null;
    if (!run.resolved) note = prompt("Resolution note (optional):", "") || null;
    await api(`/runs/${encodeURIComponent(runId)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolved: !run.resolved, resolved_note: note }),
    });
    notify(run.resolved ? "Run marked unresolved" : "Run marked resolved");
    renderRunScreen(runId);
  });
}

function buildTimeline(run) {
  const items = [{ t: run.started_at, label: "Run started", cls: "" }];
  for (const w of (run.warnings || [])) {
    items.push({ t: run.started_at, label: `Warning: ${w}`, cls: "warning" });
  }
  if (run.has_failure) {
    items.push({ t: run.ended_at, label: "Run failed", cls: "failure" });
  } else if (run.ended_at) {
    items.push({ t: run.ended_at, label: "Run completed", cls: "" });
  }
  items.sort((a, b) => (a.t || 0) - (b.t || 0));
  return items.map((i) => `
    <div class="timeline-item ${i.cls}">
      <div class="timeline-time">${fmtTimestamp(i.t)}</div>
      <div class="timeline-label">${esc(i.label)}</div>
    </div>`).join("") || `<p class="ai-empty">No timeline events recorded.</p>`;
}

// -------------------- screen: failure capsule --------------------

async function renderFailureScreen(runId) {
  app.innerHTML = `<p class="loading">loading failure capsule&hellip;</p>`;
  let failure;
  try { failure = await api(`/runs/${encodeURIComponent(runId)}/failure`); }
  catch (e) { app.innerHTML = errorState(e); return; }

  const d = failure.diagnosis || {};
  const actions = (d.suggested_actions || []).map((a) => `<li>${esc(a)}</li>`).join("");
  const evidenceIndex = {};
  (failure.evidence_index || []).forEach((e) => { evidenceIndex[e.id] = e.label; });
  const evidenceChips = (d.evidence_ids || []).map((id) =>
    `<span class="evidence-chip" title="${esc(evidenceIndex[id] || '')}">${esc(id)}</span>`).join("");

  const recentMetrics = (failure.evidence?.recent_metrics || [])
    .map((m) => `<div class="field"><span class="field-label">${esc(m.name)} (step ${m.step})</span><span class="field-value">${fmtNum(m.value)}</span></div>`)
    .join("") || `<p class="ai-empty">no metrics logged before failure</p>`;

  const similar = (failure.similar_previous_failures || [])
    .map((s) => `<li><a href="#/failure/${encodeURIComponent(s.run_id)}">${esc(s.run_id)}</a> &mdash; ${esc(s.message.slice(0, 80))}</li>`).join("");

  const comp = failure.comparison_to_last_success;
  let comparisonHtml = `<span class="ai-empty">no previous successful run to compare against</span>`;
  if (comp) {
    const checklist = (comp.checklist || []).map((c) => `
      <div class="checklist-row"><span class="${c.matched ? 'check-yes' : 'check-no'}">${c.matched ? '&check;' : '&times;'}</span> ${esc(c.label)}</div>
    `).join("");
    comparisonHtml = `
      <p>Nearest successful run: <a href="#/run/${encodeURIComponent(comp.run_id)}">${esc(comp.run_id)}</a>
         &mdash; similarity ${(comp.similarity_score * 100).toFixed(0)}%</p>
      ${checklist}
      <a href="#/compare?a=${encodeURIComponent(runId)}&b=${encodeURIComponent(comp.run_id)}">Full comparison &rarr;</a>
    `;
  }

  app.innerHTML = `
    <p class="eyebrow"><a href="#/run/${encodeURIComponent(runId)}">${esc(failure.display_name)}</a> / failure</p>
    <div class="failure-banner">
      <p class="eyebrow">${esc(d.rule || "unclassified")} ${demoBadge(failure.simulated)} ${resolvedBadge(failure.resolved)}</p>
      <h1 class="page-title" style="margin-bottom:6px;">${esc(failure.exception_type)}</h1>
      <p class="failure-message">${esc(failure.message)}</p>
    </div>

    <div class="action-bar">
      <button id="analyze-btn">Analyze locally</button>
      ${comp ? `<a href="#/compare?a=${encodeURIComponent(runId)}&b=${encodeURIComponent(comp.run_id)}"><button>Compare baseline</button></a>` : ""}
      <button id="campaign-btn">Create recovery campaign</button>
      <a href="/api/runs/${encodeURIComponent(runId)}/export"><button>Export capsule</button></a>
      <button id="resolve-btn">${failure.resolved ? "Mark unresolved" : "Mark as resolved"}</button>
    </div>

    <div class="panel-grid">
      <div class="panel">
        <h2 class="section-title">Diagnosis ${provenance("rule-based")}</h2>
        <p>${esc(d.summary)}</p>
        ${d.likely_cause ? `<p><strong>Likely cause:</strong> ${esc(d.likely_cause)}</p>` : ""}
        ${evidenceChips ? `<div class="evidence-row">${evidenceChips}</div>` : ""}
        ${actions ? `<ul class="suggested-actions">${actions}</ul>` : ""}
        <div class="ai-panel">
          <p class="eyebrow"><span>AI explanation</span>${provenance("ollama")}</p>
          <button id="advise-btn">Get AI explanation</button>
          <div id="advise-result"></div>
        </div>
      </div>
      <div class="panel">
        <h2 class="section-title">Recent metrics before failure</h2>
        ${recentMetrics}
        <h2 class="section-title" style="margin-top:16px;">Similar previous failures</h2>
        ${similar ? `<ul class="suggested-actions">${similar}</ul>` : `<p class="ai-empty">none recorded</p>`}
        <h2 class="section-title" style="margin-top:16px;">Comparison ${provenance("calculated")}</h2>
        ${comparisonHtml}
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">Full traceback</h2>
      <div class="traceback">${esc(failure.traceback)}</div>
    </div>
  `;

  document.getElementById("advise-btn").addEventListener("click", async (ev) => {
    ev.target.disabled = true;
    ev.target.textContent = "Asking Ollama\u2026";
    const resultEl = document.getElementById("advise-result");
    try {
      const res = await api(`/runs/${encodeURIComponent(runId)}/advise`, { method: "POST" });
      resultEl.innerHTML = !res.available
        ? `<p class="ai-empty">Ollama isn't running -- everything above this line is deterministic and didn't need it.</p>`
        : `<p class="ai-text">${esc(res.text || "No response from the model.")}</p>`;
    } catch (e) { resultEl.innerHTML = `<p class="ai-empty">${esc(e.message)}</p>`; }
    ev.target.textContent = "Get AI explanation";
    ev.target.disabled = false;
  });
  document.getElementById("analyze-btn").addEventListener("click", () => {
    document.getElementById("advise-btn").click();
  });
  document.getElementById("campaign-btn").addEventListener("click", () => {
    alert("Recovery campaigns are launched from Python or the CLI, since they need your " +
          "training function:\n\nwatcher.recover_from_oom(\n  project=\"" + failure.run_id.split("-")[0] + "\",\n" +
          "  failed_run_id=\"" + runId + "\",\n  train_fn=train,\n)\n\nOnce it runs, it'll show up on the Campaigns page.");
  });
  document.getElementById("resolve-btn").addEventListener("click", async () => {
    let note = null;
    if (!failure.resolved) note = prompt("Resolution note (optional):", "") || null;
    await api(`/runs/${encodeURIComponent(runId)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolved: !failure.resolved, resolved_note: note }),
    });
    notify(failure.resolved ? "Failure marked unresolved" : "Failure marked resolved");
    renderFailureScreen(runId);
  });
}

// -------------------- screen: compare --------------------

async function renderCompareScreen(params) {
  const a = params.get("a"), b = params.get("b");
  if (!a || !b) {
    app.innerHTML = `<div class="empty-state"><p class="eyebrow">missing runs</p><p>Select two runs from a project page to compare.</p></div>`;
    return;
  }
  app.innerHTML = `<p class="loading">comparing runs&hellip;</p>`;
  let diff;
  try { diff = await api(`/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`); }
  catch (e) { app.innerHTML = errorState(e); return; }

  const configRows = diff.config_diff.map((c) => diffRow(c.key, c.from, c.to)).join("") || `<p class="ai-empty">no config changes</p>`;
  const metricRows = diff.metric_diff.map((m) => {
    const deltaCls = m.delta === undefined ? "" : m.delta >= 0 ? "diff-delta-up" : "diff-delta-down";
    const deltaStr = m.delta !== undefined ? ` (${m.delta >= 0 ? "+" : ""}${fmtNum(m.delta)})` : "";
    return `<div class="diff-row"><span class="diff-key">${esc(m.metric)}</span><span class="diff-from">${fmtNum(m.from)}</span>
      <span class="diff-arrow">&rarr;</span><span class="diff-to ${deltaCls}">${fmtNum(m.to)}<span class="${deltaCls}">${esc(deltaStr)}</span></span></div>`;
  }).join("");
  const pkgRows = diff.package_diff.slice(0, 15).map((p) => diffRow(p.package, p.from, p.to)).join("");
  const pkgSection = diff.package_diff.length
    ? `<h2 class="section-title" style="margin-top:16px;">Package changes (${diff.package_diff.length})</h2>${pkgRows}` : "";

  app.innerHTML = `
    <p class="eyebrow">compare</p>
    <h1 class="page-title">${esc(a)} &rarr; ${esc(b)}</h1>
    <div class="panel">
      <h2 class="section-title">What changed?</h2>
      ${configRows}
      ${diff.dataset_changed ? `<div class="diff-row"><span class="diff-key">dataset</span><span class="diff-from" style="grid-column: span 3; text-align:left; color: var(--signal-amber);">fingerprint changed</span></div>` : ""}
      ${diff.git_diff.commit_changed ? diffRow("git commit", (diff.git_diff.commit_a || "").slice(0, 10), (diff.git_diff.commit_b || "").slice(0, 10)) : ""}
      ${pkgSection}
    </div>
    <div class="panel">
      <h2 class="section-title">What changed in results?</h2>
      ${metricRows || `<p class="ai-empty">no metrics to compare</p>`}
      <div class="diff-row"><span class="diff-key">exit status</span><span class="diff-from">${badge(diff.exit_status_a)}</span>
        <span class="diff-arrow">&rarr;</span><span class="diff-to">${badge(diff.exit_status_b)}</span></div>
    </div>
    <div class="panel">
      <div class="ai-panel">
        <p class="eyebrow"><span>Likely explanation</span>${provenance("ollama")}</p>
        <button id="advise-btn">Get AI explanation</button>
        <div id="advise-result"></div>
      </div>
    </div>
  `;
  document.getElementById("advise-btn").addEventListener("click", async (ev) => {
    ev.target.disabled = true;
    ev.target.textContent = "Asking Ollama\u2026";
    const resultEl = document.getElementById("advise-result");
    try {
      const res = await api(`/compare/advise?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`, { method: "POST" });
      resultEl.innerHTML = !res.available
        ? `<p class="ai-empty">Ollama isn't running -- the deterministic diff above didn't need it.</p>`
        : `<p class="ai-text">${esc(res.text || "No response from the model.")}</p>`;
    } catch (e) { resultEl.innerHTML = `<p class="ai-empty">${esc(e.message)}</p>`; }
    ev.target.textContent = "Get AI explanation";
    ev.target.disabled = false;
  });
}

function diffRow(key, from, to) {
  return `<div class="diff-row"><span class="diff-key">${esc(key)}</span><span class="diff-from">${esc(from)}</span>
    <span class="diff-arrow">&rarr;</span><span class="diff-to">${esc(to)}</span></div>`;
}

// -------------------- screen: global failures --------------------

async function renderFailuresScreen() {
  app.innerHTML = `<p class="loading">loading failures&hellip;</p>`;
  let failures;
  try { failures = await api("/failures"); } catch (e) { app.innerHTML = errorState(e); return; }
  if (!failures.length) {
    app.innerHTML = `<div class="empty-state"><p class="eyebrow">no failures recorded</p><p>Every run so far has completed successfully.</p></div>`;
    return;
  }
  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Failures</h1>
    <div class="panel">
      <table class="runs-table">
        <thead><tr><th>run</th><th>project</th><th>rule</th><th>message</th></tr></thead>
        <tbody>
          ${failures.map((f) => `
            <tr>
              <td><a href="#/failure/${encodeURIComponent(f.run_id)}">${esc(f.run_id)}</a></td>
              <td>${esc(f.project)}</td>
              <td>${esc(f.rule)}</td>
              <td>${esc((f.message || "").slice(0, 70))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

// -------------------- screen: campaigns --------------------

async function renderCampaignsScreen() {
  app.innerHTML = `<p class="loading">loading campaigns&hellip;</p>`;
  let campaigns;
  try { campaigns = await api("/campaigns"); } catch (e) { app.innerHTML = errorState(e); return; }
  if (!campaigns.length) {
    app.innerHTML = `<div class="empty-state">
      <p class="eyebrow">no recovery campaigns yet</p>
      <p>Launch one from Python: <code>watcher.recover_from_oom(project=..., failed_run_id=..., train_fn=...)</code></p>
    </div>`;
    return;
  }
  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Campaigns</h1>
    <div class="card-grid">
      ${campaigns.map((c) => `
        <a class="card" href="#/campaign/${encodeURIComponent(c.campaign_id)}">
          <div class="card-title">${esc(c.campaign_id)}</div>
          <div class="card-meta">
            <span>${esc(c.project)}</span>
            <span class="${c.status === 'active' ? '' : 'fail-count'}">${esc(c.status)}</span>
            <span>${c.trial_count} trial${c.trial_count === 1 ? '' : 's'}</span>
          </div>
        </a>
      `).join("")}
    </div>
  `;
}

async function renderCampaignScreen(campaignId) {
  app.innerHTML = `<p class="loading">loading campaign&hellip;</p>`;
  let c;
  try { c = await api(`/campaigns/${encodeURIComponent(campaignId)}`); }
  catch (e) { app.innerHTML = errorState(e); return; }

  const trials = Array.isArray(c.trials) ? c.trials : [];
  const bestTrial = trials.find((t) => t.run_id === c.best_run_id) || trials.filter((t) => t.outcome === "success").at(-1) || null;
  const bestIndex = bestTrial ? trials.indexOf(bestTrial) : -1;
  const contract = c.contract || {};
  const scores = trials.map((t) => firstFinite(t.score, t.objective_value)).filter((v) => v !== null);
  const baselineScore = firstFinite(c.baseline_score, contract.baseline_score, scores[0]);
  const finalScore = scores.length ? scores[scores.length - 1] : null;
  const objectiveDirection = contract.direction || "maximize";
  const rawImprovement = baselineScore !== null && finalScore !== null && baselineScore !== 0
    ? ((finalScore - baselineScore) / Math.abs(baselineScore)) * 100
    : null;
  const displayImprovement = rawImprovement === null ? null : (objectiveDirection === "minimize" ? -rawImprovement : rawImprovement);

  const peakVram = firstFinite(
    bestTrial?.peak_vram_gb, bestTrial?.peak_vram,
    c.peak_vram_gb, c.peak_vram,
    contract.peak_vram_gb, contract.max_vram_gb,
  );
  const gpuSeconds = firstFinite(c.gpu_seconds_used, c.gpu_time_seconds, c.total_gpu_seconds, contract.gpu_seconds_used);
  const gpuBudgetSeconds = firstFinite(contract.max_gpu_seconds, contract.gpu_budget_seconds,
    contract.max_gpu_hours !== undefined ? Number(contract.max_gpu_hours) * 3600 : null);
  const budgetPct = gpuBudgetSeconds && gpuSeconds !== null ? Math.min(100, (gpuSeconds / gpuBudgetSeconds) * 100) : null;
  const recovered = Boolean(c.best_run_id || c.status === "recovered" || (bestTrial && bestTrial.outcome === "success"));
  const isActive = !c.ended_at && c.status !== "stopped" && c.status !== "completed" && c.status !== "recovered";
  const statusLabel = recovered ? "Recovered" : isActive ? "Agent active" : esc(c.status || "Stopped");

  const reasoningSteps = [
    {
      text: `Captured the baseline failure signature from <strong>${esc(c.source_run_id || "source run")}</strong>.`,
      meta: "deterministic forensic capture",
      active: false,
    },
  ];
  trials.filter((t) => t.phase === "probe").slice(-2).forEach((t) => {
    reasoningSteps.push({
      text: t.outcome === "success"
        ? `Probe survived after <strong>${esc(formatPatch(t.patch))}</strong>.`
        : `Eliminated <strong>${esc(formatPatch(t.patch))}</strong> after ${esc(t.outcome || "failure")}.`,
      meta: t.outcome === "success" ? "candidate retained" : "candidate rejected",
      active: false,
    });
  });
  if (bestTrial) {
    reasoningSteps.push({
      text: `<strong>${recovered ? "Campaign objective satisfied." : "Best candidate selected."}</strong> ${esc(formatPatch(bestTrial.patch))}`,
      meta: "verified against the campaign contract",
      active: true,
    });
  } else if (isActive) {
    reasoningSteps.push({ text: "Waiting for the next bounded trial result.", meta: "agent loop in progress", active: true });
  }

  const trialRows = trials.map((t, i) => {
    let decision = "rejected";
    if (t.run_id === c.best_run_id || i === bestIndex) decision = "best";
    else if (t.phase === "probe" && t.outcome === "success") decision = "keep";
    else if (t.phase === "full" && t.outcome === "success") decision = "accept";
    const score = firstFinite(t.score, t.objective_value);
    const resultText = t.result_summary || (t.outcome === "success"
      ? (score !== null ? `${esc(contract.goal_metric || "objective")} ${fmtNum(score, 3)}` : "Trial completed")
      : esc(t.outcome || "failed"));
    return `
      <tr>
        <td>#${String(i + 1).padStart(2, "0")}</td>
        <td class="trial-intervention">${esc(formatPatch(t.patch))}</td>
        <td class="trial-result ${t.outcome === "success" ? "good" : "bad"}">${resultText}</td>
        <td><span class="decision-pill ${decision}">${decision}</span></td>
      </tr>`;
  }).join("");

  const maxTrials = firstFinite(contract.max_trials, contract.trial_budget, c.max_trials);
  const objective = contract.goal_metric || contract.metric || "success + VRAM headroom";
  const target = contract.target ?? contract.goal ?? contract.threshold ?? "bounded recovery";
  const permissions = contract.permissions || {};
  const autoChanges = Object.entries(permissions).filter(([, v]) => v === "automatic" || v === true).map(([k]) => k.replaceAll("_", " "));
  const stoppedReason = c.stopped_reason || (isActive ? "Campaign is still running within its configured guardrails." : recovered ? "Acceptance criteria satisfied." : "Campaign stopped.");

  app.innerHTML = `
    <div class="campaign-workspace">
      <div class="campaign-windowbar">
        <div class="window-dots" aria-hidden="true"><i></i><i></i><i></i></div>
        <div class="window-title">campaign / ${esc(campaignId)}</div>
        <div class="agent-state ${isActive ? "" : "stopped"}">${isActive ? "agent active" : recovered ? "objective satisfied" : "campaign stopped"}</div>
      </div>

      <div class="campaign-body">
        <section class="campaign-hero">
          <p class="eyebrow autopilot-label">autopilot campaign</p>
          <h1 class="page-title">Recover &amp; Optimize</h1>
          <p class="page-subtitle">Bounded, evidence-backed recovery for ${esc(c.project || "this experiment")}</p>
          <div class="campaign-actions">
            ${c.source_run_id ? `<a href="#/run/${encodeURIComponent(c.source_run_id)}"><button class="ghost">View source run</button></a>` : ""}
            <button class="ghost icon-button" id="refresh-campaign"><svg viewBox="0 0 24 24"><path d="M18.5 5.5A9 9 0 1 0 21 12h-2a7 7 0 1 1-2.05-4.95L14 10h7V3l-2.5 2.5Z"/></svg>Refresh</button>
            <button class="ghost" data-copy="${esc(campaignId)}" data-copy-label="Campaign ID copied">Copy campaign ID</button>
          </div>


        $

        <section class="campaign-stat-strip" aria-label="Campaign summary">
          <div class="campaign-stat"><div class="campaign-stat-label">Best trial</div><div class="campaign-stat-value">${bestIndex >= 0 ? `#${String(bestIndex + 1).padStart(2, "0")}` : "—"}</div></div>
          <div class="campaign-stat"><div class="campaign-stat-label">Peak VRAM</div><div class="campaign-stat-value">${peakVram !== null ? `${fmtNum(peakVram, 1)} GB` : "—"}</div></div>
          <div class="campaign-stat"><div class="campaign-stat-label">GPU budget used</div><div class="campaign-stat-value">${formatGpuTime(gpuSeconds)}</div></div>
          <div class="campaign-stat"><div class="campaign-stat-label">Status</div><div class="campaign-stat-value mint">${statusLabel}</div></div>
        </section>

        <section class="campaign-primary-grid">
          <article class="campaign-panel">
            <header class="campaign-panel-header"><span>Agent reasoning</span><span class="live-label">${isActive ? "live" : "verified"}</span></header>
            <div class="campaign-panel-body">
              ${reasoningSteps.slice(-4).map((step, i) => `
                <div class="reasoning-step campaign-reasoning ${step.active ? "active" : ""}">
                  <span class="reasoning-num">${String(i + 1).padStart(2, "0")}</span>
                  <span class="reasoning-text">${step.text}<span class="reasoning-meta">${esc(step.meta)}</span></span>
                </div>`).join("")}
            </div>
          </article>

          <article class="campaign-panel">
            <header class="campaign-panel-header"><span>Trial objective</span><span class="objective-change">${displayImprovement !== null ? `${displayImprovement >= 0 ? "↑" : "↓"} ${Math.abs(displayImprovement).toFixed(1)}%` : provenance("calculated")}</span></header>
            <div class="objective-panel-body">
              ${renderSparkline(scores)}
              <div class="chart-axis-labels"><span>baseline</span><span>${trials.length ? `trial ${trials.length}` : "current"}</span></div>
            </div>
          </article>
        </section>

        <section class="campaign-trials-panel">
          <div class="runs-table-wrapper">
            <table class="runs-table">
              <thead><tr><th>trial</th><th>intervention</th><th>result</th><th>decision</th></tr></thead>
              <tbody>${trialRows || `<tr><td colspan="4" class="text-muted">No trials have been recorded yet.</td></tr>`}</tbody>
            </table>
          </div>
        </section>

        <section class="campaign-support-grid">
          <article class="panel m0">
            <h2 class="section-title">Campaign contract ${provenance("calculated")}</h2>
            <div class="contract-grid">
              <div class="contract-item"><div class="field-label">objective</div><div class="contract-value">${esc(objective)}</div></div>
              <div class="contract-item"><div class="field-label">acceptance target</div><div class="contract-value">${esc(target)}</div></div>
              <div class="contract-item"><div class="field-label">trial budget</div><div class="contract-value">${maxTrials !== null ? `${trials.length} / ${maxTrials} trials` : `${trials.length} trials used`}</div></div>
              <div class="contract-item"><div class="field-label">automatic changes</div><div class="contract-value">${esc(autoChanges.join(", ") || "policy controlled")}</div></div>
            </div>
            ${budgetPct !== null ? `<div class="budget-track" title="${budgetPct.toFixed(0)}% of GPU budget used"><div class="budget-fill" style="width:${budgetPct.toFixed(1)}%"></div></div>` : ""}
          </article>

          <article class="panel m0">
            <h2 class="section-title">Guardrails</h2>
            <div class="guardrail-list">
              <div class="guardrail-item"><span class="guardrail-icon">01</span><span>Every intervention is recorded as a child trial of the original failure.</span></div>
              <div class="guardrail-item"><span class="guardrail-icon">02</span><span>Trial count, GPU time, and objective thresholds stop unbounded loops.</span></div>
              <div class="guardrail-item"><span class="guardrail-icon">03</span><span>Outcomes are evaluated deterministically before a fix is marked verified.</span></div>
            </div>
            <p class="page-subtitle mt16 m0"><strong>Stopped because:</strong> ${esc(stoppedReason)}</p>
          </article>
        </section>
      </div>
    </div>
  `;

  document.getElementById("refresh-campaign")?.addEventListener("click", () => {
    notify("Campaign refreshed");
    renderCampaignScreen(campaignId);
  });
}

// -------------------- screen: resolution memory --------------------

async function renderMemoryScreen() {
  app.innerHTML = `<p class="loading">loading resolution memory&hellip;</p>`;
  let signatures;
  try { signatures = await api("/memory"); } catch (e) { app.innerHTML = errorState(e); return; }
  if (!signatures.length) {
    app.innerHTML = `<div class="empty-state">
      <p class="eyebrow">no resolution history yet</p>
      <p>This builds up automatically as recovery campaigns run -- nothing to configure.</p>
    </div>`;
    return;
  }
  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Resolution memory ${provenance("calculated")}</h1>
    <p class="page-subtitle">Built entirely from verified recovery trials -- not a separate tracked concept.</p>
    ${signatures.map((s) => {
      const rate = s.success_rate;
      const rateCls = rate >= 0.7 ? "good" : rate <= 0.3 ? "bad" : "mixed";
      return `
        <div class="signature-card">
          <div class="signature-title">${esc(s.failure_class)} &mdash; changing ${esc(s.patch_keys.join(", ") || "(no keys)")}</div>
          <div class="resolution-row">
            <span>${s.example_patches.map((p) => esc(JSON.stringify(p))).join(" / ")}</span>
            <span class="resolution-rate ${rateCls}">${s.successes}/${s.attempts} successful (${(rate * 100).toFixed(0)}%)</span>
          </div>
        </div>`;
    }).join("")}
  `;
}

// -------------------- screen: settings --------------------

async function renderSettingsScreen() {
  app.innerHTML = `<p class="loading">loading settings&hellip;</p>`;
  let s;
  try { s = await api("/settings"); } catch (e) { app.innerHTML = errorState(e); return; }
  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Settings</h1>
    <div class="panel">
      <h2 class="section-title">Local storage</h2>
      <div class="field"><span class="field-label">data directory</span><span class="field-value">${esc(s.data_directory)}</span></div>
      <div class="field"><span class="field-label">database</span><span class="field-value">${esc(s.database_path)}</span></div>
    </div>
    <div class="panel">
      <h2 class="section-title">Ollama advisor</h2>
      <div class="field"><span class="field-label">status</span><span class="field-value">${s.ollama_available ? '<span class="badge success">available</span>' : '<span class="badge failed">not running</span>'}</span></div>
      <div class="field"><span class="field-label">host</span><span class="field-value">${esc(s.ollama_host)}</span></div>
      <div class="field"><span class="field-label">default model</span><span class="field-value">${esc(s.ollama_default_model)}</span></div>
    </div>
    <div class="panel">
      <h2 class="section-title">GPU</h2>
      <div class="field"><span class="field-label">detected</span><span class="field-value">${s.gpu.available ? "yes" : "no"}</span></div>
      ${(s.gpu.gpus || []).map((g) => `<div class="field"><span class="field-label">${esc(g.name)}</span><span class="field-value">${esc(g.memory_total_mib)} MiB total</span></div>`).join("")}
    </div>
  `;
}

function errorState(e) {
  return `<div class="empty-state"><p class="eyebrow">error</p><p>${esc(e.message)}</p></div>`;
}