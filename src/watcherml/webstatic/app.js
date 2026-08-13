// WatcherML local UI. Vanilla JS, no build step -- keeps the local-mode
// install dependency-free (matches "no Docker, no bundler" for local mode).

const app = document.getElementById("app");
const routeProgress = document.getElementById("route-progress");
let activePollTimer = null;

function stopActivePoll() {
  if (activePollTimer) {
    clearInterval(activePollTimer);
    activePollTimer = null;
  }
}

function notify(message, kind = "success", timeout = 3200) {
  const region = document.getElementById("toast-region");
  if (!region) return;

  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.innerHTML = `
    <span class="toast-symbol">${kind === "error" ? "!" : "✓"}</span>
    <span class="toast-message">${esc(message)}</span>
  `;

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

  window.setTimeout(() => {
    routeProgress.classList.remove("done");
  }, 260);
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

    if (search) {
      search.value = "";
    }

    document
      .querySelectorAll("[data-command-item]")
      .forEach((item) => {
        item.hidden = false;
      });
  };

  const openPalette = () => {
    if (!palette) return;

    palette.hidden = false;
    window.setTimeout(() => search?.focus(), 0);
  };

  trigger?.addEventListener("click", openPalette);

  document
    .querySelectorAll("[data-command-close]")
    .forEach((element) => {
      element.addEventListener("click", closePalette);
    });

  document
    .querySelectorAll("[data-command-item]")
    .forEach((element) => {
      element.addEventListener("click", closePalette);
    });

  search?.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();

    document
      .querySelectorAll("[data-command-item]")
      .forEach((item) => {
        item.hidden =
          Boolean(query) &&
          !item.textContent.toLowerCase().includes(query);
      });
  });

  window.addEventListener("keydown", (event) => {
    if (
      (event.metaKey || event.ctrlKey) &&
      event.key.toLowerCase() === "k"
    ) {
      event.preventDefault();

      if (palette?.hidden) {
        openPalette();
      } else {
        closePalette();
      }
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

  document
    .querySelectorAll("#sidebar-nav a")
    .forEach((link) => {
      link.addEventListener("click", () => {
        sidebar?.classList.remove("open");
        mobileButton?.setAttribute("aria-expanded", "false");
      });
    });

  document.addEventListener("click", async (event) => {
    const copyButton = event.target.closest("[data-copy]");

    if (!copyButton) return;

    try {
      await navigator.clipboard.writeText(
        copyButton.dataset.copy || ""
      );

      notify(
        copyButton.dataset.copyLabel || "Copied to clipboard"
      );
    } catch (_) {
      notify("Clipboard access was unavailable", "error");
    }
  });
}

function formatGpuTime(seconds) {
  if (
    seconds === null ||
    seconds === undefined ||
    Number.isNaN(Number(seconds))
  ) {
    return "—";
  }

  const total = Math.max(0, Number(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);

  return hours
    ? `${hours}h ${String(minutes).padStart(2, "0")}m`
    : `${minutes}m`;
}

function firstFinite(...values) {
  for (const value of values) {
    const numericValue = Number(value);

    if (
      value !== null &&
      value !== undefined &&
      Number.isFinite(numericValue)
    ) {
      return numericValue;
    }
  }

  return null;
}

function formatPatch(patch) {
  if (!patch || typeof patch !== "object") {
    return "No configuration change";
  }

  const entries = Object.entries(patch);

  if (!entries.length) {
    return "No configuration change";
  }

  return entries
    .map(([key, value]) => {
      const label = key.replaceAll("_", " ");

      const renderedValue =
        typeof value === "boolean"
          ? value
            ? "enabled"
            : "disabled"
          : value;

      return `${label} → ${renderedValue}`;
    })
    .join(" · ");
}

async function api(path, options) {
  const response = await fetch("/api" + path, options);

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));

    throw new Error(
      body.detail || `Request failed: ${response.status}`
    );
  }

  return response.json();
}

function esc(value) {
  if (value === null || value === undefined) {
    return "&mdash;";
  }

  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function fmtNum(value, digits = 4) {
  if (value === null || value === undefined) {
    return "&mdash;";
  }

  if (typeof value !== "number") {
    return esc(value);
  }

  if (Number.isInteger(value)) {
    return String(value);
  }

  return value
    .toFixed(digits)
    .replace(/0+$/, "")
    .replace(/\.$/, "");
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) {
    return "&mdash;";
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);

  return `${minutes}m ${remainingSeconds}s`;
}

function fmtTimestamp(timestamp) {
  if (!timestamp) {
    return "&mdash;";
  }

  const date = new Date(timestamp * 1000);

  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function badge(status) {
  const normalized = String(status || "running").toLowerCase();

  const className = [
    "success",
    "verified",
    "completed",
  ].includes(normalized)
    ? "success"
    : [
        "failed",
        "training_failed",
        "integration_error",
        "stopped",
        "not_recovered",
      ].includes(normalized)
      ? "failed"
      : "running";

  return `
    <span class="badge ${className}">
      ${esc(status || "running")}
    </span>
  `;
}

function verificationBadge(status) {
  const normalized = status || "pending";

  const className =
    normalized === "verified"
      ? "success"
      : normalized === "not_verified"
        ? "failed"
        : "running";

  return `
    <span class="badge ${className}">
      ${esc(normalized.replaceAll("_", " "))}
    </span>
  `;
}

function fmtBytes(bytes) {
  if (bytes === null || bytes === undefined) {
    return "&mdash;";
  }

  const numericValue = Number(bytes);

  if (!Number.isFinite(numericValue)) {
    return esc(bytes);
  }

  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = Math.max(0, numericValue);
  let index = 0;

  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }

  return `${size.toFixed(index ? 2 : 0)} ${units[index]}`;
}

function shortDigest(value) {
  if (!value) {
    return "&mdash;";
  }

  const text = String(value);

  return text.length > 18
    ? `${text.slice(0, 12)}…${text.slice(-6)}`
    : esc(text);
}

function jsonBlock(value) {
  return `
    <pre class="traceback"><code>${esc(
      JSON.stringify(value ?? {}, null, 2)
    )}</code></pre>
  `;
}

function demoBadge(simulated) {
  return simulated
    ? `<span class="pill demo">Simulated OOM Scenario</span>`
    : "";
}

function resolvedBadge(resolved) {
  return resolved
    ? `<span class="pill resolved">Resolved</span>`
    : "";
}

function tagPills(tags) {
  if (!tags || !tags.length) {
    return "";
  }

  return tags
    .map((tag) => `<span class="pill tag">${esc(tag)}</span>`)
    .join(" ");
}

function provenance(kind) {
  const labels = {
    "rule-based": "Rule-based",
    calculated: "Calculated",
    deterministic: "Deterministic policy",
    isolated: "Fresh subprocess",
    provisional: "Provisional only",
    verified: "Verifier-backed",
  };

  return `
    <span class="provenance ${kind}">
      ${labels[kind] || kind}
    </span>
  `;
}

// -------------------- trace strip --------------------

function renderTrace(samples, containerLabel) {
  if (!samples || samples.length < 2) {
    const sampleCount = samples ? samples.length : 0;

    return `
      <div class="trace">
        <div class="trace-label">
          <span>${containerLabel}</span>
        </div>

        <div class="trace-empty">
          Only ${sampleCount} sample${sampleCount === 1 ? "" : "s"}
          recorded so far. Telemetry is sampled every couple seconds,
          so very short runs may not have enough points for a trace.
          ${
            sampleCount > 0
              ? "This updates live while the run is active. Refresh or wait a moment."
              : ""
          }
        </div>
      </div>
    `;
  }

  const hasGpu = samples.some(
    (sample) =>
      sample.gpu_util_pct !== null &&
      sample.gpu_util_pct !== undefined
  );

  const hasCpu = samples.some(
    (sample) =>
      sample.cpu_pct !== null &&
      sample.cpu_pct !== undefined
  );

  const firstTimestamp = samples[0].t;

  const timeSpan = Math.max(
    1,
    samples[samples.length - 1].t - firstTimestamp
  );

  const width = 1000;
  const height = 130;
  const leftPadding = 34;
  const rightPadding = 10;
  const topPadding = 8;
  const bottomPadding = 20;

  function toPath(key, color) {
    const points = [];

    samples.forEach((sample) => {
      const value = sample[key];

      if (value === null || value === undefined) {
        return;
      }

      const x =
        leftPadding +
        ((sample.t - firstTimestamp) / timeSpan) *
          (width - leftPadding - rightPadding);

      const y =
        height -
        bottomPadding -
        (Math.min(100, Math.max(0, value)) / 100) *
          (height - topPadding - bottomPadding);

      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    });

    if (points.length < 2) {
      return "";
    }

    return `
      <polyline
        points="${points.join(" ")}"
        fill="none"
        stroke="${color}"
        stroke-width="2"
        stroke-linejoin="round"
        stroke-linecap="round"
      />
    `;
  }

  const gpuPath = hasGpu
    ? toPath("gpu_util_pct", "var(--signal-mint)")
    : "";

  const cpuPath = hasCpu
    ? toPath("cpu_pct", "var(--signal-cyan)")
    : "";

  const yTicks = [0, 50, 100]
    .map((value) => {
      const y =
        height -
        bottomPadding -
        (value / 100) *
          (height - topPadding - bottomPadding);

      return `
        <line
          x1="${leftPadding}"
          x2="${width - rightPadding}"
          y1="${y}"
          y2="${y}"
          stroke="rgba(109,125,147,.18)"
          stroke-width="1"
        />

        <text
          x="${leftPadding - 6}"
          y="${y + 3}"
          text-anchor="end"
          font-size="9"
          fill="var(--ink-faint)"
          font-family="var(--font-mono)"
        >
          ${value}%
        </text>
      `;
    })
    .join("");

  const formatElapsed = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = String(
      Math.round(seconds % 60)
    ).padStart(2, "0");

    return `${minutes}:${remainingSeconds}`;
  };

  const xLabels = `
    <text
      x="${leftPadding}"
      y="${height - 4}"
      font-size="9"
      fill="var(--ink-faint)"
      font-family="var(--font-mono)"
    >
      0:00
    </text>

    <text
      x="${width - rightPadding}"
      y="${height - 4}"
      text-anchor="end"
      font-size="9"
      fill="var(--ink-faint)"
      font-family="var(--font-mono)"
    >
      ${formatElapsed(timeSpan)} elapsed
    </text>
  `;

  const legend = `
    <div class="trace-legend">
      ${
        hasGpu
          ? `
            <span>
              <i style="background:var(--signal-mint)"></i>
              GPU utilization
            </span>
          `
          : ""
      }

      ${
        hasCpu
          ? `
            <span>
              <i style="background:var(--signal-cyan)"></i>
              CPU utilization
            </span>
          `
          : ""
      }
    </div>
  `;

  const vramSamples = samples.filter(
    (sample) =>
      sample.gpu_mem_used_mib !== null &&
      sample.gpu_mem_used_mib !== undefined
  );

  let vramSection = "";

  if (vramSamples.length >= 2) {
    const values = vramSamples.map(
      (sample) => sample.gpu_mem_used_mib
    );

    const maximumValue = Math.max(...values) * 1.15 || 1;
    const vramHeight = 44;

    const vramPoints = vramSamples
      .map((sample) => {
        const x =
          leftPadding +
          ((sample.t - firstTimestamp) / timeSpan) *
            (width - leftPadding - rightPadding);

        const y =
          vramHeight -
          (sample.gpu_mem_used_mib / maximumValue) *
            (vramHeight - 6) -
          2;

        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    vramSection = `
      <div class="trace-label" style="margin-top:10px;">
        <span>VRAM used</span>
        <span>peak ${Math.max(...values).toFixed(0)} MiB</span>
      </div>

      <svg
        viewBox="0 0 ${width} ${vramHeight}"
        style="width:100%; height:${vramHeight}px; display:block;"
      >
        <polyline
          points="${vramPoints}"
          fill="none"
          stroke="var(--signal-violet)"
          stroke-width="1.5"
        />
      </svg>
    `;
  }

  const diskSection = samples.some(
    (sample) =>
      sample.disk_read_mbps !== null &&
      sample.disk_read_mbps !== undefined
  )
    ? renderRateMiniChart(
        samples,
        [
          {
            key: "disk_read_mbps",
            label: "read",
            color: "var(--signal-mint)",
          },
          {
            key: "disk_write_mbps",
            label: "write",
            color: "var(--signal-amber)",
          },
        ],
        "Disk I/O"
      )
    : "";

  const networkSection = samples.some(
    (sample) =>
      sample.net_sent_mbps !== null &&
      sample.net_sent_mbps !== undefined
  )
    ? renderRateMiniChart(
        samples,
        [
          {
            key: "net_sent_mbps",
            label: "sent",
            color: "var(--signal-cyan)",
          },
          {
            key: "net_recv_mbps",
            label: "received",
            color: "var(--signal-violet)",
          },
        ],
        "Network I/O"
      )
    : "";

  return `
    <div class="trace">
      <div class="trace-label">
        <span>${containerLabel}</span>
        <span>
          ${samples.length} samples over
          ${formatElapsed(timeSpan)}
        </span>
      </div>

      ${legend}

      <svg
        viewBox="0 0 ${width} ${height}"
        style="width:100%; height:${height}px; display:block;"
      >
        ${yTicks}
        ${gpuPath}
        ${cpuPath}
        ${xLabels}
      </svg>

      ${vramSection}
      ${diskSection}
      ${networkSection}
    </div>
  `;
}

function renderRateMiniChart(samples, seriesDefinitions, title) {
  const firstTimestamp = samples[0].t;

  const timeSpan = Math.max(
    1,
    samples[samples.length - 1].t - firstTimestamp
  );

  const width = 1000;
  const height = 70;
  const leftPadding = 46;
  const rightPadding = 10;
  const topPadding = 6;
  const bottomPadding = 16;
  const allValues = [];

  seriesDefinitions.forEach((definition) => {
    samples.forEach((sample) => {
      if (
        sample[definition.key] !== null &&
        sample[definition.key] !== undefined
      ) {
        allValues.push(sample[definition.key]);
      }
    });
  });

  if (!allValues.length) {
    return "";
  }

  const maximumValue = Math.max(...allValues, 0.001) * 1.15;

  const paths = seriesDefinitions
    .map((definition) => {
      const points = [];

      samples.forEach((sample) => {
        const value = sample[definition.key];

        if (value === null || value === undefined) {
          return;
        }

        const x =
          leftPadding +
          ((sample.t - firstTimestamp) / timeSpan) *
            (width - leftPadding - rightPadding);

        const y =
          height -
          bottomPadding -
          (Math.max(0, value) / maximumValue) *
            (height - topPadding - bottomPadding);

        points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
      });

      if (points.length < 2) {
        return "";
      }

      return `
        <polyline
          points="${points.join(" ")}"
          fill="none"
          stroke="${definition.color}"
          stroke-width="1.6"
        />
      `;
    })
    .join("");

  const legend = seriesDefinitions
    .map(
      (definition) => `
        <span>
          <i style="background:${definition.color}"></i>
          ${esc(definition.label)}
        </span>
      `
    )
    .join("");

  const yLabel = `
    <text
      x="${leftPadding - 6}"
      y="${topPadding + 8}"
      text-anchor="end"
      font-size="9"
      fill="var(--ink-faint)"
      font-family="var(--font-mono)"
    >
      ${fmtNum(maximumValue, 1)} MB/s
    </text>
  `;

  return `
    <div class="trace-label" style="margin-top:10px;">
      <span>${esc(title)}</span>
    </div>

    <div class="trace-legend">${legend}</div>

    <svg
      viewBox="0 0 ${width} ${height}"
      style="width:100%; height:${height}px; display:block;"
    >
      ${yLabel}
      ${paths}
    </svg>
  `;
}

// -------------------- per-metric line charts --------------------

const METRIC_CHART_COLORS = [
  "var(--signal-mint)",
  "var(--signal-cyan)",
  "var(--signal-violet)",
  "var(--signal-amber)",
];

function renderMetricChart(name, points, color) {
  if (!points || points.length === 0) {
    return `
      <div class="metric-chart-empty">
        ${esc(name)}: no data
      </div>
    `;
  }

  const sorted = [...points].sort(
    (left, right) =>
      (left.step ?? left.timestamp) -
      (right.step ?? right.timestamp)
  );

  const values = sorted.map((point) => point.value);

  const steps = sorted.map((point, index) =>
    point.step !== null && point.step !== undefined
      ? point.step
      : index
  );

  if (values.length === 1) {
    return `
      <div class="metric-chart">
        <div class="metric-chart-head">
          <span class="metric-chart-name">${esc(name)}</span>
          <span class="metric-chart-latest">
            ${fmtNum(values[0])}
          </span>
        </div>

        <p class="ai-empty" style="margin-top:8px;">
          Only one value has been logged. A trend line needs at
          least two points.
        </p>
      </div>
    `;
  }

  const width = 620;
  const height = 170;
  const leftPadding = 50;
  const rightPadding = 12;
  const topPadding = 12;
  const bottomPadding = 24;

  const minimumValue = Math.min(...values);
  const maximumValue = Math.max(...values);

  const valueSpan =
    maximumValue - minimumValue ||
    Math.abs(maximumValue || 1) * 0.1 ||
    1;

  const minimumStep = Math.min(...steps);
  const maximumStep = Math.max(...steps);
  const stepSpan = maximumStep - minimumStep || 1;

  const coordinates = sorted.map((point, index) => ({
    x:
      leftPadding +
      ((steps[index] - minimumStep) / stepSpan) *
        (width - leftPadding - rightPadding),
    y:
      height -
      bottomPadding -
      ((point.value - minimumValue) / valueSpan) *
        (height - topPadding - bottomPadding),
    value: point.value,
    step: steps[index],
  }));

  const linePoints = coordinates
    .map(
      (coordinate) =>
        `${coordinate.x.toFixed(1)},${coordinate.y.toFixed(1)}`
    )
    .join(" ");

  const yTicks = [0, 0.5, 1]
    .map((fraction) => {
      const value = minimumValue + valueSpan * fraction;

      const y =
        height -
        bottomPadding -
        fraction * (height - topPadding - bottomPadding);

      return `
        <line
          x1="${leftPadding}"
          x2="${width - rightPadding}"
          y1="${y}"
          y2="${y}"
          stroke="rgba(109,125,147,.16)"
          stroke-width="1"
        />

        <text
          x="${leftPadding - 8}"
          y="${y + 3}"
          text-anchor="end"
          font-size="9.5"
          fill="var(--ink-faint)"
          font-family="var(--font-mono)"
        >
          ${fmtNum(value, 3)}
        </text>
      `;
    })
    .join("");

  const dots = coordinates
    .map(
      (coordinate) => `
        <circle
          cx="${coordinate.x.toFixed(1)}"
          cy="${coordinate.y.toFixed(1)}"
          r="3.2"
          fill="${color}"
          stroke="#0d151d"
          stroke-width="1.3"
        >
          <title>
            step ${coordinate.step}: ${fmtNum(coordinate.value)}
          </title>
        </circle>
      `
    )
    .join("");

  return `
    <div class="metric-chart">
      <div class="metric-chart-head">
        <span class="metric-chart-name">${esc(name)}</span>
        <span class="metric-chart-latest">
          ${fmtNum(values[values.length - 1])}
        </span>
      </div>

      <svg
        viewBox="0 0 ${width} ${height}"
        style="width:100%; height:${height}px; display:block;"
      >
        ${yTicks}

        <polyline
          points="${linePoints}"
          fill="none"
          stroke="${color}"
          stroke-width="2"
          stroke-linejoin="round"
          stroke-linecap="round"
        />

        ${dots}
      </svg>

      <div class="metric-chart-xaxis">
        <span>step ${steps[0]}</span>
        <span>step ${steps[steps.length - 1]}</span>
      </div>
    </div>
  `;
}

function renderAllMetricCharts(metricsOverTime) {
  const names = Object.keys(metricsOverTime || {});

  if (!names.length) {
    return `<p class="ai-empty">no metrics logged</p>`;
  }

  return names
    .map((name, index) =>
      renderMetricChart(
        name,
        metricsOverTime[name],
        METRIC_CHART_COLORS[index % METRIC_CHART_COLORS.length]
      )
    )
    .join("");
}

function renderSparkline(
  values,
  color = "var(--signal-mint)"
) {
  if (!values || values.length < 2) {
    return `
      <div class="trace-empty">
        not enough trials for a chart yet
      </div>
    `;
  }

  const width = 600;
  const height = 210;
  const horizontalPadding = 10;
  const verticalPadding = 22;

  const minimumValue = Math.min(...values);
  const maximumValue = Math.max(...values);

  const margin = Math.max(
    (maximumValue - minimumValue) * 0.28,
    Math.abs(maximumValue || 1) * 0.025
  );

  const lowerBound = minimumValue - margin;
  const upperBound = maximumValue + margin;
  const valueSpan = upperBound - lowerBound || 1;

  const coordinates = values.map((value, index) => ({
    x:
      horizontalPadding +
      (index / (values.length - 1)) *
        (width - horizontalPadding * 2),
    y:
      height -
      verticalPadding -
      ((value - lowerBound) / valueSpan) *
        (height - verticalPadding * 2),
  }));

  const points = coordinates
    .map(
      (coordinate) =>
        `${coordinate.x.toFixed(1)},${coordinate.y.toFixed(1)}`
    )
    .join(" ");

  const lastCoordinate = coordinates[coordinates.length - 1];

  const areaPoints = `
    ${horizontalPadding},${height - verticalPadding}
    ${points}
    ${width - horizontalPadding},${height - verticalPadding}
  `;

  const grid = [0.25, 0.5, 0.75]
    .map((fraction) => {
      const y =
        verticalPadding +
        (height - verticalPadding * 2) * fraction;

      return `
        <line
          x1="${horizontalPadding}"
          x2="${width - horizontalPadding}"
          y1="${y}"
          y2="${y}"
          stroke="rgba(109,125,147,.20)"
          stroke-width="1"
        />
      `;
    })
    .join("");

  const gradientId = `objective-gradient-${Math.random()
    .toString(36)
    .slice(2)}`;

  return `
    <div class="objective-chart">
      <svg
        viewBox="0 0 ${width} ${height}"
        preserveAspectRatio="none"
        aria-label="Trial objective trend"
      >
        <defs>
          <linearGradient
            id="${gradientId}"
            x1="0"
            x2="0"
            y1="0"
            y2="1"
          >
            <stop
              offset="0"
              stop-color="${color}"
              stop-opacity=".34"
            />

            <stop
              offset="1"
              stop-color="${color}"
              stop-opacity="0"
            />
          </linearGradient>
        </defs>

        ${grid}

        <polygon
          points="${areaPoints}"
          fill="url(#${gradientId})"
        />

        <polyline
          points="${points}"
          fill="none"
          stroke="${color}"
          stroke-width="3.6"
          vector-effect="non-scaling-stroke"
          stroke-linejoin="round"
          stroke-linecap="round"
        />

        <circle
          cx="${lastCoordinate.x}"
          cy="${lastCoordinate.y}"
          r="6"
          fill="${color}"
          stroke="#0d151d"
          stroke-width="3"
          vector-effect="non-scaling-stroke"
        />
      </svg>
    </div>
  `;
}

// -------------------- router --------------------

const routes = [
  [/^#\/$/, "overview", renderOverviewScreen],
  [/^#\/projects$/, "projects", renderProjectsScreen],
  [
    /^#\/runs$/,
    "runs",
    () =>
      renderGlobalRunsScreen(
        new URLSearchParams(location.hash.split("?")[1])
      ),
  ],
  [/^#\/failures$/, "failures", renderFailuresScreen],
  [/^#\/campaigns$/, "campaigns", renderCampaignsScreen],
  [/^#\/memory$/, "memory", renderMemoryScreen],
  [/^#\/settings$/, "settings", renderSettingsScreen],
  [
    /^#\/project\/([^/]+)$/,
    "projects",
    (match) =>
      renderProjectRunsScreen(decodeURIComponent(match[1])),
  ],
  [
    /^#\/run\/([^/]+)$/,
    "runs",
    (match) => renderRunScreen(decodeURIComponent(match[1])),
  ],
  [
    /^#\/failure\/([^/]+)$/,
    "failures",
    (match) =>
      renderFailureScreen(decodeURIComponent(match[1])),
  ],
  [
    /^#\/campaign\/([^/]+)$/,
    "campaigns",
    (match) =>
      renderCampaignScreen(decodeURIComponent(match[1])),
  ],
  [
    /^#\/compare$/,
    "runs",
    () =>
      renderCompareScreen(
        new URLSearchParams(location.hash.split("?")[1])
      ),
  ],
  [
    /^#\/overlay$/,
    "runs",
    () =>
      renderOverlayScreen(
        new URLSearchParams(location.hash.split("?")[1])
      ),
  ],
];

function route() {
  stopActivePoll();
  startRouteProgress();

  const hash = location.hash || "#/";
  const path = hash.split("?")[0];

  for (const [pattern, navigationKey, handler] of routes) {
    const match = path.match(pattern);

    if (match) {
            updateActiveNav(navigationKey);

      Promise.resolve(handler(match)).finally(() => {
        finishRouteProgress();
        app.focus({ preventScroll: true });
        window.scrollTo({ top: 0, behavior: "instant" });
      });

      return;
    }
  }

  app.innerHTML = `
    <div class="empty-state">
      <p class="eyebrow">404</p>
      <p>Unknown view.</p>
    </div>
  `;

  finishRouteProgress();
}

function updateActiveNav(navigationKey) {
  document.querySelectorAll("#sidebar-nav a").forEach((link) => {
    link.classList.toggle(
      "active",
      link.dataset.route === navigationKey
    );
  });
}

window.addEventListener("hashchange", route);

window.addEventListener("DOMContentLoaded", () => {
  setupGlobalUI();
  route();
});

// -------------------- screen: overview --------------------

async function renderOverviewScreen() {
  app.innerHTML = `
    <p class="loading">loading overview&hellip;</p>
  `;

  let overview;

  try {
    overview = await api("/overview");
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  if (overview.run_count === 0) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">no runs yet</p>
        <p>
          Start one from Python:
          <code>
            with watcher.init(project="...", config={...}) as run:
          </code>
        </p>
      </div>
    `;

    return;
  }

  const runsNeedingAttention = (
    overview.runs_needing_attention || []
  )
    .map(
      (run) => `
        <tr>
          <td>
            <a href="#/run/${encodeURIComponent(run.run_id)}">
              ${esc(run.display_name)}
            </a>
          </td>

          <td>${esc(run.project)}</td>
          <td>${esc(run.failure_category)}</td>

          <td>
            <a href="#/failure/${encodeURIComponent(run.run_id)}">
              investigate &rarr;
            </a>
          </td>
        </tr>
      `
    )
    .join("");

  const recentVerifiedRecoveries =
    overview.recent_verified_recoveries ||
    overview.recent_verified_fixes ||
    [];

  const verifiedRecoveryRows = recentVerifiedRecoveries
    .map(
      (campaign) => `
        <tr>
          <td>
            <a
              href="#/campaign/${encodeURIComponent(
                campaign.campaign_id
              )}"
            >
              ${esc(campaign.campaign_id)}
            </a>
          </td>

          <td>${esc(campaign.project)}</td>

          <td>
            ${esc(
              campaign.verified_candidate_id ||
                "verified candidate"
            )}
          </td>

          <td>
            ${(campaign.verified_run_ids || []).length}
            confirmation run${
              (campaign.verified_run_ids || []).length === 1
                ? ""
                : "s"
            }
          </td>
        </tr>
      `
    )
    .join("");

  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Overview</h1>

    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">Projects</div>
        <div class="stat-value">
          ${overview.project_count}
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Total runs</div>
        <div class="stat-value">
          ${overview.run_count}
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">
          Runs needing attention
        </div>

        <div
          class="stat-value ${
            (overview.runs_needing_attention || []).length
              ? "red"
              : ""
          }"
        >
          ${(overview.runs_needing_attention || []).length}
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Active campaigns</div>
        <div class="stat-value mint">
          ${overview.active_campaign_count}
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">Verified recoveries</div>
        <div class="stat-value mint">
          ${overview.verified_recovery_count || 0}
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">GPU</div>

        <div class="stat-value" style="font-size:15px;">
          ${
            overview.gpu_available
              ? esc(overview.gpu_name)
              : "not detected"
          }
        </div>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">
        Runs needing attention
      </h2>

      ${
        runsNeedingAttention
          ? `
            <table class="runs-table">
              <thead>
                <tr>
                  <th>run</th>
                  <th>project</th>
                  <th>failure</th>
                  <th></th>
                </tr>
              </thead>

              <tbody>${runsNeedingAttention}</tbody>
            </table>
          `
          : `
            <p class="ai-empty">
              Nothing unresolved right now.
            </p>
          `
      }
    </div>

    <div class="panel">
      <h2 class="section-title">
        Recent verified recoveries
        ${provenance("verified")}
      </h2>

      ${
        verifiedRecoveryRows
          ? `
            <table class="runs-table">
              <thead>
                <tr>
                  <th>campaign</th>
                  <th>project</th>
                  <th>candidate</th>
                  <th>confirmation</th>
                </tr>
              </thead>

              <tbody>${verifiedRecoveryRows}</tbody>
            </table>
          `
          : `
            <p class="ai-empty">
              No campaign has passed independent confirmation yet.
            </p>
          `
      }
    </div>

    <div class="panel">
      <h2 class="section-title">
        Execution boundary
      </h2>

      <p>
        Recovery work runs explicitly from the Python SDK or CLI in
        fresh subprocesses. This browser is a read-only evidence and
        audit surface.
      </p>

      <code>watcher recovery CAMPAIGN_ID</code>
    </div>
  `;
}

// -------------------- screen: projects --------------------

async function renderProjectsScreen() {
  app.innerHTML = `
    <p class="loading">loading projects&hellip;</p>
  `;

  let projects;

  try {
    projects = await api("/projects");
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  if (!projects.length) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">no runs yet</p>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="eyebrow">watcherml / projects</p>
    <h1 class="page-title">Projects</h1>

    <div class="card-grid">
      ${projects
        .map(
          (project) => `
            <a
              class="card"
              href="#/project/${encodeURIComponent(project.name)}"
            >
              <div class="card-title">
                ${esc(project.name)}
              </div>

              <div class="card-meta">
                <span>
                  ${project.run_count}
                  run${project.run_count === 1 ? "" : "s"}
                </span>

                <span
                  class="${
                    project.failure_count ? "fail-count" : ""
                  }"
                >
                  ${project.failure_count} failed
                </span>
              </div>
            </a>
          `
        )
        .join("")}
    </div>
  `;
}

// -------------------- screen: run lists --------------------

async function renderProjectRunsScreen(project) {
  app.innerHTML = `
    <p class="loading">loading runs&hellip;</p>
  `;

  let runs;

  try {
    runs = await api(
      `/projects/${encodeURIComponent(project)}/runs`
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  renderRunsTable(
    runs,
    `<a href="#/">projects</a> / ${esc(project)}`,
    project
  );
}

async function renderGlobalRunsScreen(parameters) {
  app.innerHTML = `
    <p class="loading">loading runs&hellip;</p>
  `;

  const status = parameters.get("status") || "";

  const query = status
    ? `?status=${encodeURIComponent(status)}`
    : "";

  let runs;

  try {
    runs = await api(`/runs${query}`);
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  renderRunsTable(runs, "watcherml", null, status);
}

let runsTableSort = {
  key: null,
  dir: 1,
};

const MAX_OVERLAY_RUNS = 6;

function sortValue(row, key) {
  if (key.startsWith("metric:")) {
    return row.final_metrics[key.slice(7)];
  }

  return row[key];
}

function renderRunsTable(
  runs,
  breadcrumbHtml,
  project,
  currentStatusFilter
) {
  const metricNames = [
    ...new Set(
      runs.flatMap((run) =>
        Object.keys(run.final_metrics || {})
      )
    ),
  ].slice(0, 4);

  const filterBar = project
    ? ""
    : `
      <div class="filter-bar">
        <span>filter:</span>

        <a
          href="#/runs"
          style="${
            !currentStatusFilter
              ? "color:var(--signal-mint)"
              : ""
          }"
        >
          all
        </a>

        <a
          href="#/runs?status=failed"
          style="${
            currentStatusFilter === "failed"
              ? "color:var(--signal-mint)"
              : ""
          }"
        >
          failed
        </a>

        <a
          href="#/runs?status=success"
          style="${
            currentStatusFilter === "success"
              ? "color:var(--signal-mint)"
              : ""
          }"
        >
          success
        </a>
      </div>
    `;

  const sortedRuns = [...runs];

  if (runsTableSort.key) {
    sortedRuns.sort((left, right) => {
      const leftValue = sortValue(
        left,
        runsTableSort.key
      );

      const rightValue = sortValue(
        right,
        runsTableSort.key
      );

      if (
        leftValue === null ||
        leftValue === undefined
      ) {
        return 1;
      }

      if (
        rightValue === null ||
        rightValue === undefined
      ) {
        return -1;
      }

      if (leftValue < rightValue) {
        return -1 * runsTableSort.dir;
      }

      if (leftValue > rightValue) {
        return runsTableSort.dir;
      }

      return 0;
    });
  }

  const sortableHeader = (label, key) => {
    const active = runsTableSort.key === key;

    const arrow = active
      ? runsTableSort.dir === 1
        ? " &#9650;"
        : " &#9660;"
      : "";

    return `
      <th
        class="sortable-th"
        data-sort-key="${esc(key)}"
        title="Click to sort"
      >
        ${esc(label)}${arrow}
      </th>
    `;
  };

  app.innerHTML = `
    <p class="eyebrow">${breadcrumbHtml}</p>

    <h1 class="page-title">
      ${project ? esc(project) : "Runs"}
    </h1>

    ${filterBar}

    ${
      project
        ? `
          <div class="compare-picker">
            <span>
              Select two or more runs to compare or overlay their
              metrics:
            </span>

            <button id="compare-btn" disabled>
              Compare selected (2)
            </button>

            <button id="overlay-btn" disabled>
              Overlay metrics
            </button>
          </div>
        `
        : ""
    }

    <div class="panel">
      <div class="runs-table-wrapper">
        <table class="runs-table">
          <thead>
            <tr>
              ${project ? "<th></th>" : ""}
              ${sortableHeader("run", "display_name")}
              ${sortableHeader("status", "status")}
              ${sortableHeader("started", "started_at")}
              ${sortableHeader(
                "duration",
                "duration_seconds"
              )}

              ${metricNames
                .map((metric) =>
                  sortableHeader(
                    metric,
                    `metric:${metric}`
                  )
                )
                .join("")}

              ${sortableHeader("hardware", "hardware")}
              ${sortableHeader("warnings", "warning_count")}
            </tr>
          </thead>

          <tbody>
            ${sortedRuns
              .map(
                (run) => `
                  <tr>
                    ${
                      project
                        ? `
                          <td>
                            <input
                              type="checkbox"
                              class="compare-check"
                              value="${esc(run.run_id)}"
                            />
                          </td>
                        `
                        : ""
                    }

                    <td>
                      <a
                        href="#/run/${encodeURIComponent(
                          run.run_id
                        )}"
                      >
                        <span class="run-name">
                          ${esc(run.display_name)}
                        </span>

                        <span class="run-id-sub">
                          ${esc(run.run_id)}
                        </span>
                      </a>

                      ${demoBadge(run.simulated)}
                      ${resolvedBadge(run.resolved)}
                    </td>

                    <td>${badge(run.status)}</td>
                    <td>${fmtTimestamp(run.started_at)}</td>

                    <td>
                      ${fmtDuration(run.duration_seconds)}
                    </td>

                    ${metricNames
                      .map(
                        (metric) => `
                          <td>
                            ${fmtNum(
                              run.final_metrics[metric]
                            )}
                          </td>
                        `
                      )
                      .join("")}

                    <td>${esc(run.hardware)}</td>

                    <td>
                      ${
                        run.warning_count > 0
                          ? `
                            <span
                              style="color:var(--signal-amber)"
                            >
                              ${run.warning_count}
                            </span>
                          `
                          : "0"
                      }
                    </td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;

  document
    .querySelectorAll(".sortable-th")
    .forEach((tableHeader) => {
      tableHeader.addEventListener("click", () => {
        const key = tableHeader.dataset.sortKey;

        runsTableSort.dir =
          runsTableSort.key === key
            ? -runsTableSort.dir
            : 1;

        runsTableSort.key = key;

        renderRunsTable(
          runs,
          breadcrumbHtml,
          project,
          currentStatusFilter
        );
      });
    });

  if (project) {
    const checkboxes = [
      ...document.querySelectorAll(".compare-check"),
    ];

    const compareButton =
      document.getElementById("compare-btn");

    const overlayButton =
      document.getElementById("overlay-btn");

    checkboxes.forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const checked = checkboxes.filter(
          (item) => item.checked
        );

        if (checked.length > MAX_OVERLAY_RUNS) {
          checkbox.checked = false;
          return;
        }

        compareButton.disabled = checked.length !== 2;
        overlayButton.disabled = checked.length < 2;
      });
    });

    compareButton.addEventListener("click", () => {
      const [firstRun, secondRun] = checkboxes
        .filter((checkbox) => checkbox.checked)
        .map((checkbox) => checkbox.value);

      location.hash =
        `#/compare?a=${encodeURIComponent(firstRun)}` +
        `&b=${encodeURIComponent(secondRun)}`;
    });

    overlayButton.addEventListener("click", () => {
      const runIds = checkboxes
        .filter((checkbox) => checkbox.checked)
        .map((checkbox) => checkbox.value);

      location.hash = `#/overlay?ids=${runIds
        .map(encodeURIComponent)
        .join(",")}`;
    });
  }
}

// -------------------- screen: run detail --------------------

async function renderRunScreen(runId) {
  app.innerHTML = `
    <p class="loading">loading run&hellip;</p>
  `;

  let run;
  let samples;

  try {
    [run, samples] = await Promise.all([
      api(`/runs/${encodeURIComponent(runId)}`),
      api(`/runs/${encodeURIComponent(runId)}/samples`),
    ]);
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  renderRunScreenContent(runId, run, samples);

  const MAX_LIVE_POLLS = 200;
  let pollCount = 0;

  if (run.status === "running") {
    activePollTimer = setInterval(async () => {
      pollCount += 1;

      if (pollCount > MAX_LIVE_POLLS) {
        stopActivePoll();
        return;
      }

      let updatedRun;
      let updatedSamples;

      try {
        [updatedRun, updatedSamples] =
          await Promise.all([
            api(`/runs/${encodeURIComponent(runId)}`),
            api(
              `/runs/${encodeURIComponent(runId)}/samples`
            ),
          ]);
      } catch (_) {
        return;
      }

      renderRunScreenContent(
        runId,
        updatedRun,
        updatedSamples
      );

      if (updatedRun.status !== "running") {
        stopActivePoll();
      }
    }, 3000);
  }
}

function renderRunScreenContent(
  runId,
  run,
  samples
) {
  const configRows =
    Object.entries(run.config || {})
      .filter(([key]) => key !== "_simulated")
      .map(
        ([key, value]) => `
          <div class="field">
            <span class="field-label">
              ${esc(key)}
            </span>

            <span class="field-value">
              ${esc(value)}
            </span>
          </div>
        `
      )
      .join("") ||
    `<p class="ai-empty">no config recorded</p>`;

  const failureLink = run.has_failure
    ? `
      <div class="failure-banner">
        <p class="eyebrow">this run failed</p>

        <a href="#/failure/${encodeURIComponent(runId)}">
          <strong>
            View failure capsule &rarr;
          </strong>
        </a>
      </div>
    `
    : "";

  const warningsHtml = (run.warnings || []).length
    ? `
      <div class="panel">
        <h2 class="section-title">Warnings</h2>

        <ul class="warning-list">
          ${run.warnings
            .map(
              (warning) => `
                <li>${esc(warning)}</li>
              `
            )
            .join("")}
        </ul>
      </div>
    `
    : "";

  const timelineItems = buildTimeline(run);

  const liveTag =
    run.status === "running"
      ? `
        <span
          class="live-label"
          style="margin-left:8px;"
        >
          &#9679; live, updating every 3s
        </span>
      `
      : "";

  app.innerHTML = `
    <p class="eyebrow">
      <a href="#/">projects</a>
      /
      <a
        href="#/project/${encodeURIComponent(run.project)}"
      >
        ${esc(run.project)}
      </a>
      / run
    </p>

    <h1 class="page-title">
      <span id="run-title">
        ${esc(run.display_name)}
      </span>

      ${badge(run.status)}
      ${demoBadge(run.simulated)}
      ${resolvedBadge(run.resolved)}
    </h1>

    <p class="page-subtitle">
      ${esc(run.run_id)}
    </p>

    <div style="margin-bottom:16px;">
      ${tagPills(run.tags)}
    </div>

    <div class="action-bar">
      <button id="rename-btn">Rename</button>

      <a
        href="/api/runs/${encodeURIComponent(runId)}/export"
      >
        <button>Export capsule</button>
      </a>

      <a
        href="/api/runs/${encodeURIComponent(
          runId
        )}/metrics.csv"
      >
        <button>Export metrics (CSV)</button>
      </a>
    </div>

    ${failureLink}

    <div class="panel">
      <h2 class="section-title">
        Metrics
        ${provenance("calculated")}
        ${liveTag}
      </h2>

      ${renderAllMetricCharts(run.metrics_over_time)}
    </div>

    ${renderTrace(samples, "system telemetry")}

    <div class="panel-grid">
      <div class="panel">
        <h2 class="section-title">Config</h2>
        ${configRows}
      </div>

      <div class="panel">
        <h2 class="section-title">
          Reproduction
        </h2>

        <div class="field">
          <span class="field-label">duration</span>
          <span class="field-value">
            ${fmtDuration(run.duration_seconds)}
          </span>
        </div>

        <div class="field">
          <span class="field-label">git_state</span>
          <span class="field-value">
            ${
              run.git && run.git.available
                ? run.git.dirty
                  ? "dirty"
                  : "clean"
                : "no_git_repo"
            }
          </span>
        </div>

        <div class="field">
          <span class="field-label">
            dataset_fingerprint
          </span>
          <span class="field-value">
            ${esc(run.dataset_fingerprint)}
          </span>
        </div>

        <div class="field">
          <span class="field-label">
            reproduction_score
          </span>
          <span class="field-value">
            ${
              run.reproduction_score !== null &&
              run.reproduction_score !== undefined
                ? `${run.reproduction_score}/10`
                : "&mdash;"
            }
          </span>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">Timeline</h2>
      <div class="timeline">${timelineItems}</div>
    </div>

    ${warningsHtml}
  `;

  document
    .getElementById("rename-btn")
    .addEventListener("click", async () => {
      const nextName = prompt(
        "Rename this run:",
        run.display_name === run.run_id
          ? ""
          : run.display_name
      );

      if (nextName === null) {
        return;
      }

      await api(
        `/runs/${encodeURIComponent(runId)}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            display_name: nextName || null,
          }),
        }
      );

      notify("Run name updated");
      stopActivePoll();
      renderRunScreen(runId);
    });
}

function buildTimeline(run) {
  const items = [
    {
      t: run.started_at,
      label: "Run started",
      cls: "",
    },
  ];

  for (const warning of run.warnings || []) {
    items.push({
      t: run.started_at,
      label: `Warning: ${warning}`,
      cls: "warning",
    });
  }

  if (run.has_failure) {
    items.push({
      t: run.ended_at,
      label: "Run failed",
      cls: "failure",
    });
  } else if (run.ended_at) {
    items.push({
      t: run.ended_at,
      label: "Run completed",
      cls: "",
    });
  }

  items.sort(
    (left, right) => (left.t || 0) - (right.t || 0)
  );

  return (
    items
      .map(
        (item) => `
          <div class="timeline-item ${item.cls}">
            <div class="timeline-time">
              ${fmtTimestamp(item.t)}
            </div>

            <div class="timeline-label">
              ${esc(item.label)}
            </div>
          </div>
        `
      )
      .join("") ||
    `
      <p class="ai-empty">
        No timeline events recorded.
      </p>
    `
  );
}

// -------------------- screen: failure capsule --------------------

async function renderFailureScreen(runId) {
  app.innerHTML = `
    <p class="loading">
      loading failure capsule&hellip;
    </p>
  `;

  let failure;

  try {
    failure = await api(
      `/runs/${encodeURIComponent(runId)}/failure`
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const diagnosis = failure.diagnosis || {};

  const failureClass =
    failure.failure_class ||
    diagnosis.rule ||
    "unclassified";

  const suggestedActions = (
    diagnosis.suggested_actions || []
  )
    .map((action) => `<li>${esc(action)}</li>`)
    .join("");

  const evidenceIndex = {};

  (failure.evidence_index || []).forEach((evidence) => {
    evidenceIndex[evidence.id] = evidence.label;
  });

  const evidenceChips = (
    diagnosis.evidence_ids || []
  )
    .map(
      (evidenceId) => `
        <span
          class="evidence-chip"
          title="${esc(
            evidenceIndex[evidenceId] || ""
          )}"
        >
          ${esc(evidenceId)}
        </span>
      `
    )
    .join("");

  const recentMetrics = (
    failure.evidence?.recent_metrics || []
  )
    .map(
      (metric) => `
        <div class="field">
          <span class="field-label">
            ${esc(metric.name)}
            (step ${metric.step})
          </span>

          <span class="field-value">
            ${fmtNum(metric.value)}
          </span>
        </div>
      `
    )
    .join("") ||
    `
      <p class="ai-empty">
        no metrics logged before failure
      </p>
    `;

  const similarFailures = (
    failure.similar_previous_failures || []
  )
    .map(
      (similarFailure) => `
        <li>
          <a
            href="#/failure/${encodeURIComponent(
              similarFailure.run_id
            )}"
          >
            ${esc(similarFailure.run_id)}
          </a>

          &mdash;

          ${esc(
            (similarFailure.message || "").slice(0, 80)
          )}
        </li>
      `
    )
    .join("");

  const comparison =
    failure.comparison_to_last_success;

  let comparisonHtml = `
    <span class="ai-empty">
      no previous successful run to compare against
    </span>
  `;

  if (comparison) {
    const checklist = (
      comparison.checklist || []
    )
      .map(
        (check) => `
          <div class="checklist-row">
            <span
              class="${
                check.matched
                  ? "check-yes"
                  : "check-no"
              }"
            >
              ${
                check.matched
                  ? "&check;"
                  : "&times;"
              }
            </span>

            ${esc(check.label)}
          </div>
        `
      )
      .join("");

    comparisonHtml = `
      <p>
        Nearest successful run:

        <a
          href="#/run/${encodeURIComponent(
            comparison.run_id
          )}"
        >
          ${esc(comparison.run_id)}
        </a>

        &mdash;

        similarity
        ${(
          comparison.similarity_score * 100
        ).toFixed(0)}%
      </p>

      ${checklist}

      <a
        href="#/compare?a=${encodeURIComponent(
          runId
        )}&b=${encodeURIComponent(
          comparison.run_id
        )}"
      >
        Full comparison &rarr;
      </a>
    `;
  }
    app.innerHTML = `
    <p class="eyebrow">
      <a href="#/run/${encodeURIComponent(runId)}">
        ${esc(failure.display_name)}
      </a>
      / failure
    </p>

    <div class="failure-banner">
      <p class="eyebrow">
        ${esc(failureClass)}
        ${demoBadge(failure.simulated)}
        ${resolvedBadge(failure.resolved)}
      </p>

      <h1
        class="page-title"
        style="margin-bottom:6px;"
      >
        ${esc(failure.exception_type)}
      </h1>

      <p class="failure-message">
        ${esc(failure.message)}
      </p>
    </div>

    <div class="action-bar">
      ${
        comparison
          ? `
            <a
              href="#/compare?a=${encodeURIComponent(
                runId
              )}&b=${encodeURIComponent(
                comparison.run_id
              )}"
            >
              <button>Compare baseline</button>
            </a>
          `
          : ""
      }

      <a
        href="/api/runs/${encodeURIComponent(runId)}/export"
      >
        <button>Export capsule</button>
      </a>

      <button
        data-copy="watcher prepare-recovery ${esc(
          runId
        )} --entrypoint train:main --out recovery-plan.json"
        data-copy-label="Preparation command copied"
      >
        Copy recovery command
      </button>
    </div>

    <div class="panel-grid">
      <div class="panel">
        <h2 class="section-title">
          Diagnosis
          ${provenance("rule-based")}
        </h2>

        <p>${esc(diagnosis.summary)}</p>

        ${
          diagnosis.likely_cause
            ? `
              <p>
                <strong>Likely cause:</strong>
                ${esc(diagnosis.likely_cause)}
              </p>
            `
            : ""
        }

        ${
          evidenceChips
            ? `
              <div class="evidence-row">
                ${evidenceChips}
              </div>
            `
            : ""
        }

        ${
          suggestedActions
            ? `
              <ul class="suggested-actions">
                ${suggestedActions}
              </ul>
            `
            : ""
        }
      </div>

      <div class="panel">
        <h2 class="section-title">
          Recent metrics before failure
        </h2>

        ${recentMetrics}

        <h2
          class="section-title"
          style="margin-top:16px;"
        >
          Similar previous failures
        </h2>

        ${
          similarFailures
            ? `
              <ul class="suggested-actions">
                ${similarFailures}
              </ul>
            `
            : `
              <p class="ai-empty">none recorded</p>
            `
        }

        <h2
          class="section-title"
          style="margin-top:16px;"
        >
          Comparison
          ${provenance("calculated")}
        </h2>

        ${comparisonHtml}
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">
        Prepare bounded recovery
        ${provenance("deterministic")}
      </h2>

      <p>
        The zero-compute preparation step seals the failure
        evidence, entrypoint identity, contract, capabilities,
        and policy-filtered proposals before any GPU trial starts.
      </p>

      <pre class="traceback"><code>watcher prepare-recovery ${esc(
        runId
      )} --entrypoint train:main --out recovery-plan.json
watcher recover --plan recovery-plan.json</code></pre>

      <p class="page-subtitle">
        Replace <code>train:main</code> with your importable
        training callable. The web UI never starts GPU work or
        approves interventions.
      </p>
    </div>

    <div class="panel">
      <h2 class="section-title">Full traceback</h2>

      <div class="traceback">
        ${esc(failure.traceback)}
      </div>
    </div>
  `;
}

// -------------------- screen: compare --------------------

async function renderCompareScreen(parameters) {
  const firstRunId = parameters.get("a");
  const secondRunId = parameters.get("b");

  if (!firstRunId || !secondRunId) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">missing runs</p>

        <p>
          Select two runs from a project page to compare.
        </p>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="loading">comparing runs&hellip;</p>
  `;

  let comparison;

  try {
    comparison = await api(
      `/compare?a=${encodeURIComponent(
        firstRunId
      )}&b=${encodeURIComponent(secondRunId)}`
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const configRows =
    (comparison.config_diff || [])
      .map((change) =>
        diffRow(
          change.key,
          change.from,
          change.to
        )
      )
      .join("") ||
    `<p class="ai-empty">no config changes</p>`;

  const metricRows = (
    comparison.metric_diff || []
  )
    .map((metric) => {
      const deltaClass =
        metric.delta === undefined
          ? ""
          : metric.delta >= 0
            ? "diff-delta-up"
            : "diff-delta-down";

      const deltaText =
        metric.delta !== undefined
          ? ` (${
              metric.delta >= 0 ? "+" : ""
            }${fmtNum(metric.delta)})`
          : "";

      return `
        <div class="diff-row">
          <span class="diff-key">
            ${esc(metric.metric)}
          </span>

          <span class="diff-from">
            ${fmtNum(metric.from)}
          </span>

          <span class="diff-arrow">&rarr;</span>

          <span class="diff-to ${deltaClass}">
            ${fmtNum(metric.to)}

            <span class="${deltaClass}">
              ${esc(deltaText)}
            </span>
          </span>
        </div>
      `;
    })
    .join("");

  const packageChanges =
    comparison.package_diff || [];

  const packageRows = packageChanges
    .slice(0, 15)
    .map((change) =>
      diffRow(
        change.package,
        change.from,
        change.to
      )
    )
    .join("");

  const packageSection = packageChanges.length
    ? `
      <h2
        class="section-title"
        style="margin-top:16px;"
      >
        Package changes (${packageChanges.length})
      </h2>

      ${packageRows}
    `
    : "";

  app.innerHTML = `
    <p class="eyebrow">compare</p>

    <h1 class="page-title">
      ${esc(firstRunId)}
      &rarr;
      ${esc(secondRunId)}
    </h1>

    <div class="panel">
      <h2 class="section-title">What changed?</h2>

      ${configRows}

      ${
        comparison.dataset_changed
          ? `
            <div class="diff-row">
              <span class="diff-key">dataset</span>

              <span
                class="diff-from"
                style="
                  grid-column:span 3;
                  text-align:left;
                  color:var(--signal-amber);
                "
              >
                fingerprint changed
              </span>
            </div>
          `
          : ""
      }

      ${
        comparison.git_diff?.commit_changed
          ? diffRow(
              "git commit",
              (
                comparison.git_diff.commit_a || ""
              ).slice(0, 10),
              (
                comparison.git_diff.commit_b || ""
              ).slice(0, 10)
            )
          : ""
      }

      ${packageSection}
    </div>

    <div class="panel">
      <h2 class="section-title">
        What changed in results?
      </h2>

      ${
        metricRows ||
        `
          <p class="ai-empty">
            no metrics to compare
          </p>
        `
      }

      <div class="diff-row">
        <span class="diff-key">
          exit status
        </span>

        <span class="diff-from">
          ${badge(comparison.exit_status_a)}
        </span>

        <span class="diff-arrow">&rarr;</span>

        <span class="diff-to">
          ${badge(comparison.exit_status_b)}
        </span>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">
        Interpretation boundary
      </h2>

      <p>
        This view reports recorded configuration, package,
        dataset, git, metric, and exit-status differences.
        It does not invent a causal explanation. Recovery
        candidates are generated separately by the bounded
        OOM policy and must survive isolated trials and
        independent confirmation.
      </p>
    </div>
  `;
}

function diffRow(key, fromValue, toValue) {
  return `
    <div class="diff-row">
      <span class="diff-key">
        ${esc(key)}
      </span>

      <span class="diff-from">
        ${esc(fromValue)}
      </span>

      <span class="diff-arrow">&rarr;</span>

      <span class="diff-to">
        ${esc(toValue)}
      </span>
    </div>
  `;
}

// -------------------- multi-run metric overlay --------------------

const OVERLAY_COLORS = [
  "var(--signal-mint)",
  "var(--signal-cyan)",
  "var(--signal-violet)",
  "var(--signal-amber)",
  "#f472b6",
  "#fb923c",
];

async function renderOverlayScreen(parameters) {
  const encodedIds = parameters.get("ids") || "";

  const runIds = encodedIds
    .split(",")
    .filter(Boolean);

  if (runIds.length < 2) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">
          select at least two runs
        </p>

        <p>
          Go to a project run list, check two or more
          runs, and select “Overlay metrics.”
        </p>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="loading">
      loading runs to overlay&hellip;
    </p>
  `;

  let runs;

  try {
    runs = await Promise.all(
      runIds.map((runId) =>
        api(`/runs/${encodeURIComponent(runId)}`)
      )
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const metricNames = [
    ...new Set(
      runs.flatMap((run) =>
        Object.keys(run.metrics_over_time || {})
      )
    ),
  ];

  app.innerHTML = `
    <p class="eyebrow">watcherml / overlay</p>

    <h1 class="page-title">
      Compare metrics across ${runs.length} runs
      ${provenance("calculated")}
    </h1>

    <div class="overlay-legend">
      ${runs
        .map(
          (run, index) => `
            <span>
              <i
                style="
                  background:${
                    OVERLAY_COLORS[
                      index % OVERLAY_COLORS.length
                    ]
                  }
                "
              ></i>

              <a
                href="#/run/${encodeURIComponent(
                  run.run_id
                )}"
              >
                ${esc(run.display_name)}
              </a>
            </span>
          `
        )
        .join("")}
    </div>

    <div class="panel">
      ${
        metricNames.length
          ? metricNames
              .map((metricName) =>
                renderOverlayChart(metricName, runs)
              )
              .join("")
          : `
            <p class="ai-empty">
              None of the selected runs have logged
              step-wise metrics.
            </p>
          `
      }
    </div>
  `;
}

function renderOverlayChart(metricName, runs) {
  const series = runs
    .map((run, index) => ({
      color:
        OVERLAY_COLORS[
          index % OVERLAY_COLORS.length
        ],
      points:
        run.metrics_over_time?.[metricName] || [],
    }))
    .filter((item) => item.points.length > 0);

  if (!series.length) {
    return `
      <div class="metric-chart-empty">
        ${esc(metricName)}:
        no data in any selected run
      </div>
    `;
  }

  const allValues = series.flatMap((item) =>
    item.points.map((point) => point.value)
  );

  const allSteps = series.flatMap((item) =>
    item.points.map((point) => point.step ?? 0)
  );

  const minimumValue = Math.min(...allValues);
  const maximumValue = Math.max(...allValues);

  const valueSpan =
    maximumValue - minimumValue ||
    Math.abs(maximumValue || 1) * 0.1 ||
    1;

  const minimumStep = Math.min(...allSteps);
  const maximumStep = Math.max(...allSteps);
  const stepSpan = maximumStep - minimumStep || 1;

  const width = 640;
  const height = 200;
  const leftPadding = 50;
  const rightPadding = 12;
  const topPadding = 12;
  const bottomPadding = 24;

  const toCoordinates = (step, value) => ({
    x:
      leftPadding +
      ((step - minimumStep) / stepSpan) *
        (width - leftPadding - rightPadding),
    y:
      height -
      bottomPadding -
      ((value - minimumValue) / valueSpan) *
        (height - topPadding - bottomPadding),
  });

  const yTicks = [0, 0.5, 1]
    .map((fraction) => {
      const value =
        minimumValue + valueSpan * fraction;

      const y =
        height -
        bottomPadding -
        fraction *
          (height - topPadding - bottomPadding);

      return `
        <line
          x1="${leftPadding}"
          x2="${width - rightPadding}"
          y1="${y}"
          y2="${y}"
          stroke="rgba(109,125,147,.16)"
          stroke-width="1"
        />

        <text
          x="${leftPadding - 8}"
          y="${y + 3}"
          text-anchor="end"
          font-size="9.5"
          fill="var(--ink-faint)"
          font-family="var(--font-mono)"
        >
          ${fmtNum(value, 3)}
        </text>
      `;
    })
    .join("");

  const lines = series
    .map((item) => {
      const sortedPoints = [...item.points].sort(
        (left, right) =>
          (left.step ?? 0) -
          (right.step ?? 0)
      );

      const points = sortedPoints
        .map((point) => {
          const coordinate = toCoordinates(
            point.step ?? 0,
            point.value
          );

          return `${coordinate.x.toFixed(
            1
          )},${coordinate.y.toFixed(1)}`;
        })
        .join(" ");

      return `
        <polyline
          points="${points}"
          fill="none"
          stroke="${item.color}"
          stroke-width="2"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
      `;
    })
    .join("");

  return `
    <div class="metric-chart">
      <div class="metric-chart-head">
        <span class="metric-chart-name">
          ${esc(metricName)}
        </span>
      </div>

      <svg
        viewBox="0 0 ${width} ${height}"
        style="width:100%; height:${height}px; display:block;"
      >
        ${yTicks}
        ${lines}
      </svg>

      <div class="metric-chart-xaxis">
        <span>step ${minimumStep}</span>
        <span>step ${maximumStep}</span>
      </div>
    </div>
  `;
}

// -------------------- screen: global failures --------------------

async function renderFailuresScreen() {
  app.innerHTML = `
    <p class="loading">
      loading failures&hellip;
    </p>
  `;

  let failures;

  try {
    failures = await api("/failures");
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  if (!failures.length) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">
          no failures recorded
        </p>

        <p>
          Every run so far has completed successfully.
        </p>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Failures</h1>

    <div class="panel">
      <table class="runs-table">
        <thead>
          <tr>
            <th>run</th>
            <th>project</th>
            <th>rule</th>
            <th>message</th>
          </tr>
        </thead>

        <tbody>
          ${failures
            .map(
              (failure) => `
                <tr>
                  <td>
                    <a
                      href="#/failure/${encodeURIComponent(
                        failure.run_id
                      )}"
                    >
                      ${esc(failure.run_id)}
                    </a>
                  </td>

                  <td>${esc(failure.project)}</td>
                  <td>${esc(failure.rule)}</td>

                  <td>
                    ${esc(
                      (failure.message || "").slice(
                        0,
                        70
                      )
                    )}
                  </td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

// -------------------- screen: campaigns --------------------

async function renderCampaignsScreen() {
  app.innerHTML = `
    <p class="loading">
      loading campaigns&hellip;
    </p>
  `;

  let campaigns;

  try {
    campaigns = await api("/campaigns");
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  if (!campaigns.length) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">
          no recovery campaigns yet
        </p>

        <p>
          Prepare one without GPU compute, review the
          sealed plan, and then run it explicitly from
          the CLI.
        </p>

        <code>
          watcher prepare-recovery RUN_ID
          --entrypoint train:main
          --out recovery-plan.json
        </code>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="eyebrow">
      watcherml / recovery
    </p>

    <h1 class="page-title">
      Recovery campaigns
    </h1>

    <p class="page-subtitle">
      Read-only records of bounded proposals,
      isolated trials, and independent confirmation.
    </p>

    <div class="panel">
      <div class="runs-table-wrapper">
        <table class="runs-table">
          <thead>
            <tr>
              <th>campaign</th>
              <th>project</th>
              <th>status</th>
              <th>verification</th>
              <th>phases</th>
              <th>stopped reason</th>
            </tr>
          </thead>

          <tbody>
            ${campaigns
              .map(
                (campaign) => `
                  <tr>
                    <td>
                      <a
                        href="#/campaign/${encodeURIComponent(
                          campaign.campaign_id
                        )}"
                      >
                        ${esc(campaign.campaign_id)}
                      </a>
                    </td>

                    <td>${esc(campaign.project)}</td>
                    <td>${badge(campaign.status)}</td>

                    <td>
                      ${verificationBadge(
                        campaign.verification_status
                      )}
                    </td>

                    <td>
                      ${
                        campaign.phase_counts?.probe || 0
                      }
                      probe ·
                      ${
                        campaign.phase_counts?.full || 0
                      }
                      full ·
                      ${
                        campaign.phase_counts
                          ?.confirmation || 0
                      }
                      confirmation
                    </td>

                    <td>
                      ${esc(
                        campaign.stopped_reason ||
                          "running"
                      )}
                    </td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

async function renderCampaignScreen(campaignId) {
  app.innerHTML = `
    <p class="loading">
      loading campaign&hellip;
    </p>
  `;

  let campaign;

  try {
    campaign = await api(
      `/campaigns/${encodeURIComponent(
        campaignId
      )}`
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const trials = Array.isArray(campaign.trials)
    ? campaign.trials
    : [];

  const proposals = Array.isArray(
    campaign.proposals
  )
    ? campaign.proposals
    : [];

  const verifications = Array.isArray(
    campaign.verifications
  )
    ? campaign.verifications
    : [];

  const contract = campaign.contract || {};
  const usage = campaign.usage || {};
  const phaseCounts = campaign.phase_counts || {};

  const active =
    campaign.status === "running" ||
    (
      !campaign.ended_at &&
      campaign.verification_status === "pending"
    );

  const verified = campaign.verified === true;

  const maximumTrials = firstFinite(
    contract.max_trials,
    usage.max_trials
  );

  const maximumGpuSeconds = firstFinite(
    contract.max_gpu_seconds,
    usage.max_gpu_seconds
  );

  const gpuSeconds = firstFinite(
    usage.gpu_seconds,
    usage.total_gpu_seconds,
    usage.used_gpu_seconds
  );

  const elapsedSeconds = firstFinite(
    usage.elapsed_seconds,
    usage.wall_seconds
  );

  const rankingRows = Array.isArray(
    campaign.ranking
  )
    ? campaign.ranking
    : campaign.ranking?.assessments ||
      campaign.ranking?.candidates ||
      [];

  const proposalRows = proposals
    .map((proposalRecord) => {
      const proposal =
        proposalRecord.proposal || {};

      const configPatch =
        proposal.config_patch ||
        proposal.patch ||
        proposal.changes ||
        {};

      const environmentPatch =
        proposal.environment_patch || {};

      const serializedChanges = Array.isArray(
        proposal.changes
      )
        ? proposal.changes
            .map(
              (change) =>
                `${change.capability_id} ` +
                `${change.operation} ` +
                `${change.proposed_value}`
            )
            .join(" · ")
        : formatPatch(configPatch);

      const changeText =
        [
          serializedChanges,
          Object.keys(environmentPatch).length
            ? `environment: ${formatPatch(
                environmentPatch
              )}`
            : "",
        ]
          .filter(
            (value, index) =>
              value &&
              (
                index > 0 ||
                value !== "No configuration change"
              )
          )
          .join(" · ") ||
        "No executable change";

      return `
        <tr>
          <td>
            <code>
              ${esc(proposalRecord.proposal_id)}
            </code>
          </td>

          <td>
            ${esc(proposalRecord.policy_rule)}
          </td>

          <td>
            ${esc(
              (
                proposalRecord.authorization_mode ||
                "unknown"
              ).replaceAll("_", " ")
            )}
          </td>

          <td>
            ${badge(proposalRecord.state)}
          </td>

          <td>${esc(changeText)}</td>

          <td>
            ${
              proposalRecord.skip_code
                ? `
                  <strong>
                    ${esc(proposalRecord.skip_code)}
                  </strong>:
                  ${esc(
                    proposalRecord.skip_reason ||
                      "skipped"
                  )}
                `
                : esc(
                    proposalRecord.rationale || ""
                  )
            }
          </td>
        </tr>
      `;
    })
    .join("");

  const trialRows = trials
    .map(
      (trial) => `
        <tr>
          <td>${esc(trial.phase)}</td>

          <td>
            <a
              href="#/run/${encodeURIComponent(
                trial.run_id
              )}"
            >
              ${esc(trial.run_id)}
            </a>

            <br />

            <small>
              ${esc(trial.candidate_id)}
            </small>
          </td>

          <td>
            ${badge(trial.status)}

            ${
              trial.failure_class
                ? `
                  <br />
                  <small>
                    ${esc(trial.failure_class)}
                  </small>
                `
                : ""
            }
          </td>

          <td>
            ${esc(
              formatPatch(trial.config_patch)
            )}

            ${
              Object.keys(
                trial.environment_patch || {}
              ).length
                ? `
                  <br />
                  <small>
                    env:
                    ${esc(
                      formatPatch(
                        trial.environment_patch
                      )
                    )}
                  </small>
                `
                : ""
            }
          </td>

          <td>
            ${trial.progress_steps ?? "&mdash;"}
          </td>

          <td>
            ${
              trial.peak_vram_gib !== null &&
              trial.peak_vram_gib !== undefined
                ? `${fmtNum(
                    trial.peak_vram_gib,
                    2
                  )} GiB`
                : "&mdash;"
            }
          </td>

          <td>
            ${fmtDuration(trial.duration_seconds)}
          </td>

          <td>
            ${
              trial.verified
                ? verificationBadge("verified")
                : "&mdash;"
            }
          </td>
        </tr>
      `
    )
    .join("");

  const verificationCards = verifications
    .map((verification) => {
      const report = verification.report || {};

      const checks = Array.isArray(report.checks)
        ? report.checks
        : [];

      const confirmationRuns = (
        verification.confirmation_run_ids || []
      )
        .map(
          (runId) => `
            <a href="#/run/${encodeURIComponent(runId)}">
              ${esc(runId)}
            </a>
          `
        )
        .join(" · ");

      const checkRows = checks
        .map(
          (check) => `
            <div class="guardrail-item">
              <span class="guardrail-icon">
                ${
                  check.outcome === "pass"
                    ? "✓"
                    : check.outcome === "missing"
                      ? "?"
                      : "×"
                }
              </span>

              <span>
                <strong>
                  ${esc(
                    check.code ||
                      "verification_check"
                  )}
                </strong>

                ${
                  check.run_id
                    ? `
                      —
                      <a
                        href="#/run/${encodeURIComponent(
                          check.run_id
                        )}"
                      >
                        ${esc(check.run_id)}
                      </a>
                    `
                    : ""
                }

                :
                ${esc(
                  check.message ||
                    check.outcome ||
                    "recorded"
                )}
              </span>
            </div>
          `
        )
        .join("");

      return `
        <article class="panel m0">
          <h3 class="section-title">
            ${esc(verification.candidate_id)}

            ${verificationBadge(
              verification.verified
                ? "verified"
                : "not_verified"
            )}
          </h3>

          <div class="field">
            <span class="field-label">
              confirmation runs
            </span>

            <span class="field-value">
              ${confirmationRuns || "none"}
            </span>
          </div>

          ${
            checkRows
              ? `
                <div class="guardrail-list">
                  ${checkRows}
                </div>
              `
              : `
                <p class="ai-empty">
                  The complete verifier report remains
                  available in the campaign artifact.
                </p>
              `
          }
        </article>
      `;
    })
    .join("");

  const rankingHtml = rankingRows.length
    ? `
      <table class="runs-table">
        <thead>
          <tr>
            <th>candidate</th>
            <th>recorded ranking fields</th>
          </tr>
        </thead>

        <tbody>
          ${rankingRows
            .map(
              (ranking) => `
                <tr>
                  <td>
                    ${esc(
                      ranking.candidate_id ||
                        ranking.proposal_id ||
                        "candidate"
                    )}
                  </td>

                  <td>
                    <code>
                      ${esc(JSON.stringify(ranking))}
                    </code>
                  </td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    `
    : `
      <p class="ai-empty">
        No provisional ranking was persisted.
      </p>
    `;

  app.innerHTML = `
    <div class="campaign-workspace">
      <div class="campaign-windowbar">
        <div
          class="window-dots"
          aria-hidden="true"
        >
          <i></i>
          <i></i>
          <i></i>
        </div>

        <div class="window-title">
          campaign / ${esc(campaignId)}
        </div>

        <div
          class="agent-state ${
            active ? "" : "stopped"
          }"
        >
          ${
            active
              ? "campaign running"
              : verified
                ? "verified recovery"
                : "campaign stopped"
          }
        </div>
      </div>

      <div class="campaign-body">
        <section class="campaign-hero">
          <p class="eyebrow">
            deterministic OOM recovery
          </p>

          <h1 class="page-title">
            Campaign audit trail
          </h1>

          <p class="page-subtitle">
            ${esc(campaign.project)}
            · source failure

            <a
              href="#/failure/${encodeURIComponent(
                campaign.source_run_id
              )}"
            >
              ${esc(campaign.source_run_id)}
            </a>
          </p>

          <div class="campaign-actions">
            ${
              campaign.source_run_id
                ? `
                  <a
                    href="#/run/${encodeURIComponent(
                      campaign.source_run_id
                    )}"
                  >
                    <button class="ghost">
                      View source run
                    </button>
                  </a>
                `
                : ""
            }

            <button
              class="ghost icon-button"
              id="refresh-campaign"
            >
              Refresh
            </button>

            <button
              class="ghost"
              data-copy="${esc(campaignId)}"
              data-copy-label="Campaign ID copied"
            >
              Copy campaign ID
            </button>

            ${
              campaign.artifact?.available
                ? `
                  <a
                    href="${esc(
                      campaign.artifact.download_url
                    )}"
                  >
                    <button>
                      Download recovery artifact
                    </button>
                  </a>
                `
                : ""
            }
          </div>
        </section>

        ${
          verified
            ? `
              <aside
                class="verified-fix-toast"
                aria-label="Verified recovery"
              >
                <div
                  class="verified-icon"
                  aria-hidden="true"
                >
                  ↳
                </div>

                <div>
                  <div class="verified-title">
                    Verified recovery
                  </div>

                  <div class="verified-copy">
                    ${esc(
                      campaign.verified_candidate_id
                    )}
                    passed
                    ${
                      (
                        campaign.verified_run_ids ||
                        []
                      ).length
                    }
                    independent confirmation run${
                      (
                        campaign.verified_run_ids ||
                        []
                      ).length === 1
                        ? ""
                        : "s"
                    }.
                  </div>
                </div>
              </aside>
            `
            : ""
        }

        <section
          class="campaign-stat-strip"
          aria-label="Campaign summary"
        >
          <div class="campaign-stat">
            <div class="campaign-stat-label">
              Trials
            </div>

            <div class="campaign-stat-value">
              ${campaign.trial_count}

              ${
                maximumTrials !== null
                  ? ` / ${maximumTrials}`
                  : ""
              }
            </div>
          </div>

          <div class="campaign-stat">
            <div class="campaign-stat-label">
              Phase split
            </div>

            <div
              class="campaign-stat-value"
              style="font-size:15px;"
            >
              ${phaseCounts.probe || 0}P ·
              ${phaseCounts.full || 0}F ·
              ${phaseCounts.confirmation || 0}C
            </div>
          </div>

          <div class="campaign-stat">
            <div class="campaign-stat-label">
              GPU time
            </div>

            <div class="campaign-stat-value">
              ${formatGpuTime(gpuSeconds)}

              ${
                maximumGpuSeconds !== null
                  ? ` / ${formatGpuTime(
                      maximumGpuSeconds
                    )}`
                  : ""
              }
            </div>
          </div>

          <div class="campaign-stat">
            <div class="campaign-stat-label">
              Verification
            </div>

            <div class="campaign-stat-value">
              ${verificationBadge(
                campaign.verification_status
              )}
            </div>
          </div>
        </section>

        <section class="campaign-primary-grid">
          <article class="campaign-panel">
            <header class="campaign-panel-header">
              <span>Bounded execution phases</span>
              ${provenance("isolated")}
            </header>

            <div class="campaign-panel-body">
              <div
                class="reasoning-step campaign-reasoning"
              >
                <span class="reasoning-num">01</span>

                <span class="reasoning-text">
                  Policy proposals

                  <span class="reasoning-meta">
                    ${proposals.length} recorded ·
                    ${
                      campaign.skipped_proposals
                        ?.length || 0
                    }
                    skipped
                  </span>
                </span>
              </div>

              <div
                class="reasoning-step campaign-reasoning"
              >
                <span class="reasoning-num">02</span>

                <span class="reasoning-text">
                  Short probes

                  <span class="reasoning-meta">
                    ${phaseCounts.probe || 0}
                    fresh subprocesses ·
                    ${
                      campaign.probe_survivor_ids
                        ?.length || 0
                    }
                    survivors
                  </span>
                </span>
              </div>

              <div
                class="reasoning-step campaign-reasoning"
              >
                <span class="reasoning-num">03</span>

                <span class="reasoning-text">
                  Full trials

                  <span class="reasoning-meta">
                    ${phaseCounts.full || 0}
                    candidates evaluated against
                    the contract
                  </span>
                </span>
              </div>

              <div
                class="reasoning-step campaign-reasoning ${
                  verified ? "active" : ""
                }"
              >
                <span class="reasoning-num">04</span>

                <span class="reasoning-text">
                  Independent confirmation

                  <span class="reasoning-meta">
                    ${
                      phaseCounts.confirmation || 0
                    }
                    runs ·
                    ${
                      verified
                        ? "recovery verified"
                        : "no verified recovery"
                    }
                  </span>
                </span>
              </div>
            </div>
          </article>

          <article class="campaign-panel">
            <header class="campaign-panel-header">
              <span>Evidence integrity</span>
              ${provenance("calculated")}
            </header>

            <div class="campaign-panel-body">
              <div class="field">
                <span class="field-label">
                  contract digest
                </span>

                <span class="field-value">
                  <code
                    title="${esc(
                      campaign.contract_digest
                    )}"
                  >
                    ${shortDigest(
                      campaign.contract_digest
                    )}
                  </code>
                </span>
              </div>

              <div class="field">
                <span class="field-label">
                  preparation digest
                </span>

                <span class="field-value">
                  <code
                    title="${esc(
                      campaign.preparation_digest
                    )}"
                  >
                    ${shortDigest(
                      campaign.preparation_digest
                    )}
                  </code>
                </span>
              </div>

              <div class="field">
                <span class="field-label">
                  report digest
                </span>

                <span class="field-value">
                  <code
                    title="${esc(
                      campaign.report_digest
                    )}"
                  >
                    ${shortDigest(
                      campaign.report_digest
                    )}
                  </code>
                </span>
              </div>

              <div class="field">
                <span class="field-label">
                  artifact
                </span>

                <span class="field-value">
                  ${
                    campaign.artifact?.available
                      ? `
                        ${fmtBytes(
                          campaign.artifact.size_bytes
                        )}
                        ·
                        <code>
                          ${shortDigest(
                            campaign.artifact.checksum
                          )}
                        </code>
                      `
                      : "not available yet"
                  }
                </span>
              </div>
            </div>
          </article>
        </section>

        <section class="panel">
          <h2 class="section-title">
            Policy proposals
            ${provenance("deterministic")}
          </h2>

          <div class="runs-table-wrapper">
            <table class="runs-table">
              <thead>
                <tr>
                  <th>proposal</th>
                  <th>rule</th>
                  <th>authorization</th>
                  <th>state</th>
                  <th>bounded change</th>
                  <th>rationale / skip</th>
                </tr>
              </thead>

              <tbody>
                ${
                  proposalRows ||
                  `
                    <tr>
                      <td colspan="6">
                        No proposals were recorded.
                      </td>
                    </tr>
                  `
                }
              </tbody>
            </table>
          </div>
        </section>

        <section class="campaign-trials-panel">
          <header class="campaign-panel-header">
            <span>Isolated trial evidence</span>
            ${provenance("isolated")}
          </header>

          <div class="runs-table-wrapper">
            <table class="runs-table">
              <thead>
                <tr>
                  <th>phase</th>
                  <th>run / candidate</th>
                  <th>status</th>
                  <th>intervention</th>
                  <th>steps</th>
                  <th>peak VRAM</th>
                  <th>duration</th>
                  <th>proof</th>
                </tr>
              </thead>

              <tbody>
                ${
                  trialRows ||
                  `
                    <tr>
                      <td
                        colspan="8"
                        class="text-muted"
                      >
                        No trials have been recorded yet.
                      </td>
                    </tr>
                  `
                }
              </tbody>
            </table>
          </div>
        </section>

        <section class="panel">
          <h2 class="section-title">
            Provisional ranking
            ${provenance("provisional")}
          </h2>

          <p>
            Ranking chooses which completed candidate
            is worth confirming next. It is not a
            recovery verdict and never sets
            verifier-owned state.
          </p>

          ${rankingHtml}
        </section>

        <section>
          <h2 class="section-title">
            Independent verification

            ${provenance(
              verified ? "verified" : "calculated"
            )}
          </h2>

          <div class="campaign-support-grid">
            ${
              verificationCards ||
              `
                <article class="panel m0">
                  <p>
                    No confirmation report has been
                    recorded. A successful probe or
                    full trial is still only a
                    candidate.
                  </p>
                </article>
              `
            }
          </div>
        </section>

        <section class="campaign-support-grid">
          <article class="panel m0">
            <h2 class="section-title">
              Sealed contract
            </h2>

            ${jsonBlock(contract)}
          </article>

          <article class="panel m0">
            <h2 class="section-title">
              Campaign boundary
            </h2>

            <div class="field">
              <span class="field-label">
                status
              </span>

              <span class="field-value">
                ${badge(campaign.status)}
              </span>
            </div>

            <div class="field">
              <span class="field-label">
                stopped reason
              </span>

              <span class="field-value">
                ${esc(
                  campaign.stopped_reason ||
                    "running"
                )}
              </span>
            </div>

            <div class="field">
              <span class="field-label">
                wall time
              </span>

              <span class="field-value">
                ${fmtDuration(elapsedSeconds)}
              </span>
            </div>

            <div class="field">
              <span class="field-label">
                verified candidate
              </span>

              <span class="field-value">
                ${esc(
                  campaign.verified_candidate_id
                )}
              </span>
            </div>

            <p class="page-subtitle">
              Inspect from the terminal:

              <code>
                watcher recovery ${esc(campaignId)}
              </code>
            </p>
          </article>
        </section>
      </div>
    </div>
  `;

  document
    .getElementById("refresh-campaign")
    ?.addEventListener("click", () => {
      notify("Campaign refreshed");
      renderCampaignScreen(campaignId);
    });
}

// -------------------- resolution memory --------------------

async function renderMemoryScreen() {
  app.innerHTML = `
    <p class="loading">
      loading resolution memory&hellip;
    </p>
  `;

  let signatures;

  try {
    signatures = await api("/memory");
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  if (!signatures.length) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">
          no resolution history yet
        </p>

        <p>
          This builds automatically as recovery
          campaigns run. There is nothing to configure.
        </p>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="eyebrow">watcherml</p>

    <h1 class="page-title">
      Resolution memory
      ${provenance("calculated")}
    </h1>

    <p class="page-subtitle">
      Historical full-trial outcomes and
      verifier-backed recoveries are reported
      separately. Completion is not proof.
    </p>

    ${signatures
      .map((signature) => {
        const completionRate =
          signature.success_rate || 0;

        const verificationRate =
          signature.verification_rate || 0;

        const completionClass =
          completionRate >= 0.7
            ? "good"
            : completionRate <= 0.3
              ? "bad"
              : "mixed";

        return `
          <div class="signature-card">
            <div class="signature-title">
              ${esc(signature.failure_class)}
              &mdash;
              changing
              ${esc(
                (signature.patch_keys || []).join(
                  ", "
                ) || "(no keys)"
              )}
            </div>

            <div class="resolution-row">
              <span>
                ${(signature.example_patches || [])
                  .map((patch) =>
                    esc(JSON.stringify(patch))
                  )
                  .join(" / ")}
              </span>

              <span
                class="resolution-rate ${completionClass}"
              >
                ${signature.successes}/${
                  signature.attempts
                }
                full trials completed
                (${(completionRate * 100).toFixed(0)}%)
              </span>
            </div>

            <div class="resolution-row">
              <span>
                ${provenance("verified")}
                independent confirmation
              </span>

              <span
                class="resolution-rate ${
                  verificationRate > 0
                    ? "good"
                    : "mixed"
                }"
              >
                ${
                  signature.verified_recoveries || 0
                }/${signature.attempts}
                verified
                (${(verificationRate * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
        `;
      })
      .join("")}
  `;
}

// -------------------- settings --------------------

async function renderSettingsScreen() {
  app.innerHTML = `
    <p class="loading">
      loading settings&hellip;
    </p>
  `;

  let settings;

  try {
    settings = await api("/settings");
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const gpu = settings.gpu || {};
  const gpus = gpu.gpus || [];

  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Settings</h1>

    <div class="panel">
      <h2 class="section-title">
        Local storage
      </h2>

      <div class="field">
        <span class="field-label">
          data directory
        </span>

        <span class="field-value">
          ${esc(settings.data_directory)}
        </span>
      </div>

      <div class="field">
        <span class="field-label">
          database
        </span>

        <span class="field-value">
          ${esc(settings.database_path)}
        </span>
      </div>

      <div class="field">
        <span class="field-label">
          schema version
        </span>

        <span class="field-value">
          ${esc(settings.storage_schema_version)}
        </span>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">
        Recovery safety boundary
      </h2>

      <div class="field">
        <span class="field-label">
          execution surface
        </span>

        <span class="field-value">
          ${esc(
            (
              settings.recovery_execution_surface ||
              "sdk_or_cli"
            ).replaceAll("_", " + ")
          )}
        </span>
      </div>

      <div class="field">
        <span class="field-label">
          trial isolation
        </span>

        <span class="field-value">
          ${esc(
            (
              settings.trial_isolation ||
              "fresh_subprocess"
            ).replaceAll("_", " ")
          )}
        </span>
      </div>

      <div class="field">
        <span class="field-label">
          browser mutations
        </span>

        <span class="field-value">
          ${
            settings.web_recovery_mutations_enabled
              ? badge("enabled")
              : "disabled (read-only audit UI)"
          }
        </span>
      </div>

      <div class="field">
        <span class="field-label">
          LLM required
        </span>

        <span class="field-value">
          ${settings.llm_required ? "yes" : "no"}
        </span>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">GPU</h2>

      <div class="field">
        <span class="field-label">
          detected
        </span>

        <span class="field-value">
          ${gpu.available ? "yes" : "no"}
        </span>
      </div>

      ${gpus
        .map(
          (device) => `
            <div class="field">
              <span class="field-label">
                ${esc(device.name)}
              </span>

              <span class="field-value">
                ${esc(device.memory_total_mib)}
                MiB total
              </span>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function errorState(error) {
  return `
    <div class="empty-state">
      <p class="eyebrow">error</p>
      <p>${esc(error.message)}</p>
    </div>
  `;
}