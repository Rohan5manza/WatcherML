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

  // Keep older index.html installations aligned with the v1 product vocabulary.
  const setNavCopy = (route, label, title = label) => {
    const link = document.querySelector(`#sidebar-nav [data-route="${route}"]`);
    const text = link?.querySelector(".nav-label");
    if (text) text.textContent = label;
    if (link) link.title = title;
  };

  const setCommandCopy = (href, label, description) => {
    const item = document.querySelector(`[data-command-item][href="${href}"]`);
    const labelNode = [...(item?.childNodes || [])]
      .find((node) => node.nodeType === 3);

    if (labelNode) labelNode.nodeValue = `${label} `;

    const detail = item?.querySelector("span");
    if (detail) detail.textContent = description;
  };

  setNavCopy(
    "campaigns",
    "Recoveries",
    "Controlled OOM recovery trials",
  );
  setNavCopy(
    "memory",
    "Verified history",
    "Verified recovery history",
  );

  setCommandCopy(
    "#/campaigns",
    "Recoveries",
    "Review bounded, isolated OOM trials",
  );
  setCommandCopy(
    "#/memory",
    "Verified history",
    "Confirmed recovery evidence only",
  );
  setCommandCopy(
    "#/settings",
    "Settings",
    "Local storage, runtime, and GPU",
  );

  const brandKicker = document.querySelector(".brand-kicker");
  if (brandKicker) brandKicker.textContent = "Recovery console";

  if (search) {
    search.placeholder = "Go to runs, failures, recoveries…";
  }

  const closePalette = () => {
    if (!palette) return;

    palette.hidden = true;

    if (search) search.value = "";

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
        item.hidden = Boolean(
          query &&
          !item.textContent.toLowerCase().includes(query)
        );
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

    mobileButton.setAttribute(
      "aria-expanded",
      String(Boolean(isOpen)),
    );
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
        copyButton.dataset.copy || "",
      );

      notify(
        copyButton.dataset.copyLabel ||
        "Copied to clipboard",
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
    const number = Number(value);

    if (
      value !== null &&
      value !== undefined &&
      Number.isFinite(number)
    ) {
      return number;
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
      const formattedValue =
        typeof value === "boolean"
          ? value
            ? "enabled"
            : "disabled"
          : value;

      return `${key.replaceAll("_", " ")} → ${formattedValue}`;
    })
    .join(" · ");
}

async function api(path, options) {
  const response = await fetch("/api" + path, options);

  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({}));

    throw new Error(
      body.detail || `Request failed: ${response.status}`,
    );
  }

  return response.json();
}

function esc(value) {
  if (value === null || value === undefined) {
    return "&mdash;";
  }

  return String(value).replace(
    /[&<>"']/g,
    (character) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[character],
  );
}

function fmtNum(value, digits = 4) {
  if (value === null || value === undefined) {
    return "&mdash;";
  }

  if (typeof value !== "number") {
    return esc(value);
  }

  return Number.isInteger(value)
    ? String(value)
    : value
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
  if (!timestamp) return "&mdash;";

  const date = new Date(timestamp * 1000);

  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function badge(status) {
  const badgeClass =
    status === "success"
      ? "success"
      : status === "failed"
        ? "failed"
        : "running";

  return `
    <span class="badge ${badgeClass}">
      ${esc(status || "running")}
    </span>
  `;
}

function demoBadge(simulated) {
  return simulated
    ? `<span class="pill demo">Simulated OOM</span>`
    : "";
}

function resolvedBadge(resolved) {
  return resolved
    ? `<span class="pill resolved">Resolved</span>`
    : "";
}

function tagPills(tags) {
  if (!tags || !tags.length) return "";

  return tags
    .map((tag) => `<span class="pill tag">${esc(tag)}</span>`)
    .join(" ");
}

function provenance(kind) {
  const labels = {
    captured: "Captured evidence",
    "rule-based": "Deterministic rule",
    policy: "Bounded policy",
    calculated: "Calculated",
    verified: "Verified outcome",
    inconclusive: "Inconclusive",
  };

  return `
    <span class="provenance ${kind}">
      ${labels[kind] || kind}
    </span>
  `;
}

// -----------------------------------------------------------------------------
// Telemetry traces
// -----------------------------------------------------------------------------

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
              ? "This updates live while the run is active."
              : ""
          }
        </div>
      </div>
    `;
  }

  const hasGpu = samples.some(
    (sample) =>
      sample.gpu_util_pct !== null &&
      sample.gpu_util_pct !== undefined,
  );

  const hasCpu = samples.some(
    (sample) =>
      sample.cpu_pct !== null &&
      sample.cpu_pct !== undefined,
  );

  const startTime = samples[0].t;
  const timeSpan = Math.max(
    1,
    samples[samples.length - 1].t - startTime,
  );

  const width = 1000;
  const height = 130;
  const paddingLeft = 34;
  const paddingRight = 10;
  const paddingTop = 8;
  const paddingBottom = 20;

  function toPath(key, color) {
    const points = [];

    samples.forEach((sample) => {
      const value = sample[key];

      if (value === null || value === undefined) return;

      const x =
        paddingLeft +
        ((sample.t - startTime) / timeSpan) *
          (width - paddingLeft - paddingRight);

      const y =
        height -
        paddingBottom -
        (Math.min(100, Math.max(0, value)) / 100) *
          (height - paddingTop - paddingBottom);

      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    });

    if (points.length < 2) return "";

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
        paddingBottom -
        (value / 100) *
          (height - paddingTop - paddingBottom);

      return `
        <line
          x1="${paddingLeft}"
          x2="${width - paddingRight}"
          y1="${y}"
          y2="${y}"
          stroke="rgba(109,125,147,.18)"
          stroke-width="1"
        />
        <text
          x="${paddingLeft - 6}"
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
    const remaining = Math.round(seconds % 60);

    return `${minutes}:${String(remaining).padStart(2, "0")}`;
  };

  const xLabels = `
    <text
      x="${paddingLeft}"
      y="${height - 4}"
      font-size="9"
      fill="var(--ink-faint)"
      font-family="var(--font-mono)"
    >
      0:00
    </text>

    <text
      x="${width - paddingRight}"
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
      sample.gpu_mem_used_mib !== undefined,
  );

  let vramSection = "";

  if (vramSamples.length >= 2) {
    const values = vramSamples.map(
      (sample) => sample.gpu_mem_used_mib,
    );

    const maximum = Math.max(...values) * 1.15 || 1;
    const vramHeight = 44;

    const points = vramSamples
      .map((sample) => {
        const x =
          paddingLeft +
          ((sample.t - startTime) / timeSpan) *
            (width - paddingLeft - paddingRight);

        const y =
          vramHeight -
          (sample.gpu_mem_used_mib / maximum) *
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
        style="width:100%;height:${vramHeight}px;display:block;"
      >
        <polyline
          points="${points}"
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
      sample.disk_read_mbps !== undefined,
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
        "Disk I/O",
      )
    : "";

  const networkSection = samples.some(
    (sample) =>
      sample.net_sent_mbps !== null &&
      sample.net_sent_mbps !== undefined,
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
        "Network I/O",
      )
    : "";

  return `
    <div class="trace">
      <div class="trace-label">
        <span>${containerLabel}</span>
        <span>
          ${samples.length} samples over ${formatElapsed(timeSpan)}
        </span>
      </div>

      ${legend}

      <svg
        viewBox="0 0 ${width} ${height}"
        style="width:100%;height:${height}px;display:block;"
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
  const startTime = samples[0].t;
  const timeSpan = Math.max(
    1,
    samples[samples.length - 1].t - startTime,
  );

  const width = 1000;
  const height = 70;
  const paddingLeft = 46;
  const paddingRight = 10;
  const paddingTop = 6;
  const paddingBottom = 16;

  const allValues = [];

  seriesDefinitions.forEach((series) => {
    samples.forEach((sample) => {
      const value = sample[series.key];

      if (value !== null && value !== undefined) {
        allValues.push(value);
      }
    });
  });

  if (!allValues.length) return "";

  const maximum = Math.max(...allValues, 0.001) * 1.15;

  const paths = seriesDefinitions
    .map((series) => {
      const points = [];

      samples.forEach((sample) => {
        const value = sample[series.key];

        if (value === null || value === undefined) return;

        const x =
          paddingLeft +
          ((sample.t - startTime) / timeSpan) *
            (width - paddingLeft - paddingRight);

        const y =
          height -
          paddingBottom -
          (Math.max(0, value) / maximum) *
            (height - paddingTop - paddingBottom);

        points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
      });

      if (points.length < 2) return "";

      return `
        <polyline
          points="${points.join(" ")}"
          fill="none"
          stroke="${series.color}"
          stroke-width="1.6"
        />
      `;
    })
    .join("");

  const legend = seriesDefinitions
    .map(
      (series) => `
        <span>
          <i style="background:${series.color}"></i>
          ${esc(series.label)}
        </span>
      `,
    )
    .join("");

  return `
    <div class="trace-label" style="margin-top:10px;">
      <span>${esc(title)}</span>
    </div>

    <div class="trace-legend">
      ${legend}
    </div>

    <svg
      viewBox="0 0 ${width} ${height}"
      style="width:100%;height:${height}px;display:block;"
    >
      <text
        x="${paddingLeft - 6}"
        y="${paddingTop + 8}"
        text-anchor="end"
        font-size="9"
        fill="var(--ink-faint)"
        font-family="var(--font-mono)"
      >
        ${fmtNum(maximum, 1)} MB/s
      </text>

      ${paths}
    </svg>
  `;
}

// -----------------------------------------------------------------------------
// Metric charts
// -----------------------------------------------------------------------------

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
      (right.step ?? right.timestamp),
  );

  const values = sorted.map((point) => point.value);

  const steps = sorted.map((point, index) =>
    point.step !== null && point.step !== undefined
      ? point.step
      : index,
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
          Only one value has been logged. A trend line needs
          at least two points.
        </p>
      </div>
    `;
  }

  const width = 620;
  const height = 170;
  const paddingLeft = 50;
  const paddingRight = 12;
  const paddingTop = 12;
  const paddingBottom = 24;

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
      paddingLeft +
      ((steps[index] - minimumStep) / stepSpan) *
        (width - paddingLeft - paddingRight),

    y:
      height -
      paddingBottom -
      ((point.value - minimumValue) / valueSpan) *
        (height - paddingTop - paddingBottom),

    value: point.value,
    step: steps[index],
  }));

  const linePoints = coordinates
    .map(
      (coordinate) =>
        `${coordinate.x.toFixed(1)},${coordinate.y.toFixed(1)}`,
    )
    .join(" ");

  const yTicks = [0, 0.5, 1]
    .map((fraction) => {
      const value = minimumValue + valueSpan * fraction;

      const y =
        height -
        paddingBottom -
        fraction * (height - paddingTop - paddingBottom);

      return `
        <line
          x1="${paddingLeft}"
          x2="${width - paddingRight}"
          y1="${y}"
          y2="${y}"
          stroke="rgba(109,125,147,.16)"
          stroke-width="1"
        />

        <text
          x="${paddingLeft - 8}"
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
      `,
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
        style="width:100%;height:${height}px;display:block;"
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
        METRIC_CHART_COLORS[
          index % METRIC_CHART_COLORS.length
        ],
      ),
    )
    .join("");
}

function renderSparkline(
  values,
  color = "var(--signal-mint)",
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
  const paddingX = 10;
  const paddingY = 22;

  const minimum = Math.min(...values);
  const maximum = Math.max(...values);

  const margin = Math.max(
    (maximum - minimum) * 0.28,
    Math.abs(maximum || 1) * 0.025,
  );

  const low = minimum - margin;
  const high = maximum + margin;
  const span = high - low || 1;

  const coordinates = values.map((value, index) => ({
    x:
      paddingX +
      (index / (values.length - 1)) *
        (width - paddingX * 2),

    y:
      height -
      paddingY -
      ((value - low) / span) *
        (height - paddingY * 2),
  }));

  const points = coordinates
    .map(
      (coordinate) =>
        `${coordinate.x.toFixed(1)},${coordinate.y.toFixed(1)}`,
    )
    .join(" ");

  const last = coordinates[coordinates.length - 1];

  const areaPoints =
    `${paddingX},${height - paddingY} ` +
    `${points} ` +
    `${width - paddingX},${height - paddingY}`;

  const grid = [0.25, 0.5, 0.75]
    .map(
      (fraction) => `
        <line
          x1="${paddingX}"
          x2="${width - paddingX}"
          y1="${
            paddingY +
            (height - paddingY * 2) * fraction
          }"
          y2="${
            paddingY +
            (height - paddingY * 2) * fraction
          }"
          stroke="rgba(109,125,147,.20)"
          stroke-width="1"
        />
      `,
    )
    .join("");

  const gradientId =
    `trial-gradient-${Math.random().toString(36).slice(2)}`;

  return `
    <div class="objective-chart">
      <svg
        viewBox="0 0 ${width} ${height}"
        preserveAspectRatio="none"
        aria-label="Recorded trial metric"
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
          cx="${last.x}"
          cy="${last.y}"
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

// -----------------------------------------------------------------------------
// Router
// -----------------------------------------------------------------------------

const routes = [
  [/^#\/$/, "overview", renderOverviewScreen],

  [
    /^#\/projects$/,
    "projects",
    renderProjectsScreen,
  ],

  [
    /^#\/runs$/,
    "runs",
    () =>
      renderGlobalRunsScreen(
        new URLSearchParams(
          location.hash.split("?")[1],
        ),
      ),
  ],

  [
    /^#\/failures$/,
    "failures",
    renderFailuresScreen,
  ],

  [
    /^#\/campaigns$/,
    "campaigns",
    renderCampaignsScreen,
  ],

  [
    /^#\/memory$/,
    "memory",
    renderMemoryScreen,
  ],

  [
    /^#\/settings$/,
    "settings",
    renderSettingsScreen,
  ],

  [
    /^#\/project\/([^/]+)$/,
    "projects",
    (match) =>
      renderProjectRunsScreen(
        decodeURIComponent(match[1]),
      ),
  ],

  [
    /^#\/run\/([^/]+)$/,
    "runs",
    (match) =>
      renderRunScreen(
        decodeURIComponent(match[1]),
      ),
  ],

  [
    /^#\/failure\/([^/]+)$/,
    "failures",
    (match) =>
      renderFailureScreen(
        decodeURIComponent(match[1]),
      ),
  ],

  [
    /^#\/campaign\/([^/]+)$/,
    "campaigns",
    (match) =>
      renderCampaignScreen(
        decodeURIComponent(match[1]),
      ),
  ],

  [
    /^#\/compare$/,
    "runs",
    () =>
      renderCompareScreen(
        new URLSearchParams(
          location.hash.split("?")[1],
        ),
      ),
  ],

  [
    /^#\/overlay$/,
    "runs",
    () =>
      renderOverlayScreen(
        new URLSearchParams(
          location.hash.split("?")[1],
        ),
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

    if (!match) continue;

    updateActiveNav(navigationKey);

    Promise
      .resolve(handler(match))
      .finally(() => {
        finishRouteProgress();

        app.focus({
          preventScroll: true,
        });

        window.scrollTo({
          top: 0,
          behavior: "instant",
        });
      });

    return;
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
  document
    .querySelectorAll("#sidebar-nav a")
    .forEach((link) => {
      link.classList.toggle(
        "active",
        link.dataset.route === navigationKey,
      );
    });
}

window.addEventListener("hashchange", route);

window.addEventListener("DOMContentLoaded", () => {
  setupGlobalUI();
  route();
});

// -----------------------------------------------------------------------------
// Overview
// -----------------------------------------------------------------------------

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

  const attentionRuns = Array.isArray(
    overview.runs_needing_attention,
  )
    ? overview.runs_needing_attention
    : [];

  const verifiedRecoveries = Array.isArray(
    overview.recent_verified_fixes,
  )
    ? overview.recent_verified_fixes
    : [];

  const oomFailureCount =
    firstFinite(overview.oom_failure_count) ??
    attentionRuns.filter((run) => {
      const category = String(
        run.failure_category || run.rule || "",
      ).toLowerCase();

      return (
        category.includes("out_of_memory") ||
        category.includes("oom")
      );
    }).length;

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

  const attentionRows = attentionRuns
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
      `,
    )
    .join("");

  const verifiedRows = verifiedRecoveries
    .map(
      (campaign) => `
        <tr>
          <td>
            <a href="#/campaign/${encodeURIComponent(
              campaign.campaign_id,
            )}">
              ${esc(campaign.campaign_id)}
            </a>
          </td>

          <td>${esc(campaign.project)}</td>

          <td>
            <a href="#/run/${encodeURIComponent(
              campaign.best_run_id,
            )}">
              ${esc(campaign.best_run_id)}
            </a>
          </td>
        </tr>
      `,
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
        <div class="stat-value ${
          attentionRuns.length ? "red" : ""
        }">
          ${attentionRuns.length}
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">
          Active recovery trials
        </div>
        <div class="stat-value mint">
          ${overview.active_campaign_count ?? 0}
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">GPU</div>
        <div
          class="stat-value"
          style="font-size:15px;"
        >
          ${
            overview.gpu_available
              ? esc(overview.gpu_name)
              : "not detected"
          }
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-label">OOM failures</div>
        <div class="stat-value ${
          oomFailureCount ? "red" : ""
        }">
          ${oomFailureCount}
        </div>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">
        Runs needing attention
      </h2>

      ${
        attentionRows
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
              <tbody>
                ${attentionRows}
              </tbody>
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
        verifiedRows
          ? `
            <table class="runs-table">
              <thead>
                <tr>
                  <th>campaign</th>
                  <th>project</th>
                  <th>verified run</th>
                </tr>
              </thead>
              <tbody>
                ${verifiedRows}
              </tbody>
            </table>
          `
          : `
            <p class="ai-empty">
              No recovery campaign has produced a verified
              recovery yet.
            </p>
          `
      }
    </div>
  `;
}

// -----------------------------------------------------------------------------
// Projects
// -----------------------------------------------------------------------------

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

                <span class="${
                  project.failure_count
                    ? "fail-count"
                    : ""
                }">
                  ${project.failure_count} failed
                </span>
              </div>
            </a>
          `,
        )
        .join("")}
    </div>
  `;
}

// -----------------------------------------------------------------------------
// Run list
// -----------------------------------------------------------------------------

async function renderProjectRunsScreen(project) {
  app.innerHTML = `
    <p class="loading">loading runs&hellip;</p>
  `;

  let runs;

  try {
    runs = await api(
      `/projects/${encodeURIComponent(project)}/runs`,
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  renderRunsTable(
    runs,
    `<a href="#/">projects</a> / ${esc(project)}`,
    project,
  );
}

async function renderGlobalRunsScreen(params) {
  app.innerHTML = `
    <p class="loading">loading runs&hellip;</p>
  `;

  const status = params.get("status") || "";
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

  renderRunsTable(
    runs,
    "watcherml",
    null,
    status,
  );
}

let runsTableSort = {
  key: null,
  direction: 1,
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
  currentStatusFilter,
) {
  const metricNames = [
    ...new Set(
      runs.flatMap((run) =>
        Object.keys(run.final_metrics || {}),
      ),
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
        runsTableSort.key,
      );

      const rightValue = sortValue(
        right,
        runsTableSort.key,
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
        return -1 * runsTableSort.direction;
      }

      if (leftValue > rightValue) {
        return runsTableSort.direction;
      }

      return 0;
    });
  }

  const sortableHeader = (label, key) => {
    const active = runsTableSort.key === key;

    const arrow = active
      ? runsTableSort.direction === 1
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
              Select two or more runs to compare or overlay
              their metrics.
            </span>

            <button id="compare-btn" disabled>
              Compare selected
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
              ${sortableHeader("duration", "duration_seconds")}

              ${metricNames
                .map((metric) =>
                  sortableHeader(
                    metric,
                    `metric:${metric}`,
                  ),
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
                      <a href="#/run/${encodeURIComponent(run.run_id)}">
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
                    <td>${fmtDuration(run.duration_seconds)}</td>

                    ${metricNames
                      .map(
                        (metric) => `
                          <td>
                            ${fmtNum(run.final_metrics[metric])}
                          </td>
                        `,
                      )
                      .join("")}

                    <td>${esc(run.hardware)}</td>

                    <td>
                      ${
                        run.warning_count > 0
                          ? `
                            <span style="color:var(--signal-amber)">
                              ${run.warning_count}
                            </span>
                          `
                          : "0"
                      }
                    </td>
                  </tr>
                `,
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;

  document
    .querySelectorAll(".sortable-th")
    .forEach((header) => {
      header.addEventListener("click", () => {
        const key = header.dataset.sortKey;

        runsTableSort.direction =
          runsTableSort.key === key
            ? -runsTableSort.direction
            : 1;

        runsTableSort.key = key;

        renderRunsTable(
          runs,
          breadcrumbHtml,
          project,
          currentStatusFilter,
        );
      });
    });

  if (!project) return;

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
        (candidate) => candidate.checked,
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

    location.hash =
      "#/overlay?ids=" +
      runIds
        .map(encodeURIComponent)
        .join(",");
  });
}

// -----------------------------------------------------------------------------
// Run detail
// -----------------------------------------------------------------------------

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

  renderRunScreenContent(
    runId,
    run,
    samples,
  );

  const MAX_LIVE_POLLS = 200;
  let pollCount = 0;

  if (run.status !== "running") return;

  activePollTimer = setInterval(async () => {
    pollCount += 1;

    if (pollCount > MAX_LIVE_POLLS) {
      stopActivePoll();
      return;
    }

    let freshRun;
    let freshSamples;

    try {
      [freshRun, freshSamples] = await Promise.all([
        api(`/runs/${encodeURIComponent(runId)}`),
        api(`/runs/${encodeURIComponent(runId)}/samples`),
      ]);
    } catch (_) {
      return;
    }

    renderRunScreenContent(
      runId,
      freshRun,
      freshSamples,
    );

    if (freshRun.status !== "running") {
      stopActivePoll();
    }
  }, 3000);
}

function renderRunScreenContent(
  runId,
  run,
  samples,
) {
  const configRows =
    Object.entries(run.config || {})
      .filter(([key]) => key !== "_simulated")
      .map(
        ([key, value]) => `
          <div class="field">
            <span class="field-label">${esc(key)}</span>
            <span class="field-value">${esc(value)}</span>
          </div>
        `,
      )
      .join("") ||
    `<p class="ai-empty">no config recorded</p>`;

  const failureLink = run.has_failure
    ? `
      <div class="failure-banner">
        <p class="eyebrow">this run failed</p>

        <a href="#/failure/${encodeURIComponent(runId)}">
          <strong>View failure capsule &rarr;</strong>
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
              (warning) => `<li>${esc(warning)}</li>`,
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

  const captureCompleteness = firstFinite(
    run.capture_completeness,
  );

  const capsuleSchema =
    run.capsule_schema_version ||
    run.schema_version ||
    "v1";

  app.innerHTML = `
    <p class="eyebrow">
      <a href="#/">projects</a>
      /
      <a href="#/project/${encodeURIComponent(run.project)}">
        ${esc(run.project)}
      </a>
      / run
    </p>

    <h1 class="page-title">
      <span id="run-title">${esc(run.display_name)}</span>
      ${badge(run.status)}
      ${demoBadge(run.simulated)}
      ${resolvedBadge(run.resolved)}
    </h1>

    <p class="page-subtitle">${esc(run.run_id)}</p>

    <div style="margin-bottom:16px;">
      ${tagPills(run.tags)}
    </div>

    <div class="action-bar">
      <button id="rename-btn">Rename</button>

      <a href="/api/runs/${encodeURIComponent(runId)}/export">
        <button>Export capsule</button>
      </a>

      <a href="/api/runs/${encodeURIComponent(runId)}/metrics.csv">
        <button>Export metrics (CSV)</button>
      </a>

      <button id="resolve-btn">
        ${
          run.resolved
            ? "Mark unresolved"
            : "Mark as resolved"
        }
      </button>
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
          Failure-capture context
          ${provenance("captured")}
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
            capsule_schema
          </span>
          <span class="field-value">
            ${esc(capsuleSchema)}
          </span>
        </div>

        <div class="field">
          <span class="field-label">
            capture_completeness
          </span>
          <span class="field-value">
            ${
              captureCompleteness !== null
                ? `${fmtNum(captureCompleteness)}/10`
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
          : run.display_name,
      );

      if (nextName === null) return;

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
        },
      );

      notify("Run name updated");
      stopActivePoll();
      renderRunScreen(runId);
    });

  document
    .getElementById("resolve-btn")
    .addEventListener("click", async () => {
      let note = null;

      if (!run.resolved) {
        note =
          prompt(
            "Resolution note (optional):",
            "",
          ) || null;
      }

      await api(
        `/runs/${encodeURIComponent(runId)}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            resolved: !run.resolved,
            resolved_note: note,
          }),
        },
      );

      notify(
        run.resolved
          ? "Run marked unresolved"
          : "Run marked resolved",
      );

      stopActivePoll();
      renderRunScreen(runId);
    });
}

function buildTimeline(run) {
  const items = [
    {
      timestamp: run.started_at,
      label: "Run started",
      className: "",
    },
  ];

  for (const warning of run.warnings || []) {
    items.push({
      timestamp: run.started_at,
      label: `Warning: ${warning}`,
      className: "warning",
    });
  }

  if (run.has_failure) {
    items.push({
      timestamp: run.ended_at,
      label: "Run failed",
      className: "failure",
    });
  } else if (run.ended_at) {
    items.push({
      timestamp: run.ended_at,
      label: "Run completed",
      className: "",
    });
  }

  items.sort(
    (left, right) =>
      (left.timestamp || 0) -
      (right.timestamp || 0),
  );

  return (
    items
      .map(
        (item) => `
          <div class="timeline-item ${item.className}">
            <div class="timeline-time">
              ${fmtTimestamp(item.timestamp)}
            </div>

            <div class="timeline-label">
              ${esc(item.label)}
            </div>
          </div>
        `,
      )
      .join("") ||
    `<p class="ai-empty">No timeline events recorded.</p>`
  );
}

// -----------------------------------------------------------------------------
// Failure capsule
// -----------------------------------------------------------------------------

async function renderFailureScreen(runId) {
  app.innerHTML = `
    <p class="loading">loading failure capsule&hellip;</p>
  `;

  let failure;

  try {
    failure = await api(
      `/runs/${encodeURIComponent(runId)}/failure`,
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const diagnosis =
    failure.classification ||
    failure.diagnosis ||
    {};

  const failureRule =
    diagnosis.rule ||
    diagnosis.failure_class ||
    diagnosis.category ||
    "unclassified";

  const actions = (diagnosis.suggested_actions || [])
    .map((action) => `<li>${esc(action)}</li>`)
    .join("");

  const proposedInterventions = (
    failure.proposed_interventions || []
  )
    .map((intervention, index) => {
      const patch =
        intervention.patch ||
        intervention.config_patch ||
        {};

      const label =
        intervention.label ||
        intervention.name ||
        `intervention ${index + 1}`;

      const rationale =
        intervention.rationale ||
        intervention.reason ||
        "bounded policy proposal";

      return `
        <div class="checklist-row">
          <strong>${esc(label)}:</strong>
          ${esc(formatPatch(patch))}
          <br />
          <span class="text-muted">
            ${esc(rationale)}
          </span>
        </div>
      `;
    })
    .join("");

  const evidenceIndex = {};

  (failure.evidence_index || []).forEach((entry) => {
    evidenceIndex[entry.id] = entry.label;
  });

  const evidenceChips = (
    diagnosis.evidence_ids || []
  )
    .map(
      (id) => `
        <span
          class="evidence-chip"
          title="${esc(evidenceIndex[id] || "")}"
        >
          ${esc(id)}
        </span>
      `,
    )
    .join("");

  const evidence = failure.evidence || {};

  const recentMetrics =
    (evidence.recent_metrics || [])
      .map(
        (metric) => `
          <div class="field">
            <span class="field-label">
              ${esc(metric.name)} (step ${metric.step})
            </span>
            <span class="field-value">
              ${fmtNum(metric.value)}
            </span>
          </div>
        `,
      )
      .join("") ||
    `
      <p class="ai-empty">
        no metrics logged before failure
      </p>
    `;

  const evidenceFields =
    Object.entries(evidence)
      .filter(
        ([key, value]) =>
          key !== "recent_metrics" &&
          value !== null &&
          value !== undefined,
      )
      .slice(0, 16)
      .map(([key, value]) => {
        const rendered =
          typeof value === "object"
            ? JSON.stringify(value)
            : value;

        return `
          <div class="field">
            <span class="field-label">
              ${esc(key)}
            </span>
            <span class="field-value">
              ${esc(rendered)}
            </span>
          </div>
        `;
      })
      .join("") ||
    `
      <p class="ai-empty">
        no structured runtime evidence recorded
      </p>
    `;

  const similarFailures = (
    failure.similar_previous_failures || []
  )
    .map(
      (similarFailure) => `
        <li>
          <a href="#/failure/${encodeURIComponent(
            similarFailure.run_id,
          )}">
            ${esc(similarFailure.run_id)}
          </a>
          &mdash;
          ${esc(
            (similarFailure.message || "").slice(0, 80),
          )}
        </li>
      `,
    )
    .join("");

  const comparison =
    failure.nearest_successful_run ||
    failure.comparison_to_last_success;

  let comparisonHtml = `
    <span class="ai-empty">
      no previous successful run to compare against
    </span>
  `;

  if (comparison) {
    const checklist = (comparison.checklist || [])
      .map(
        (item) => `
          <div class="checklist-row">
            <span class="${
              item.matched
                ? "check-yes"
                : "check-no"
            }">
              ${item.matched ? "&check;" : "&times;"}
            </span>

            ${esc(item.label)}
          </div>
        `,
      )
      .join("");

    const similarity = firstFinite(
      comparison.similarity_score,
      comparison.similarity,
    );

    comparisonHtml = `
      <p>
        Nearest successful run:
        <a href="#/run/${encodeURIComponent(comparison.run_id)}">
          ${esc(comparison.run_id)}
        </a>

        ${
          similarity !== null
            ? `&mdash; similarity ${(similarity * 100).toFixed(0)}%`
            : ""
        }
      </p>

      ${checklist}

      <a
        href="#/compare?a=${encodeURIComponent(runId)}&b=${encodeURIComponent(
          comparison.run_id,
        )}"
      >
        Full comparison &rarr;
      </a>
    `;
  }

  const schemaVersion =
    failure.capsule_schema_version ||
    failure.schema_version ||
    "v1";

  const recoveryCommand =
    `watcher recover ${runId} ` +
    "--entrypoint train_trial:run";

  app.innerHTML = `
    <p class="eyebrow">
      <a href="#/run/${encodeURIComponent(runId)}">
        ${esc(failure.display_name)}
      </a>
      / failure
    </p>

    <div class="failure-banner">
      <p class="eyebrow">
        ${esc(failureRule)}
        ${provenance("rule-based")}
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
                runId,
              )}&b=${encodeURIComponent(
                comparison.run_id,
              )}"
            >
              <button>Compare baseline</button>
            </a>
          `
          : ""
      }

      <button
        data-copy="${esc(recoveryCommand)}"
        data-copy-label="Recovery command copied"
      >
        Copy recovery command
      </button>

      <a href="/api/runs/${encodeURIComponent(runId)}/export">
        <button>Export capsule</button>
      </a>

      <button id="resolve-btn">
        ${
          failure.resolved
            ? "Mark unresolved"
            : "Mark as resolved"
        }
      </button>
    </div>

    <div class="panel-grid">
      <div class="panel">
        <h2 class="section-title">
          OOM classification
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
          diagnosis.confidence !== undefined
            ? `
              <p>
                <strong>Rule confidence:</strong>
                ${esc(diagnosis.confidence)}
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
          actions
            ? `
              <ul class="suggested-actions">
                ${actions}
              </ul>
            `
            : ""
        }

        <h2
          class="section-title"
          style="margin-top:16px;"
        >
          Proposed bounded interventions
          ${provenance("policy")}
        </h2>

        ${
          proposedInterventions ||
          `
            <p class="ai-empty">
              No intervention proposal has been recorded.
              The exported capsule remains usable evidence;
              no automatic change was made.
            </p>
          `
        }
      </div>

      <div class="panel">
        <h2 class="section-title">
          Runtime evidence
          ${provenance("captured")}
        </h2>

        ${evidenceFields}

        <h2
          class="section-title"
          style="margin-top:16px;"
        >
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
            : `<p class="ai-empty">none recorded</p>`
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
        Verification boundary
        ${provenance("captured")}
      </h2>

      <p>
        A proposed change is not a fix. WatcherML marks
        recovery as verified only after the original OOM
        is reproduced as a control and the bounded
        intervention passes fresh-process confirmation trials.
      </p>

      <div class="field">
        <span class="field-label">capsule schema</span>
        <span class="field-value">
          ${esc(schemaVersion)}
        </span>
      </div>

      <div class="field">
        <span class="field-label">next command</span>
        <span class="field-value">
          <code>${esc(recoveryCommand)}</code>
        </span>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">Full traceback</h2>
      <div class="traceback">
        ${esc(failure.traceback)}
      </div>
    </div>
  `;

  document
    .getElementById("resolve-btn")
    .addEventListener("click", async () => {
      let note = null;

      if (!failure.resolved) {
        note =
          prompt(
            "Resolution note (optional):",
            "",
          ) || null;
      }

      await api(
        `/runs/${encodeURIComponent(runId)}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            resolved: !failure.resolved,
            resolved_note: note,
          }),
        },
      );

      notify(
        failure.resolved
          ? "Failure marked unresolved"
          : "Failure marked resolved",
      );

      renderFailureScreen(runId);
    });
}

// -----------------------------------------------------------------------------
// Run comparison
// -----------------------------------------------------------------------------

async function renderCompareScreen(params) {
  const firstRunId = params.get("a");
  const secondRunId = params.get("b");

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

  let difference;

  try {
    difference = await api(
      `/compare?a=${encodeURIComponent(firstRunId)}` +
      `&b=${encodeURIComponent(secondRunId)}`,
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const configRows =
    (difference.config_diff || [])
      .map((change) =>
        diffRow(
          change.key,
          change.from,
          change.to,
        ),
      )
      .join("") ||
    `<p class="ai-empty">no config changes</p>`;

  const metricRows = (difference.metric_diff || [])
    .map((metric) => {
      const deltaClass =
        metric.delta === undefined
          ? ""
          : metric.delta >= 0
            ? "diff-delta-up"
            : "diff-delta-down";

      const delta =
        metric.delta !== undefined
          ? ` (${metric.delta >= 0 ? "+" : ""}${fmtNum(metric.delta)})`
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
              ${esc(delta)}
            </span>
          </span>
        </div>
      `;
    })
    .join("");

  const packageDiff =
    difference.package_diff || [];

  const packageRows = packageDiff
    .slice(0, 15)
    .map((change) =>
      diffRow(
        change.package,
        change.from,
        change.to,
      ),
    )
    .join("");

  const packageSection = packageDiff.length
    ? `
      <h2
        class="section-title"
        style="margin-top:16px;"
      >
        Package changes (${packageDiff.length})
      </h2>

      ${packageRows}
    `
    : "";

  const gitDifference = difference.git_diff || {};

  app.innerHTML = `
    <p class="eyebrow">compare</p>

    <h1 class="page-title">
      ${esc(firstRunId)} &rarr; ${esc(secondRunId)}
    </h1>

    <div class="panel">
      <h2 class="section-title">What changed?</h2>

      ${configRows}

      ${
        difference.dataset_changed
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
        gitDifference.commit_changed
          ? diffRow(
              "git commit",
              (gitDifference.commit_a || "").slice(0, 10),
              (gitDifference.commit_b || "").slice(0, 10),
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
        `<p class="ai-empty">no metrics to compare</p>`
      }

      <div class="diff-row">
        <span class="diff-key">exit status</span>
        <span class="diff-from">
          ${badge(difference.exit_status_a)}
        </span>
        <span class="diff-arrow">&rarr;</span>
        <span class="diff-to">
          ${badge(difference.exit_status_b)}
        </span>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">
        Interpretation boundary
        ${provenance("calculated")}
      </h2>

      <p>
        This view reports recorded differences only.
        Correlation between a changed configuration value
        and an outcome is not proof of causation. Use a
        fresh-process controlled trial to verify an intervention.
      </p>
    </div>
  `;
}

function diffRow(key, from, to) {
  return `
    <div class="diff-row">
      <span class="diff-key">${esc(key)}</span>
      <span class="diff-from">${esc(from)}</span>
      <span class="diff-arrow">&rarr;</span>
      <span class="diff-to">${esc(to)}</span>
    </div>
  `;
}

// -----------------------------------------------------------------------------
// Multi-run metric overlay
// -----------------------------------------------------------------------------

const OVERLAY_COLORS = [
  "var(--signal-mint)",
  "var(--signal-cyan)",
  "var(--signal-violet)",
  "var(--signal-amber)",
  "#f472b6",
  "#fb923c",
];

async function renderOverlayScreen(params) {
  const idsParameter = params.get("ids") || "";

  const runIds = idsParameter
    .split(",")
    .filter(Boolean);

  if (runIds.length < 2) {
    app.innerHTML = `
      <div class="empty-state">
        <p class="eyebrow">
          select at least two runs
        </p>

        <p>
          Go to a project's run list, select two or more
          runs, then click "Overlay metrics."
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
        api(`/runs/${encodeURIComponent(runId)}`),
      ),
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const metricNames = [
    ...new Set(
      runs.flatMap((run) =>
        Object.keys(run.metrics_over_time || {}),
      ),
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
                style="background:${
                  OVERLAY_COLORS[
                    index % OVERLAY_COLORS.length
                  ]
                }"
              ></i>

              <a href="#/run/${encodeURIComponent(run.run_id)}">
                ${esc(run.display_name)}
              </a>
            </span>
          `,
        )
        .join("")}
    </div>

    <div class="panel">
      ${
        metricNames.length
          ? metricNames
              .map((metricName) =>
                renderOverlayChart(
                  metricName,
                  runs,
                ),
              )
              .join("")
          : `
            <p class="ai-empty">
              None of the selected runs has logged
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
        (
          run.metrics_over_time &&
          run.metrics_over_time[metricName]
        ) || [],
    }))
    .filter((entry) => entry.points.length > 0);

  if (!series.length) {
    return `
      <div class="metric-chart-empty">
        ${esc(metricName)}:
        no data in any selected run
      </div>
    `;
  }

  const allValues = series.flatMap((entry) =>
    entry.points.map((point) => point.value),
  );

  const allSteps = series.flatMap((entry) =>
    entry.points.map((point) => point.step ?? 0),
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
  const paddingLeft = 50;
  const paddingRight = 12;
  const paddingTop = 12;
  const paddingBottom = 24;

  const toCoordinate = (step, value) => ({
    x:
      paddingLeft +
      ((step - minimumStep) / stepSpan) *
        (width - paddingLeft - paddingRight),

    y:
      height -
      paddingBottom -
      ((value - minimumValue) / valueSpan) *
        (height - paddingTop - paddingBottom),
  });

  const yTicks = [0, 0.5, 1]
    .map((fraction) => {
      const value =
        minimumValue + valueSpan * fraction;

      const y =
        height -
        paddingBottom -
        fraction *
          (height - paddingTop - paddingBottom);

      return `
        <line
          x1="${paddingLeft}"
          x2="${width - paddingRight}"
          y1="${y}"
          y2="${y}"
          stroke="rgba(109,125,147,.16)"
          stroke-width="1"
        />

        <text
          x="${paddingLeft - 8}"
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
    .map((entry) => {
      const sortedPoints = [...entry.points].sort(
        (left, right) =>
          (left.step ?? 0) -
          (right.step ?? 0),
      );

      const points = sortedPoints
        .map((point) => {
          const coordinate = toCoordinate(
            point.step ?? 0,
            point.value,
          );

          return (
            `${coordinate.x.toFixed(1)},` +
            `${coordinate.y.toFixed(1)}`
          );
        })
        .join(" ");

      return `
        <polyline
          points="${points}"
          fill="none"
          stroke="${entry.color}"
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
        style="width:100%;height:${height}px;display:block;"
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

// -----------------------------------------------------------------------------
// Global failures
// -----------------------------------------------------------------------------

async function renderFailuresScreen() {
  app.innerHTML = `
    <p class="loading">loading failures&hellip;</p>
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
        <p class="eyebrow">no failures recorded</p>
        <p>Every run so far has completed successfully.</p>
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
                    <a href="#/failure/${encodeURIComponent(
                      failure.run_id,
                    )}">
                      ${esc(failure.run_id)}
                    </a>
                  </td>

                  <td>${esc(failure.project)}</td>
                  <td>${esc(failure.rule)}</td>

                  <td>
                    ${esc(
                      (failure.message || "").slice(0, 70),
                    )}
                  </td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

// -----------------------------------------------------------------------------
// Controlled recovery campaigns
// -----------------------------------------------------------------------------

async function renderCampaignsScreen() {
  app.innerHTML = `
    <p class="loading">loading campaigns&hellip;</p>
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
          no controlled recovery trials yet
        </p>

        <p>
          Start from a captured OOM:
          <code>
            watcher recover RUN_ID --entrypoint train_trial:run
          </code>
        </p>

        <p class="page-subtitle">
          The runner first reproduces the failure as a
          control, then evaluates bounded interventions
          in fresh processes.
        </p>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="eyebrow">watcherml</p>

    <h1 class="page-title">
      OOM recovery campaigns
    </h1>

    <p class="page-subtitle">
      Controlled trials, explicit budgets, and confirmation
      before verification.
    </p>

    <div class="card-grid">
      ${campaigns
        .map(
          (campaign) => {
            const status =
              campaign.verification_status ||
              campaign.status ||
              "pending";

            const active =
              campaign.status === "active" ||
              campaign.status === "running";

            const trialCount =
              campaign.trial_count ?? 0;

            return `
              <a
                class="card"
                href="#/campaign/${encodeURIComponent(
                  campaign.campaign_id,
                )}"
              >
                <div class="card-title">
                  ${esc(campaign.campaign_id)}
                </div>

                <div class="card-meta">
                  <span>${esc(campaign.project)}</span>

                  <span class="${
                    active ? "" : "fail-count"
                  }">
                    ${esc(status)}
                  </span>

                  <span>
                    ${trialCount}
                    trial${trialCount === 1 ? "" : "s"}
                  </span>
                </div>
              </a>
            `;
          },
        )
        .join("")}
    </div>
  `;
}

async function renderCampaignScreen(campaignId) {
  app.innerHTML = `
    <p class="loading">loading campaign&hellip;</p>
  `;

  let campaign;

  try {
    campaign = await api(
      `/campaigns/${encodeURIComponent(campaignId)}`,
    );
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const trials = Array.isArray(campaign.trials)
    ? campaign.trials
    : [];

  const contract = campaign.contract || {};

  const phaseOf = (trial) =>
    String(
      trial.phase ||
      trial.role ||
      "probe",
    ).toLowerCase();

  const outcomeOf = (trial) =>
    String(
      trial.outcome ||
      trial.status ||
      "pending",
    ).toLowerCase();

  const isConfirmationTrial = (trial) =>
    [
      "confirmation",
      "confirm",
      "verification",
      "verify",
      "full",
    ].includes(phaseOf(trial));

  const trialIsVerified = (trial) =>
    Boolean(
      trial &&
      (
        trial.verified === true ||
        String(
          trial.verification_status || "",
        ).toLowerCase() === "verified" ||
        (
          isConfirmationTrial(trial) &&
          [
            "success",
            "passed",
            "verified",
          ].includes(outcomeOf(trial))
        )
      ),
    );

  const verifiedTrial =
    trials.find(trialIsVerified) || null;

  const candidateTrial =
    trials.find(
      (trial) =>
        trial.run_id === campaign.best_run_id,
    ) ||
    trials
      .filter(
        (trial) =>
          phaseOf(trial) === "probe" &&
          ["success", "passed"].includes(
            outcomeOf(trial),
          ),
      )
      .at(-1) ||
    null;

  const selectedTrial =
    verifiedTrial || candidateTrial;

  const selectedIndex = selectedTrial
    ? trials.indexOf(selectedTrial)
    : -1;

  const campaignVerified = Boolean(
    campaign.verified === true ||
    String(
      campaign.verification_status || "",
    ).toLowerCase() === "verified" ||
    verifiedTrial,
  );

  const recordedMetrics = trials
    .map((trial) =>
      firstFinite(
        trial.peak_vram_gb,
        trial.peak_vram,
        trial.score,
        trial.objective_value,
      ),
    )
    .filter((value) => value !== null);

  const peakVram = firstFinite(
    selectedTrial?.peak_vram_gb,
    selectedTrial?.peak_vram,
    campaign.peak_vram_gb,
    campaign.peak_vram,
    contract.peak_vram_gb,
    contract.max_vram_gb,
  );

  const gpuSeconds = firstFinite(
    campaign.gpu_seconds_used,
    campaign.gpu_time_seconds,
    campaign.total_gpu_seconds,
    contract.gpu_seconds_used,
  );

  const gpuBudgetSeconds = firstFinite(
    contract.max_gpu_seconds,
    contract.gpu_budget_seconds,
    contract.max_gpu_hours !== undefined
      ? Number(contract.max_gpu_hours) * 3600
      : null,
  );

  const budgetPercentage =
    gpuBudgetSeconds && gpuSeconds !== null
      ? Math.min(
          100,
          (gpuSeconds / gpuBudgetSeconds) * 100,
        )
      : null;

  const terminalStatuses = [
    "stopped",
    "completed",
    "verified",
    "recovered",
    "inconclusive",
    "failed",
  ];

  const isActive =
    !campaign.ended_at &&
    !terminalStatuses.includes(
      String(campaign.status || "").toLowerCase(),
    );

  const statusLabel = campaignVerified
    ? "Verified"
    : isActive
      ? "Trial running"
      : candidateTrial
        ? "Awaiting confirmation"
        : "Inconclusive";

  const evidenceSteps = [
    {
      text: `
        Captured the baseline failure signature from
        <strong>
          ${esc(campaign.source_run_id || "source run")}
        </strong>.
      `,
      meta: "deterministic evidence capture",
      active: false,
    },
  ];

  trials.slice(-3).forEach((trial) => {
    const phase = phaseOf(trial);
    const outcome = outcomeOf(trial);

    const passed = [
      "success",
      "passed",
      "verified",
    ].includes(outcome);

    evidenceSteps.push({
      text:
        phase === "control"
          ? `
            Control trial
            <strong>${esc(outcome)}</strong>
            with the original configuration.
          `
          : `
            ${esc(phase)} trial
            <strong>${esc(outcome)}</strong>
            after ${esc(formatPatch(trial.patch))}.
          `,

      meta: trialIsVerified(trial)
        ? "confirmation criteria satisfied"
        : passed
          ? "candidate evidence only"
          : "intervention rejected or inconclusive",

      active: false,
    });
  });

  if (campaignVerified && verifiedTrial) {
    evidenceSteps.push({
      text: `
        <strong>Recovery verified.</strong>
        ${esc(formatPatch(verifiedTrial.patch))}
      `,
      meta: "fresh-process confirmation passed",
      active: true,
    });
  } else if (isActive) {
    evidenceSteps.push({
      text: "Waiting for the next bounded trial result.",
      meta: "no verification claim yet",
      active: true,
    });
  }

  const trialRows = trials
    .map((trial, index) => {
      const phase = phaseOf(trial);
      const outcome = outcomeOf(trial);

      const passed = [
        "success",
        "passed",
        "verified",
      ].includes(outcome);

      let decision = "inconclusive";
      let decisionClass = "rejected";

      if (trialIsVerified(trial)) {
        decision = "verified";
        decisionClass = "best";
      } else if (
        phase === "control" &&
        !passed
      ) {
        decision = "oom reproduced";
        decisionClass = "keep";
      } else if (
        phase === "probe" &&
        passed
      ) {
        decision = "candidate";
        decisionClass = "keep";
      } else if (
        ["failed", "oom", "rejected"].includes(outcome)
      ) {
        decision = "rejected";
      }

      const score = firstFinite(
        trial.score,
        trial.objective_value,
      );

      const resultText =
        trial.result_summary ||
        (
          passed
            ? score !== null
              ? `${
                  contract.goal_metric ||
                  "recorded metric"
                } ${fmtNum(score, 3)}`
              : "Trial completed"
            : outcome
        );

      return `
        <tr>
          <td>
            #${String(index + 1).padStart(2, "0")}
            <span class="text-muted">
              ${esc(phase)}
            </span>
          </td>

          <td class="trial-intervention">
            ${esc(formatPatch(trial.patch))}
          </td>

          <td class="trial-result ${
            passed ? "good" : "bad"
          }">
            ${esc(resultText)}
          </td>

          <td>
            <span class="decision-pill ${decisionClass}">
              ${decision}
            </span>
          </td>
        </tr>
      `;
    })
    .join("");

  const maximumTrials = firstFinite(
    contract.max_trials,
    contract.trial_budget,
    campaign.max_trials,
  );

  const confirmationRuns = firstFinite(
    contract.confirmation_runs,
    contract.required_confirmations,
    campaign.confirmation_runs,
  );

  const maximumRegression = firstFinite(
    contract.max_metric_regression_pct,
    contract.max_regression_pct,
  );

  const minimumHeadroom = firstFinite(
    contract.min_vram_headroom_pct,
    contract.vram_headroom_pct,
  );

  const permissions = contract.permissions || {};

  const allowedAutomaticChanges = Object.entries(
    permissions,
  )
    .filter(
      ([, value]) =>
        value === "automatic" ||
        value === true,
    )
    .map(([key]) => key.replaceAll("_", " "));

  const stoppedReason =
    campaign.stopped_reason ||
    (
      isActive
        ? "Campaign is still running within its configured guardrails."
        : campaignVerified
          ? "Confirmation criteria satisfied."
          : "No intervention met every acceptance criterion."
    );

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

        <div class="agent-state ${
          isActive ? "" : "stopped"
        }">
          ${esc(statusLabel)}
        </div>
      </div>

      <div class="campaign-body">
        <section class="campaign-hero">
          <p class="eyebrow autopilot-label">
            Recovery campaign
          </p>

          <h1 class="page-title">
            Verify an OOM recovery
          </h1>

          <p class="page-subtitle">
            Controlled, isolated trials for
            ${esc(
              campaign.project ||
              "this experiment",
            )}
            
          </p>

          <div class="campaign-actions">
            ${
              campaign.source_run_id
                ? `
                  <a href="#/run/${encodeURIComponent(
                    campaign.source_run_id,
                  )}">
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
              <svg viewBox="0 0 24 24">
                <path
                  d="M18.5 5.5A9 9 0 1 0 21 12h-2a7 7 0 1 1-2.05-4.95L14 10h7V3l-2.5 2.5Z"
                />
              </svg>
              Refresh
            </button>

            <button
              class="ghost"
              data-copy="${esc(campaignId)}"
              data-copy-label="Campaign ID copied"
            >
              Copy campaign ID
            </button>
          </div>

          <aside
            class="campaign-agent-card"
            aria-label="Deterministic recovery policy"
          >
            <span
              class="agent-glyph"
              aria-hidden="true"
            >
              <svg viewBox="0 0 24 24">
                <path
                  d="m12 2 2.55 6.45L21 11l-6.45 2.55L12 20l-2.55-6.45L3 11l6.45-2.55L12 2Z"
                />
              </svg>
            </span>

            <div>
              <div class="agent-card-title">
                Deterministic policy
              </div>
              <div class="agent-card-copy">
                Bounded interventions, fresh processes
              </div>
            </div>
          </aside>
        </section>

        <section
          class="campaign-stat-strip"
          aria-label="Campaign summary"
        >
          <div class="campaign-stat">
            <div class="campaign-stat-label">
              Selected trial
            </div>
            <div class="campaign-stat-value">
              ${
                selectedIndex >= 0
                  ? `#${String(
                      selectedIndex + 1,
                    ).padStart(2, "0")}`
                  : "—"
              }
            </div>
          </div>

          <div class="campaign-stat">
            <div class="campaign-stat-label">
              Peak VRAM
            </div>
            <div class="campaign-stat-value">
              ${
                peakVram !== null
                  ? `${fmtNum(peakVram, 1)} GB`
                  : "—"
              }
            </div>
          </div>

          <div class="campaign-stat">
            <div class="campaign-stat-label">
              GPU budget used
            </div>
            <div class="campaign-stat-value">
              ${formatGpuTime(gpuSeconds)}
            </div>
          </div>

          <div class="campaign-stat">
            <div class="campaign-stat-label">
              Status
            </div>
            <div class="campaign-stat-value mint">
              ${statusLabel}
            </div>
          </div>
        </section>

        <section class="campaign-primary-grid">
          <article class="campaign-panel">
            <header class="campaign-panel-header">
              <span>Intervention trace</span>
              <span class="live-label">
                ${
                  campaignVerified
                    ? "verified"
                    : isActive
                      ? "live"
                      : "recorded"
                }
              </span>
            </header>

            <div class="campaign-panel-body">
              ${evidenceSteps
                .slice(-5)
                .map(
                  (step, index) => `
                    <div class="reasoning-step campaign-reasoning ${
                      step.active ? "active" : ""
                    }">
                      <span class="reasoning-num">
                        ${String(index + 1).padStart(2, "0")}
                      </span>

                      <span class="reasoning-text">
                        ${step.text}

                        <span class="reasoning-meta">
                          ${esc(step.meta)}
                        </span>
                      </span>
                    </div>
                  `,
                )
                .join("")}
            </div>
          </article>

          <article class="campaign-panel">
            <header class="campaign-panel-header">
              <span>Recorded trial metric</span>
              <span class="objective-change">
                ${provenance("captured")}
              </span>
            </header>

            <div class="objective-panel-body">
              ${renderSparkline(recordedMetrics)}

              <div class="chart-axis-labels">
                <span>control</span>
                <span>
                  ${
                    trials.length
                      ? `trial ${trials.length}`
                      : "current"
                  }
                </span>
              </div>
            </div>
          </article>
        </section>

        <section class="campaign-trials-panel">
          <div class="runs-table-wrapper">
            <table class="runs-table">
              <thead>
                <tr>
                  <th>trial</th>
                  <th>intervention</th>
                  <th>result</th>
                  <th>decision</th>
                </tr>
              </thead>

              <tbody>
                ${
                  trialRows ||
                  `
                    <tr>
                      <td
                        colspan="4"
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

        <section class="campaign-support-grid">
          <article class="panel m0">
            <h2 class="section-title">
              Recovery contract
              ${provenance("policy")}
            </h2>

            <div class="contract-grid">
              <div class="contract-item">
                <div class="field-label">
                  trial budget
                </div>
                <div class="contract-value">
                  ${
                    maximumTrials !== null
                      ? `${trials.length} / ${maximumTrials} trials`
                      : `${trials.length} trials used`
                  }
                </div>
              </div>

              <div class="contract-item">
                <div class="field-label">
                  confirmation runs
                </div>
                <div class="contract-value">
                  ${
                    confirmationRuns !== null
                      ? confirmationRuns
                      : "required before verification"
                  }
                </div>
              </div>

              <div class="contract-item">
                <div class="field-label">
                  max metric regression
                </div>
                <div class="contract-value">
                  ${
                    maximumRegression !== null
                      ? `${fmtNum(maximumRegression, 1)}%`
                      : "must be declared"
                  }
                </div>
              </div>

              <div class="contract-item">
                <div class="field-label">
                  minimum VRAM headroom
                </div>
                <div class="contract-value">
                  ${
                    minimumHeadroom !== null
                      ? `${fmtNum(minimumHeadroom, 1)}%`
                      : "must be declared"
                  }
                </div>
              </div>

              <div class="contract-item">
                <div class="field-label">
                  effective batch
                </div>
                <div class="contract-value">
                  ${
                    contract.preserve_effective_batch === false
                      ? "change allowed"
                      : "preserved"
                  }
                </div>
              </div>

              <div class="contract-item">
                <div class="field-label">
                  allowed interventions
                </div>
                <div class="contract-value">
                  ${esc(
                    allowedAutomaticChanges.join(", ") ||
                    "policy controlled",
                  )}
                </div>
              </div>
            </div>

            ${
              budgetPercentage !== null
                ? `
                  <div
                    class="budget-track"
                    title="${budgetPercentage.toFixed(0)}% of GPU budget used"
                  >
                    <div
                      class="budget-fill"
                      style="width:${budgetPercentage.toFixed(1)}%"
                    ></div>
                  </div>
                `
                : ""
            }
          </article>

          <article class="panel m0">
            <h2 class="section-title">Guardrails</h2>

            <div class="guardrail-list">
              <div class="guardrail-item">
                <span class="guardrail-icon">01</span>
                <span>
                  Reproduce the original OOM as a control
                  before testing a change.
                </span>
              </div>

              <div class="guardrail-item">
                <span class="guardrail-icon">02</span>
                <span>
                  Run every trial in a fresh process with
                  the same environment and dataset identity.
                </span>
              </div>

              <div class="guardrail-item">
                <span class="guardrail-icon">03</span>
                <span>
                  Change one bounded factor at a time and
                  stop at the declared trial or GPU budget.
                </span>
              </div>

              <div class="guardrail-item">
                <span class="guardrail-icon">04</span>
                <span>
                  Require confirmation runs, memory headroom,
                  and metric constraints before verification.
                </span>
              </div>
            </div>

            <p class="page-subtitle mt16 m0">
              <strong>Stopped because:</strong>
              ${esc(stoppedReason)}
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

// -----------------------------------------------------------------------------
// Verified recovery history
// -----------------------------------------------------------------------------

async function renderMemoryScreen() {
  app.innerHTML = `
    <p class="loading">
      loading verified recovery history&hellip;
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
          no verified recovery history yet
        </p>

        <p>
          History is added only after controlled confirmation
          trials satisfy their recovery contract.
        </p>
      </div>
    `;

    return;
  }

  app.innerHTML = `
    <p class="eyebrow">watcherml</p>

    <h1 class="page-title">
      Verified recovery history
      ${provenance("verified")}
    </h1>

    <p class="page-subtitle">
      Evidence from confirmed OOM recoveries. Prior outcomes
      can inform a proposal, but they never replace a new
      controlled trial.
    </p>

    ${signatures
      .map((signature) => {
        const successRate =
          firstFinite(signature.success_rate) ?? 0;

        const rateClass =
          successRate >= 0.7
            ? "good"
            : successRate <= 0.3
              ? "bad"
              : "mixed";

        const patchKeys = Array.isArray(
          signature.patch_keys,
        )
          ? signature.patch_keys
          : [];

        const examplePatches = Array.isArray(
          signature.example_patches,
        )
          ? signature.example_patches
          : [];

        return `
          <div class="signature-card">
            <div class="signature-title">
              ${esc(signature.failure_class)}
              &mdash;
              changing
              ${esc(
                patchKeys.join(", ") ||
                "(no keys)",
              )}
            </div>

            <div class="resolution-row">
              <span>
                ${
                  examplePatches
                    .map((patch) =>
                      esc(JSON.stringify(patch)),
                    )
                    .join(" / ") ||
                  "No patch example recorded"
                }
              </span>

              <span class="resolution-rate ${rateClass}">
                ${signature.successes ?? 0}/${
                  signature.attempts ?? 0
                }
                successful
                (${(successRate * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
        `;
      })
      .join("")}
  `;
}

// -----------------------------------------------------------------------------
// Settings
// -----------------------------------------------------------------------------

async function renderSettingsScreen() {
  app.innerHTML = `
    <p class="loading">loading settings&hellip;</p>
  `;

  let settings;

  try {
    settings = await api("/settings");
  } catch (error) {
    app.innerHTML = errorState(error);
    return;
  }

  const runtime = settings.runtime || {};
  const gpu = settings.gpu || {};

  const isolatedTrialRunner =
    settings.isolated_trial_runner ??
    runtime.isolated_trial_runner;

  app.innerHTML = `
    <p class="eyebrow">watcherml</p>
    <h1 class="page-title">Settings</h1>

    <div class="panel">
      <h2 class="section-title">Local storage</h2>

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
    </div>

    <div class="panel">
      <h2 class="section-title">Runtime</h2>

      <div class="field">
        <span class="field-label">mode</span>
        <span class="field-value">
          ${esc(
            settings.mode ||
            runtime.mode ||
            "local",
          )}
        </span>
      </div>

      <div class="field">
        <span class="field-label">python</span>
        <span class="field-value">
          ${esc(
            settings.python_version ||
            runtime.python_version,
          )}
        </span>
      </div>

      <div class="field">
        <span class="field-label">pytorch</span>
        <span class="field-value">
          ${esc(
            settings.torch_version ||
            runtime.torch_version,
          )}
        </span>
      </div>

      <div class="field">
        <span class="field-label">
          capsule schema
        </span>
        <span class="field-value">
          ${esc(
            settings.capsule_schema_version ||
            runtime.capsule_schema_version ||
            "v1",
          )}
        </span>
      </div>

      <div class="field">
        <span class="field-label">
          isolated trial runner
        </span>

        <span class="field-value">
          ${
            isolatedTrialRunner === true
              ? `
                <span class="badge success">
                  available
                </span>
              `
              : isolatedTrialRunner === false
                ? `
                  <span class="badge failed">
                    not available
                  </span>
                `
                : "not reported"
          }
        </span>
      </div>
    </div>

    <div class="panel">
      <h2 class="section-title">GPU</h2>

      <div class="field">
        <span class="field-label">detected</span>
        <span class="field-value">
          ${gpu.available ? "yes" : "no"}
        </span>
      </div>

      ${(gpu.gpus || [])
        .map(
          (device) => `
            <div class="field">
              <span class="field-label">
                ${esc(device.name)}
              </span>

              <span class="field-value">
                ${esc(device.memory_total_mib)} MiB total
              </span>
            </div>
          `,
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