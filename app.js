const MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS_SHORT = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const DAYS_FULL = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

const STATE = {
  predictions: null,
  drawsCsvRaw: "",
  drawsRows: [],
  activeRadar: "any_pos_gaps",
  activeFilter: "ALL",
  selectedYear: null,
  selectedMonth: null,
};

const STORAGE_KEY = "sai2d_gh_config_v1";

function getConfig() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveConfig(cfg) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
}

function parseDateParts(dateStr) {
  const [m, d, y] = dateStr.split("/").map((x) => parseInt(x, 10));
  const dt = new Date(y, m - 1, d);
  return {
    dt,
    year: y,
    monthIndex: m - 1,
    day: d,
    shortDate: `${d} ${MONTHS_SHORT[m - 1]}`,
    dayShort: DAYS_SHORT[dt.getDay()],
    dayFull: DAYS_FULL[dt.getDay()],
    dow: dt.getDay(), // 1=Mon ... 5=Fri
  };
}

function formatNum(val) {
  if (val === null || val === undefined || val === "" || val === "NaN") return "—";
  const n = parseInt(val, 10);
  return Number.isNaN(n) ? "—" : String(n).padStart(2, "0");
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const obj = {};
    headers.forEach((h, i) => (obj[h.trim()] = (vals[i] || "").trim()));
    return obj;
  });
}

function toCsv(rows) {
  const headers = ["Session_ID", "Year", "Day_of_Year", "Date", "Is_PM", "Winning_Number"];
  const body = rows.map((r) => headers.map((h) => r[h] ?? "").join(","));
  return [headers.join(","), ...body].join("\n") + "\n";
}

function encodeBase64Unicode(str) {
  return btoa(unescape(encodeURIComponent(str)));
}

function decodeBase64Unicode(b64) {
  return decodeURIComponent(escape(atob(b64.replace(/\n/g, ""))));
}

async function fetchRepoData() {
  const syncEl = document.getElementById("syncStatus");
  syncEl.textContent = "Syncing...";
  const cacheBust = `?t=${Date.now()}`;

  try {
    const [predRes, csvRes] = await Promise.all([
      fetch(`data/predictions.json${cacheBust}`),
      fetch(`data/draws.csv${cacheBust}`),
    ]);

    if (csvRes.ok) {
      STATE.drawsCsvRaw = await csvRes.text();
      STATE.drawsRows = parseCsv(STATE.drawsCsvRaw);
      initHistorySelectors();
      populateSessionDropdown();
      renderCurrentWeek();
      renderWinningHistory();
    }

    if (predRes.ok) {
      STATE.predictions = await predRes.json();
      syncEl.textContent = `Updated: ${STATE.predictions.updated_at || "Ready"}`;
      renderPredictionViews();
    } else {
      syncEl.textContent = "Waiting for Action run";
    }
  } catch (err) {
    syncEl.textContent = "Offline / Sync error";
    console.error(err);
  }
}

function populateSessionDropdown() {
  const select = document.getElementById("sessionSelect");
  select.innerHTML = "";
  const openRows = STATE.drawsRows.filter((r) => !r.Winning_Number || r.Winning_Number === "NaN");
  const completedRows = STATE.drawsRows.filter((r) => r.Winning_Number && r.Winning_Number !== "NaN");

  if (completedRows.length > 0) {
    const last = completedRows[completedRows.length - 1];
    const p = parseDateParts(last.Date);
    document.getElementById("latestBadge").textContent =
      `Latest: ${p.shortDate} (${p.dayFull}) ${last.Is_PM === "1" ? "PM" : "AM"} = ${formatNum(last.Winning_Number)}`;
  }

  openRows.slice(0, 10).forEach((r) => {
    const p = parseDateParts(r.Date);
    const opt = document.createElement("option");
    opt.value = r.Session_ID;
    opt.textContent = `${p.shortDate} (${p.dayFull}) — ${r.Is_PM === "1" ? "PM (4:30)" : "AM (12:01)"}`;
    select.appendChild(opt);
  });
}

function renderCurrentWeek() {
  const grid = document.getElementById("currentWeekGrid");
  const labelEl = document.getElementById("weekRangeLabel");
  if (!STATE.drawsRows.length) return;

  // Find the active week containing the first open slot (or latest completed draw)
  const firstOpen = STATE.drawsRows.find((r) => !r.Winning_Number || r.Winning_Number === "NaN");
  const anchorRow = firstOpen || STATE.drawsRows[STATE.drawsRows.length - 1];
  const anchorDate = parseDateParts(anchorRow.Date).dt;

  // Compute Monday of that week
  const day = anchorDate.getDay(); // 0=Sun..6=Sat
  const diffToMon = day === 0 ? -6 : 1 - day;
  const monday = new Date(anchorDate);
  monday.setDate(anchorDate.getDate() + diffToMon);

  const weekDates = [];
  for (let i = 0; i < 5; i++) {
    const d = new Date(monday);
    d.setDate(monday.getDate() + i);
    const key = `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
    weekDates.push({
      key,
      shortDate: `${d.getDate()} ${MONTHS_SHORT[d.getMonth()]}`,
      dayShort: DAYS_SHORT[d.getDay()],
      am: "—",
      pm: "—",
    });
  }

  const byDate = Object.fromEntries(weekDates.map((w) => [w.key, w]));
  STATE.drawsRows.forEach((r) => {
    if (byDate[r.Date]) {
      const val = formatNum(r.Winning_Number);
      if (r.Is_PM === "1") byDate[r.Date].pm = val;
      else byDate[r.Date].am = val;
    }
  });

  labelEl.textContent = `${weekDates[0].shortDate} – ${weekDates[4].shortDate}`;
  grid.innerHTML = weekDates
    .map(
      (w) => `
      <div class="week-day-box">
        <div class="week-day-header">
          <span>${w.shortDate}</span>
          <span>${w.dayShort}</span>
        </div>
        <div class="week-sessions">
          <div class="week-slot ${w.am !== "—" ? "filled" : ""}">
            <span class="slot-tag">AM</span>
            <span class="slot-num">${w.am}</span>
          </div>
          <div class="week-slot ${w.pm !== "—" ? "filled" : ""}">
            <span class="slot-tag">PM</span>
            <span class="slot-num">${w.pm}</span>
          </div>
        </div>
      </div>`
    )
    .join("");
}

function initHistorySelectors() {
  const completed = STATE.drawsRows.filter((r) => r.Winning_Number && r.Winning_Number !== "NaN");
  if (!completed.length) return;

  const years = [...new Set(completed.map((r) => parseInt(r.Year, 10)))].sort((a, b) => b - a);
  const latest = completed[completed.length - 1];
  const latestParts = parseDateParts(latest.Date);

  if (!STATE.selectedYear) STATE.selectedYear = latestParts.year;
  if (STATE.selectedMonth === null) STATE.selectedMonth = latestParts.monthIndex;

  const yearSelect = document.getElementById("historyYearSelect");
  yearSelect.innerHTML = years
    .map((y) => `<option value="${y}" ${y === STATE.selectedYear ? "selected" : ""}>${y}</option>`)
    .join("");

  renderMonthPills();
}

function renderMonthPills() {
  const pillsEl = document.getElementById("monthPills");
  pillsEl.innerHTML = MONTHS_SHORT.map(
    (mName, idx) => `
      <button class="month-pill ${idx === STATE.selectedMonth ? "active" : ""}" data-month="${idx}">
        ${mName}
      </button>`
  ).join("");

  pillsEl.querySelectorAll(".month-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      STATE.selectedMonth = parseInt(btn.dataset.month, 10);
      renderMonthPills();
      renderWinningHistory();
    });
  });
}

function renderWinningHistory() {
  const container = document.getElementById("winningMonthGrid");
  if (!STATE.drawsRows.length || STATE.selectedYear === null) return;

  const monthRows = STATE.drawsRows.filter((r) => {
    const p = parseDateParts(r.Date);
    return p.year === STATE.selectedYear && p.monthIndex === STATE.selectedMonth;
  });

  if (!monthRows.length) {
    container.innerHTML = `<p class="muted">No draws recorded for ${MONTHS_SHORT[STATE.selectedMonth]} ${STATE.selectedYear}.</p>`;
    return;
  }

  // Group by date first, then group dates into Monday-Friday weeks
  const dayMap = new Map();
  monthRows.forEach((r) => {
    if (!dayMap.has(r.Date)) {
      const p = parseDateParts(r.Date);
      const dayOfWeek = p.dt.getDay();
      const diffToMon = dayOfWeek === 0 ? -6 : 1 - dayOfWeek;
      const monDate = new Date(p.dt);
      monDate.setDate(p.dt.getDate() + diffToMon);
      const weekKey = `${monDate.getFullYear()}-${String(monDate.getMonth() + 1).padStart(2, "0")}-${String(monDate.getDate()).padStart(2, "0")}`;
      const weekLabel = `Week of ${monDate.getDate()} ${MONTHS_SHORT[monDate.getMonth()]}`;

      dayMap.set(r.Date, {
        dateStr: r.Date,
        shortDate: p.shortDate,
        dayShort: p.dayShort,
        weekKey,
        weekLabel,
        am: "—",
        pm: "—",
      });
    }
    const entry = dayMap.get(r.Date);
    const val = formatNum(r.Winning_Number);
    if (r.Is_PM === "1") entry.pm = val;
    else entry.am = val;
  });

  const weeksMap = new Map();
  dayMap.forEach((dayObj) => {
    if (!weeksMap.has(dayObj.weekKey)) {
      weeksMap.set(dayObj.weekKey, { label: dayObj.weekLabel, days: [] });
    }
    weeksMap.get(dayObj.weekKey).days.push(dayObj);
  });

  // Sort weeks newest first within the month
  const sortedWeeks = [...weeksMap.entries()].sort((a, b) => b[0].localeCompare(a[0]));

  container.innerHTML = sortedWeeks
    .map(
      ([, wk]) => `
      <div class="archive-week-row">
        <div class="archive-week-title">${wk.label}</div>
        <div class="archive-days-grid">
          ${wk.days
            .map(
              (d) => `
              <div class="week-day-box">
                <div class="week-day-header">
                  <span>${d.shortDate}</span>
                  <span>${d.dayShort}</span>
                </div>
                <div class="week-sessions">
                  <div class="week-slot ${d.am !== "—" ? "filled" : ""}">
                    <span class="slot-tag">AM</span>
                    <span class="slot-num">${d.am}</span>
                  </div>
                  <div class="week-slot ${d.pm !== "—" ? "filled" : ""}">
                    <span class="slot-tag">PM</span>
                    <span class="slot-num">${d.pm}</span>
                  </div>
                </div>
              </div>`
            )
            .join("")}
        </div>
      </div>`
    )
    .join("");
}

async function githubGetFile(path) {
  const cfg = getConfig();
  if (!cfg.owner || !cfg.repo || !cfg.token) {
    throw new Error("Please configure your GitHub token in '⚙ GitHub Setup' first.");
  }
  const branch = cfg.branch || "main";
  const url = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/contents/${path}?ref=${branch}`;
  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${cfg.token}`,
      Accept: "application/vnd.github+json",
    },
  });
  if (!res.ok) {
    throw new Error(`GitHub API error (${res.status}) reading ${path}`);
  }
  return res.json();
}

async function githubPutFile(path, contentStr, sha, message) {
  const cfg = getConfig();
  const branch = cfg.branch || "main";
  const url = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/contents/${path}`;
  const res = await fetch(url, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${cfg.token}`,
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      content: encodeBase64Unicode(contentStr),
      sha,
      branch,
    }),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.message || `GitHub commit failed (${res.status})`);
  }
  return res.json();
}

async function handleLogDraw(e) {
  e.preventDefault();
  const feedback = document.getElementById("loggerFeedback");
  const btn = document.getElementById("submitDrawBtn");
  const sid = document.getElementById("sessionSelect").value;
  const rawVal = document.getElementById("winningInput").value.trim();

  if (!/^\d{1,2}$/.test(rawVal)) {
    feedback.textContent = "Enter a valid 2-digit number from 00 to 99.";
    return;
  }
  const numInt = parseInt(rawVal, 10);
  const numFormatted = String(numInt).padStart(2, "0");

  btn.disabled = true;
  feedback.textContent = `Saving ${numFormatted} to GitHub...`;

  try {
    const fileData = await githubGetFile("data/draws.csv");
    const latestCsv = decodeBase64Unicode(fileData.content);
    const rows = parseCsv(latestCsv);
    const target = rows.find((r) => r.Session_ID === String(sid));
    if (!target) throw new Error("Target date slot not found in draws.csv");

    const p = parseDateParts(target.Date);
    const sessLabel = target.Is_PM === "1" ? "PM" : "AM";
    target.Winning_Number = String(numInt);
    const updatedCsv = toCsv(rows);

    await githubPutFile(
      "data/draws.csv",
      updatedCsv,
      fileData.sha,
      `feat(draw): ${p.shortDate} (${p.dayShort}) ${sessLabel} = ${numFormatted}`
    );

    document.getElementById("winningInput").value = "";
    feedback.textContent = `Saved ${p.shortDate} (${p.dayFull}) ${sessLabel} = ${numFormatted}! Models are updating in the background (~45s). Click Refresh shortly.`;
    STATE.drawsRows = rows;
    populateSessionDropdown();
    renderCurrentWeek();
    renderWinningHistory();
  } catch (err) {
    feedback.textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
}

function renderRadar() {
  const grid = document.getElementById("radarGrid");
  const radar = STATE.predictions?.drought_radar?.[STATE.activeRadar];
  if (!radar) {
    grid.innerHTML = `<p class="muted">No radar data loaded yet.</p>`;
    return;
  }
  const thresholdHigh = STATE.activeRadar === "any_pos_gaps" ? 9 : 22;
  const thresholdMid = STATE.activeRadar === "any_pos_gaps" ? 6 : 15;

  grid.innerHTML = Object.entries(radar)
    .map(([digit, gap]) => {
      const cls = gap >= thresholdHigh ? "hot-alert" : gap >= thresholdMid ? "warm-alert" : "";
      return `
        <div class="radar-cell ${cls}">
          <div class="radar-digit">${digit}</div>
          <div class="radar-gap">${gap} draws</div>
        </div>`;
    })
    .join("");
}

function renderUpcoming() {
  const container = document.getElementById("upcomingGrid");
  const list = STATE.predictions?.upcoming || [];
  if (!list.length) {
    container.innerHTML = `<p class="muted">No upcoming predictions available yet.</p>`;
    return;
  }

  container.innerHTML = list
    .map((item) => {
      const p = parseDateParts(item.date);
      const shortDate = item.short_date || p.shortDate;
      const dayFull = item.day_full || p.dayFull;
      const p1 = item.primary_anchor;
      const p2 = item.secondary_anchor;

      const pairsHtml = (item.top_pairs || [])
        .map(
          (pairObj) => `
          <div class="pair-badge" title="Supporters: ${(pairObj.supporters || []).join(", ")}">
            ${pairObj.pair} <span>(${(pairObj.prob * 100).toFixed(2)}%)</span>
          </div>`
        )
        .join("");

      const modelsHtml = Object.entries(item.models || {})
        .map(
          ([mName, m]) => `
          <div class="model-mini-row">
            <strong>${mName.replace("_", " ")}</strong>
            <span>Anchor: <b>${m.anchor_digit}</b> (${(m.anchor_prob * 100).toFixed(1)}%) • Pairs: ${m.top_pairs.join(", ")}</span>
          </div>`
        )
        .join("");

      return `
        <article class="pred-card">
          <div class="card-header" style="margin-bottom:0">
            <div>
              <strong style="font-size:15px">${shortDate} (${dayFull})</strong>
            </div>
            <span class="session-pill ${item.session.toLowerCase()}">${item.session}</span>
          </div>

          <div class="anchor-box">
            <div class="anchor-pill primary">
              <div class="anchor-label">Primary Anchor Digit</div>
              <div class="anchor-val">${p1.digit}</div>
              <div class="anchor-meta">${(p1.prob * 100).toFixed(1)}% • ${p1.votes} Models</div>
            </div>
            <div class="anchor-pill">
              <div class="anchor-label">Backup Anchor Digit</div>
              <div class="anchor-val">${p2.digit}</div>
              <div class="anchor-meta">${(p2.prob * 100).toFixed(1)}% • ${p2.votes} Models</div>
            </div>
          </div>

          <div>
            <div class="muted small" style="margin-bottom:6px">
              Anchor-Filtered Pairs (Cross-Pair: <b>${(item.intersection_pairs || []).join(" / ")}</b>)
            </div>
            <div class="pairs-row">${pairsHtml}</div>
          </div>

          <details>
            <summary>5-Model Breakdown</summary>
            <div class="model-mini-list">${modelsHtml}</div>
          </details>
        </article>`;
    })
    .join("");
}

function renderWinLog() {
  const hist = STATE.predictions?.history || [];
  const strip = document.getElementById("scoreSummaryStrip");
  const listEl = document.getElementById("historyList");

  const total = hist.length;
  const exact = hist.filter((h) => h.outcome === "EXACT").length;
  const rev = hist.filter((h) => h.outcome === "REVERSE").length;
  const anchor = hist.filter((h) => h.outcome === "ANCHOR_HIT").length;
  const anyWin = exact + rev + anchor;

  strip.innerHTML = `
    <div class="stat-box"><div class="muted small">Evaluated Draws</div><div class="num">${total}</div></div>
    <div class="stat-box"><div class="muted small">Total Wins (Pair + Digit)</div><div class="num">${anyWin} (${total ? ((anyWin / total) * 100).toFixed(1) : 0}%)</div></div>
    <div class="stat-box"><div class="muted small">Exact + Reverse (R)</div><div class="num" style="color:#fbbf24">${exact + rev}</div></div>
    <div class="stat-box"><div class="muted small">Anchor Digit Hits</div><div class="num" style="color:#38bdf8">${anchor}</div></div>
  `;

  const filtered = STATE.activeFilter === "ALL" ? hist : hist.filter((h) => h.outcome === STATE.activeFilter);
  if (!filtered.length) {
    listEl.innerHTML = `<p class="muted">No graded predictions in this filter yet. Every upcoming forecast is automatically archived and graded here as soon as you log its winning number.</p>`;
    return;
  }

  listEl.innerHTML = filtered
    .map((h) => {
      const p = parseDateParts(h.date);
      const shortDate = h.short_date || p.shortDate;
      const dayFull = h.day_full || p.dayFull;
      const pairs = (h.top_pairs || []).map((x) => x.pair).join(", ");
      const labelMap = {
        EXACT: "★ GOLD: EXACT PAIR",
        REVERSE: "◆ EMERALD: REVERSE (R)",
        ANCHOR_HIT: "● BLUE: ANCHOR DIGIT",
        MISS: "MISS",
      };
      return `
        <div class="hist-row ${h.outcome}">
          <div>
            <strong>${shortDate} (${dayFull}) • ${h.session}</strong> — Actual: <b style="font-size:16px">${h.actual}</b>
            <div class="muted small">
              Anchors: <b>${h.primary_anchor?.digit}</b> & <b>${h.secondary_anchor?.digit}</b> • Top Pairs: ${pairs}
            </div>
          </div>
          <span class="outcome-tag ${h.outcome}">${labelMap[h.outcome] || h.outcome}</span>
        </div>`;
    })
    .join("");
}

function renderModels() {
  const board = document.getElementById("modelLeaderboard");
  const weights = STATE.predictions?.model_weights || {};
  const stats = STATE.predictions?.model_stats || {};

  board.innerHTML = Object.keys(weights)
    .map((mName) => {
      const w = weights[mName] ?? 0.2;
      const st = stats[mName] || { evaluated: 0, anchor_hits: 0, exact_hits: 0, reverse_hits: 0 };
      const rate = st.evaluated ? ((st.anchor_hits / st.evaluated) * 100).toFixed(1) + "%" : "—";
      return `
        <div class="model-stat-card">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong>${mName.replace("_", " ")}</strong>
            <span class="chip">${(w * 100).toFixed(1)}% wt</span>
          </div>
          <div class="muted small" style="margin-top:8px">
            Anchor Hit Rate: <b>${rate}</b> (${st.anchor_hits}/${st.evaluated})<br/>
            Exact Pairs: <b>${st.exact_hits}</b> • Reverse (R): <b>${st.reverse_hits}</b>
          </div>
        </div>`;
    })
    .join("");
}

function renderPredictionViews() {
  renderRadar();
  renderUpcoming();
  renderWinLog();
  renderModels();
}

document.addEventListener("DOMContentLoaded", () => {
  const cfg = getConfig();
  if (cfg.owner) document.getElementById("ghOwner").value = cfg.owner;
  if (cfg.repo) document.getElementById("ghRepo").value = cfg.repo;
  if (cfg.branch) document.getElementById("ghBranch").value = cfg.branch;
  if (cfg.token) document.getElementById("ghToken").value = cfg.token;

  document.getElementById("logDrawForm").addEventListener("submit", handleLogDraw);
  document.getElementById("refreshBtn").addEventListener("click", fetchRepoData);

  document.getElementById("historyYearSelect").addEventListener("change", (e) => {
    STATE.selectedYear = parseInt(e.target.value, 10);
    renderWinningHistory();
  });

  document.getElementById("settingsBtn").addEventListener("click", () => {
    document.getElementById("settingsModal").classList.remove("hidden");
  });
  document.getElementById("closeSettingsBtn").addEventListener("click", () => {
    document.getElementById("settingsModal").classList.add("hidden");
  });
  document.getElementById("settingsForm").addEventListener("submit", (e) => {
    e.preventDefault();
    saveConfig({
      owner: document.getElementById("ghOwner").value.trim(),
      repo: document.getElementById("ghRepo").value.trim(),
      branch: document.getElementById("ghBranch").value.trim() || "main",
      token: document.getElementById("ghToken").value.trim(),
    });
    document.getElementById("settingsModal").classList.add("hidden");
    document.getElementById("loggerFeedback").textContent = "GitHub connection saved locally in browser.";
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      STATE.activeRadar = btn.dataset.radar;
      renderRadar();
    });
  });

  document.querySelectorAll(".view-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.view).classList.add("active");
    });
  });

  document.querySelectorAll(".filter-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-chip").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      STATE.activeFilter = btn.dataset.filter;
      renderWinLog();
    });
  });

  fetchRepoData();
});
