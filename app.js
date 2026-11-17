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

function formatYangonTimestamp(predObj) {
  if (!predObj) return "Ready (MMT)";
  const raw = predObj.updated_at_iso || predObj.updated_at;
  if (!raw) return "Ready (MMT)";

  let dt = new Date(raw);

  if (Number.isNaN(dt.getTime())) {
    const match = String(raw).match(/(\d{1,2})\s+([A-Za-z]{3})\s+•\s+(\d{1,2}):(\d{2})\s*UTC/i);
    if (match) {
      const day = parseInt(match[1], 10);
      const monIdx = MONTHS_SHORT.findIndex((m) => m.toLowerCase() === match[2].toLowerCase());
      const hr = parseInt(match[3], 10);
      const min = parseInt(match[4], 10);
      if (monIdx >= 0) {
        dt = new Date(Date.UTC(new Date().getFullYear(), monIdx, day, hr, min));
      }
    }
  }

  if (Number.isNaN(dt.getTime())) {
    return String(raw);
  }

  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Yangon",
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).formatToParts(dt);

  const get = (type) => (parts.find((p) => p.type === type) || {}).value || "";
  const dayPeriod = get("dayPeriod").toUpperCase();
  return `${get("day")} ${get("month")} • ${get("hour")}:${get("minute")} ${dayPeriod} MMT`;
}

function parseDateParts(dateStr) {
  const [m, d, y] = dateStr.split("/").map((x) => parseInt(x, 10));
  const dt = new Date(y, m - 1, d);
  return {
    dt,
    year: y,
    monthIndex: m - 1,
    day: d,
    shortDate: `${d}${MONTHS_SHORT[m - 1]}`,
    dayShort: DAYS_SHORT[dt.getDay()],
    dayFull: DAYS_FULL[dt.getDay()],
    dow: dt.getDay(),
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
  const cacheBust = `?t=${Date.now()}`;  try { const [predRes, csvRes] = await Promise.all([ fetch(`data/predictions.json${cacheBust}`),
      fetch(`data/draws.csv${cacheBust}`), ]);  if (csvRes.ok) { STATE.drawsCsvRaw = await csvRes.text(); STATE.drawsRows = parseCsv(STATE.drawsCsvRaw); initHistorySelectors(); populateSessionDropdown(); renderCurrentWeek(); renderWinningHistory(); }  if (predRes.ok) { STATE.predictions = await predRes.json(); syncEl.textContent = formatYangonTimestamp(STATE.predictions); renderPredictionViews(); } else { syncEl.textContent = "Waiting for Action"; } } catch (err) { syncEl.textContent = "Sync error"; console.error(err); } }  function populateSessionDropdown() { const select = document.getElementById("sessionSelect"); select.innerHTML = ""; const openRows = STATE.drawsRows.filter((r) => !r.Winning_Number \vert{}\vert{} r.Winning_Number === "NaN"); const completedRows = STATE.drawsRows.filter((r) => r.Winning_Number && r.Winning_Number !== "NaN");  if (completedRows.length > 0) { const last = completedRows[completedRows.length - 1]; const p = parseDateParts(last.Date); document.getElementById("latestBadge").textContent = `Latest: ${p.shortDate} (${p.dayShort})${last.Is_PM === "1" ? "PM" : "AM"} = ${formatNum(last.Winning_Number)}`; }  openRows.slice(0, 10).forEach((r) => { const p = parseDateParts(r.Date); const opt = document.createElement("option"); opt.value = r.Session_ID; opt.textContent = `${p.shortDate} (${p.dayFull}) —${r.Is_PM === "1" ? "PM (4:30)" : "AM (12:01)"}`;
    select.appendChild(opt);
  });
}

function renderDayColumnHtml(d) {
  return `
    <div class="compact-day-col">
      <div class="compact-day-head">
        <span>${d.dayNum}</span>
        <span>${d.dayShort}</span>
      </div>
      <div class="slot-box am-slot ${d.am === "—" ? "is-empty" : ""}">${d.am}</div>
      <div class="slot-box pm-slot ${d.pm === "—" ? "is-empty" : ""}">${d.pm}</div>
    </div>`;
}

function renderCurrentWeek() {
  const grid = document.getElementById("currentWeekGrid");
  const labelEl = document.getElementById("weekRangeLabel");
  if (!STATE.drawsRows.length) return;

  const firstOpen = STATE.drawsRows.find((r) => !r.Winning_Number || r.Winning_Number === "NaN");
  const anchorRow = firstOpen || STATE.drawsRows[STATE.drawsRows.length - 1];
  const anchorDate = parseDateParts(anchorRow.Date).dt;

  const day = anchorDate.getDay();
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
      dayNum: `${d.getDate()}${MONTHS_SHORT[d.getMonth()]}`,
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

  labelEl.textContent = `${weekDates[0].dayNum} –${weekDates[4].dayNum}`;
  grid.innerHTML = weekDates.map(renderDayColumnHtml).join("");
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

  const lookup = {};
  STATE.drawsRows.forEach((r) => {
    const p = parseDateParts(r.Date);
    if (p.year === STATE.selectedYear && p.monthIndex === STATE.selectedMonth) {
      if (!lookup[p.day]) lookup[p.day] = { am: "—", pm: "—" };
      const val = formatNum(r.Winning_Number);
      if (r.Is_PM === "1") lookup[p.day].pm = val;
      else lookup[p.day].am = val;
    }
  });

  const daysInMonth = new Date(STATE.selectedYear, STATE.selectedMonth + 1, 0).getDate();
  const weeks = [];
  let currentWeek = null;

  for (let day = 1; day <= daysInMonth; day++) {
    const dt = new Date(STATE.selectedYear, STATE.selectedMonth, day);
    const dow = dt.getDay();
    if (dow === 0 || dow === 6) continue;

    if (!currentWeek || dow === 1) {
      currentWeek = {
        startDay: day,
        endDay: day,
        days: [],
      };
      weeks.push(currentWeek);
    }
    currentWeek.endDay = day;
    const vals = lookup[day] || { am: "—", pm: "—" };
    currentWeek.days.push({
      dayNum: `${day}${MONTHS_SHORT[STATE.selectedMonth]}`,
      dayShort: DAYS_SHORT[dow],
      am: vals.am,
      pm: vals.pm,
    });
  }

  if (!weeks.length) {
    container.innerHTML = `<p class="muted">No weekday draws for ${MONTHS_SHORT[STATE.selectedMonth]}${STATE.selectedYear}.</p>`;
    return;
  }

  container.innerHTML = weeks
    .map(
      (wk, idx) => `
      <div class="week-card">
        <div class="week-card-label">
          <span>Week ${idx + 1}</span>
          <span>${wk.startDay}–${wk.endDay}${MONTHS_SHORT[STATE.selectedMonth]}</span>
        </div>
        <div class="compact-days-row">
          ${wk.days.map(renderDayColumnHtml).join("")}
        </div>
      </div>`
    )
    .join("");
}

async function githubGetFile(path) {
  const cfg = getConfig();
  if (!cfg.owner || !cfg.repo || !cfg.token) {
    throw new Error("Please configure your GitHub token in the ⚙ Gear icon first.");
  }
  const branch = cfg.branch || "main";
  const url = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/contents/${path}?ref=${branch}`; const res = await fetch(url, { headers: { Authorization: `Bearer ${cfg.token}`,
      Accept: "application/vnd.github+json",
    },
  });
  if (!res.ok) {
    throw new Error(`GitHub API error (${res.status}) reading${path}`);
  }
  return res.json();
}

async function githubPutFile(path, contentStr, sha, message) {
  const cfg = getConfig();
  const branch = cfg.branch || "main";
  const url = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/contents/${path}`; const res = await fetch(url, { method: "PUT", headers: { Authorization: `Bearer ${cfg.token}`,
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
    throw new Error(errBody.message || `GitHub commit failed (${res.status})`); } return res.json(); }  async function handleLogDraw(e) { e.preventDefault(); const feedback = document.getElementById("loggerFeedback"); const btn = document.getElementById("submitDrawBtn"); const sid = document.getElementById("sessionSelect").value; const rawVal = document.getElementById("winningInput").value.trim();  if (!/^\d{1,2}$/.test(rawVal)) { feedback.textContent = "Enter a valid 2-digit number from 00 to 99."; return; } const numInt = parseInt(rawVal, 10); const numFormatted = String(numInt).padStart(2, "0");  btn.disabled = true; feedback.textContent = `Saving ${numFormatted} to GitHub...`;

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
      `feat(draw): ${p.shortDate} (${p.dayShort}) ${sessLabel} =${numFormatted}`
    );

    document.getElementById("winningInput").value = "";
    feedback.textContent = `Saved ${p.shortDate} (${p.dayFull}) ${sessLabel} =${numFormatted}! Models are updating (~45s). Tap ↻ shortly.`;
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
          <div class="radar-gap">${gap}d</div>
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
            <span>Anchor: <b>${m.anchor_digit}</b> (Backup: ${m.backup_digit ?? "—"}) • Pairs: ${m.top_pairs.join(", ")}</span>
          </div>`
        )
        .join("");

      return `
        <article class="pred-card">
          <div class="card-header" style="margin-bottom:0">
            <strong style="font-size:15px">${shortDate} (${dayFull})</strong>
            <span class="session-pill ${item.session.toLowerCase()}">${item.session}</span>
          </div>

          <div class="anchor-box">
            <div class="anchor-pill primary">
              <div class="anchor-label">Primary Anchor</div>
              <div class="anchor-val">${p1.digit}</div>
              <div class="anchor-meta">${(p1.prob * 100).toFixed(1)}\% •${p1.votes} Models</div>
            </div>
            <div class="anchor-pill">
              <div class="anchor-label">Backup Anchor</div>
              <div class="anchor-val">${p2.digit}</div>
              <div class="anchor-meta">${(p2.prob * 100).toFixed(1)}\% •${p2.votes} Models</div>
            </div>
          </div>

          <div>
            <div class="muted small" style="margin-bottom:5px">
              Top Pairs (Cross: <b>${(item.intersection_pairs || []).join(" / ")}</b>)
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
    <div class="stat-box"><div class="muted small">Evaluated</div><div class="num">${total}</div></div>
    <div class="stat-box"><div class="muted small">Total Wins</div><div class="num">${anyWin} (${total ? ((anyWin / total) * 100).toFixed(1) : 0}%)</div></div>
    <div class="stat-box"><div class="muted small">Exact + Reverse</div><div class="num" style="color:#fbbf24">${exact + rev}</div></div>
    <div class="stat-box"><div class="muted small">Anchor Hits</div><div class="num" style="color:#38bdf
