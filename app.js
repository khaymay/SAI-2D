const MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS_SHORT = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const DAYS_FULL = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

// Built-in fallback Thailand / SET public holidays (merged automatically with predictions.json thai_holidays)
const BUILTIN_THAI_HOLIDAYS = {
  "1/1/2026": "New Year's Day",
  "3/3/2026": "Makha Bucha Day",
  "4/6/2026": "Chakri Memorial Day",
  "4/13/2026": "Songkran Festival",
  "4/14/2026": "Songkran Festival",
  "4/15/2026": "Songkran Festival",
  "5/1/2026": "National Labour Day",
  "5/4/2026": "Coronation Day",
  "6/1/2026": "Substitution for Visakha Bucha Day",
  "6/3/2026": "H.M. Queen Suthida's Birthday",
  "7/28/2026": "H.M. King Maha Vajiralongkorn's Birthday",
  "7/29/2026": "Asarnha Bucha Day",
  "8/12/2026": "H.M. Queen Sirikit The Queen Mother's Birthday",
  "10/13/2026": "H.M. King Bhumibol Adulyadej Memorial Day",
  "10/23/2026": "Chulalongkorn Day",
  "12/7/2026": "Substitution for H.M. King Bhumibol's Birthday",
  "12/10/2026": "Constitution Day",
  "12/31/2026": "New Year's Eve",
  "1/1/2027": "New Year's Day",
};

const STATE = {
  predictions: null,
  drawsCsvRaw: "",
  drawsRows: [],
  activeRadar: "any_pos_gaps",
  activeFilter: "ALL",
  selectedYear: null,
  selectedMonth: null,
  winLogWeeks: [],
  winLogWeekIndex: 0,
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

function getHolidayName(dateKey) {
  const fromPred = STATE.predictions?.thai_holidays || {};
  return fromPred[dateKey] || BUILTIN_THAI_HOLIDAYS[dateKey] || null;
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
    shortDate: `${d} ${MONTHS_SHORT[m - 1]}`,
    dayShort: DAYS_SHORT[dt.getDay()],
    dayFull: DAYS_FULL[dt.getDay()],
    dow: dt.getDay(),
  };
}

function getMondayKeyAndLabel(dateStr) {
  const dt = parseDateParts(dateStr).dt;
  const day = dt.getDay();
  const diffToMon = day === 0 ? -6 : 1 - day;
  const mon = new Date(dt);
  mon.setDate(dt.getDate() + diffToMon);
  const fri = new Date(mon);
  fri.setDate(mon.getDate() + 4);

  const key = `${mon.getFullYear()}-${String(mon.getMonth() + 1).padStart(2, "0")}-${String(mon.getDate()).padStart(2, "0")}`;
  const monLbl = `${mon.getDate()} ${MONTHS_SHORT[mon.getMonth()]}`;
  const friLbl = `${fri.getDate()} ${MONTHS_SHORT[fri.getMonth()]} ${fri.getFullYear()}`;
  return { key, label: `${monLbl} – ${friLbl}` };
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

async function fetchLatestRepoFileText(path) {
  const cfg = getConfig();
  const cacheBust = `?t=${Date.now()}`;

  if (cfg.owner && cfg.repo && cfg.token) {
    try {
      const branch = cfg.branch || "main";
      const apiUrl = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/contents/${path}?ref=${branch}&_=${Date.now()}`;
      const apiRes = await fetch(apiUrl, {
        cache: "no-store",
        headers: {
          Authorization: `Bearer ${cfg.token}`,
          Accept: "application/vnd.github+json",
        },
      });
      if (apiRes.ok) {
        const json = await apiRes.json();
        if (json && json.content) {
          return decodeBase64Unicode(json.content);
        }
      }
    } catch (e) {
      console.warn("Direct GitHub API fetch fallback to Pages:", e);
    }
  }

  const res = await fetch(`${path}${cacheBust}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${path}`);
  return res.text();
}

async function fetchRepoData() {
  const syncEl = document.getElementById("syncStatus");
  syncEl.textContent = "Syncing...";

  try {
    const [predText, csvText] = await Promise.all([
      fetchLatestRepoFileText("data/predictions.json").catch(() => null),
      fetchLatestRepoFileText("data/draws.csv").catch(() => null),
    ]);

    if (predText) {
      STATE.predictions = JSON.parse(predText);
      syncEl.textContent = formatYangonTimestamp(STATE.predictions);
    } else {
      syncEl.textContent = "Waiting for Action";
    }

    if (csvText) {
      STATE.drawsCsvRaw = csvText;
      STATE.drawsRows = parseCsv(STATE.drawsCsvRaw);
      initHistorySelectors();
      populateSessionDropdown();
      renderCurrentWeek();
      renderWinningHistory();
    }

    if (STATE.predictions) {
      renderPredictionViews();
    }
  } catch (err) {
    syncEl.textContent = "Sync error";
    console.error(err);
  }
}

function populateSessionDropdown() {
  const select = document.getElementById("sessionSelect");
  select.innerHTML = "";
  const openRows = STATE.drawsRows.filter(
    (r) => (!r.Winning_Number || r.Winning_Number === "NaN") && !getHolidayName(r.Date)
  );
  const completedRows = STATE.drawsRows.filter((r) => r.Winning_Number && r.Winning_Number !== "NaN");

  if (completedRows.length > 0) {
    const last = completedRows[completedRows.length - 1];
    const p = parseDateParts(last.Date);
    document.getElementById("latestBadge").textContent =
      `Latest: ${p.shortDate} (${p.dayShort}) ${last.Is_PM === "1" ? "PM" : "AM"} = ${formatNum(last.Winning_Number)}`;
  }

  openRows.slice(0, 10).forEach((r) => {
    const p = parseDateParts(r.Date);
    const opt = document.createElement("option");
    opt.value = r.Session_ID;
    opt.textContent = `${p.shortDate} (${p.dayFull}) — ${r.Is_PM === "1" ? "PM (4:30)" : "AM (12:01)"}`;
    select.appendChild(opt);
  });
}

function renderDayColumnHtml(d) {
  const holName = getHolidayName(d.key);
  if (holName && d.am === "—" && d.pm === "—") {
    return `
      <div class="compact-day-col is-holiday" title="Thai SET Holiday: ${holName}">
        <div class="compact-day-head">
          <span>${d.dayNum}</span>
          <span>${d.dayShort}</span>
        </div>
        <div class="holiday-box">
          <span class="holiday-tag">HOLIDAY</span>
          <span class="holiday-sub">${holName}</span>
        </div>
      </div>`;
  }

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

  const firstOpen = STATE.drawsRows.find(
    (r) => (!r.Winning_Number || r.Winning_Number === "NaN") && !getHolidayName(r.Date)
  );
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
      dayNum: `${d.getDate()} ${MONTHS_SHORT[d.getMonth()]}`,
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

  labelEl.textContent = `${weekDates[0].dayNum} – ${weekDates[4].dayNum}`;
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
    const key = `${STATE.selectedMonth + 1}/${day}/${STATE.selectedYear}`;
    const vals = lookup[day] || { am: "—", pm: "—" };
    currentWeek.days.push({
      key,
      dayNum: `${day} ${MONTHS_SHORT[STATE.selectedMonth]}`,
      dayShort: DAYS_SHORT[dow],
      am: vals.am,
      pm: vals.pm,
    });
  }

  if (!weeks.length) {
    container.innerHTML = `<p class="muted">No weekday draws for ${MONTHS_SHORT[STATE.selectedMonth]} ${STATE.selectedYear}.</p>`;
    return;
  }

  container.innerHTML = weeks
    .map(
      (wk, idx) => `
      <div class="week-card">
        <div class="week-card-label">
          <span>Week ${idx + 1}</span>
          <span>${wk.startDay}–${wk.endDay} ${MONTHS_SHORT[STATE.selectedMonth]}</span>
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
  const url = `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/contents/${path}?ref=${branch}`;
  const res = await fetch(url, {
    cache: "no-store",
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
    feedback.textContent = `Saved ${p.shortDate} (${p.dayFull}) ${sessLabel} = ${numFormatted}! Chained predictions updating (~45s). Tap ↻ once Action finishes.`;
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

async function handleUndoLastDraw() {
  const feedback = document.getElementById("loggerFeedback");
  const undoBtn = document.getElementById("undoLastDrawBtn");
  if (!confirm("Clear the most recently logged draw result in draws.csv?")) return;

  undoBtn.disabled = true;
  feedback.textContent = "Clearing last draw on GitHub...";

  try {
    const fileData = await githubGetFile("data/draws.csv");
    const latestCsv = decodeBase64Unicode(fileData.content);
    const rows = parseCsv(latestCsv);
    const completed = rows.filter((r) => r.Winning_Number && r.Winning_Number !== "NaN");
    if (!completed.length) throw new Error("No completed draws found to undo.");

    const lastRow = completed[completed.length - 1];
    const p = parseDateParts(lastRow.Date);
    const sessLabel = lastRow.Is_PM === "1" ? "PM" : "AM";
    const oldVal = formatNum(lastRow.Winning_Number);
    lastRow.Winning_Number = "";

    await githubPutFile(
      "data/draws.csv",
      toCsv(rows),
      fileData.sha,
      `fix(draw): undo ${p.shortDate} (${p.dayShort}) ${sessLabel} (${oldVal})`
    );

    feedback.textContent = `Cleared ${p.shortDate} ${sessLabel} (${oldVal}). Models re-syncing (~45s).`;
    STATE.drawsRows = rows;
    populateSessionDropdown();
    renderCurrentWeek();
    renderWinningHistory();
  } catch (err) {
    feedback.textContent = `Undo error: ${err.message}`;
  } finally {
    undoBtn.disabled = false;
  }
}

function renderRadar() {
  const grid = document.getElementById("radarGrid");
  const radar = STATE.predictions?.drought_radar?.[STATE.activeRadar];
  if (!radar) {
    grid.innerHTML = `<span class="muted">No radar data</span>`;
    return;
  }
  const thresholdHigh = STATE.activeRadar === "any_pos_gaps" ? 9 : 22;
  const thresholdMid = STATE.activeRadar === "any_pos_gaps" ? 6 : 15;

  grid.innerHTML = Object.entries(radar)
    .map(([digit, gap]) => {
      const cls = gap >= thresholdHigh ? "hot-alert" : gap >= thresholdMid ? "warm-alert" : "";
      return `
        <div class="radar-chip ${cls}" title="Digit ${digit}: ${gap} draws since last appearance">
          <b>${digit}</b><span>${gap}d</span>
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
    .map((item, idx) => {
      const p = parseDateParts(item.date);
      const shortDate = item.short_date || p.shortDate;
      const dayFull = item.day_full || p.dayFull;
      const p1 = item.primary_anchor;
      const p2 = item.secondary_anchor;

      const chainTrail = (item.chained_from || []).join(" → ");
      const modeTag = idx === 0
        ? `<span class="chip" style="margin-left:6px;font-size:10px">Next Live Draw</span>`
        : `<span class="chain-pill" title="Calculated sequentially from prior predicted top pairs">Chained${chainTrail ? ` (${chainTrail})` : ""}</span>`;

      const pairsHtml = (item.top_pairs || [])
        .map(
          (pairObj, pIdx) => `
          <div class="pair-badge ${pIdx === 0 ? "top1-pair" : ""}" title="Supporters: ${(pairObj.supporters || []).join(", ")}">
            ${pairObj.pair} <span>(${(pairObj.prob * 100).toFixed(2)}%)</span>
          </div>`
        )
        .join("");

      const modelsHtml = Object.entries(item.models || {})
        .map(
          ([mName, m]) => `
          <div class="model-mini-row">
            <strong>${mName.replace("_", " ")}</strong>
            <span>Anchor: <b>${m.anchor_digit}</b> / <b>${m.backup_digit ?? "—"}</b> • Pairs: ${m.top_pairs.join(", ")}</span>
          </div>`
        )
        .join("");

      return `
        <article class="pred-card">
          <div class="card-header" style="margin-bottom:0">
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px">
              <strong style="font-size:15px">${shortDate} (${dayFull})</strong>
              ${modeTag}
            </div>
            <span class="session-pill ${item.session.toLowerCase()}">${item.session}</span>
          </div>

          <div class="anchor-box">
            <div class="anchor-pill primary">
              <div class="anchor-label">Primary Anchor</div>
              <div class="anchor-val">${p1.digit}</div>
              <div class="anchor-meta">${(p1.prob * 100).toFixed(1)}% • ${p1.votes} Models</div>
            </div>
            <div class="anchor-pill">
              <div class="anchor-label">Backup Anchor</div>
              <div class="anchor-val">${p2.digit}</div>
              <div class="anchor-meta">${(p2.prob * 100).toFixed(1)}% • ${p2.votes} Models</div>
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

function computeClientModelScore(m, actStr) {
  if (typeof m.earned_pts === "number") {
    const d = m.score_detail || {};
    return {
      earned: m.earned_pts,
      hitExact: (d.exact_cnt || 0) > 0,
      hitRev: (d.rev_cnt || 0) > 0,
      hitAnchor: Boolean(d.hit_any_anchor),
      hitPos: (d.pos_cnt || 0) > 0,
      hitAnyDigit: (d.any_cnt || 0) > 0,
    };
  }

  const actNum = parseInt(actStr, 10);
  const actT = Math.floor(actNum / 10);
  const actU = actNum % 10;
  const revStr = `${actU}${actT}`;
  const actDigits = new Set([actT, actU]);

  const a1 = Number(m.anchor_digit);
  const a2 = m.backup_digit !== undefined ? Number(m.backup_digit) : -1;
  const hitA1 = actDigits.has(a1);
  const hitA2 = actDigits.has(a2);

  let anchorPts = 0;
  if (hitA1 && hitA2 && a1 !== a2) anchorPts = 1.5;
  else if (hitA1 || hitA2) anchorPts = 1.0;

  let pairPts = 0;
  let hitExact = false;
  let hitRev = false;
  let hitPos = false;
  let hitAnyDigit = false;

  (m.top_pairs || []).slice(0, 3).forEach((pStr) => {
    const pNum = parseInt(pStr, 10);
    const pt = Math.floor(pNum / 10);
    const pu = pNum % 10;
    if (pStr === actStr) {
      pairPts += 3.0;
      hitExact = true;
    } else if (pStr === revStr && actStr !== revStr) {
      pairPts += 2.0;
      hitRev = true;
    } else if (pt == actT || pu == actU) {
      pairPts += 0.2;
      hitPos = true;
    } else if (actDigits.has(pt) || actDigits.has(pu)) {
      pairPts += 0.1;
      hitAnyDigit = true;
    }
  });

  return {
    earned: Math.round((anchorPts + pairPts) * 100) / 100,
    hitExact,
    hitRev,
    hitAnchor: hitA1 || hitA2,
    hitPos,
    hitAnyDigit,
  };
}

function groupWinLogByWeeks(hist) {
  const weekMap = new Map();
  hist.forEach((item) => {
    const { key, label } = getMondayKeyAndLabel(item.date);
    if (!weekMap.has(key)) {
      weekMap.set(key, { key, label, items: [] });
    }
    weekMap.get(key).items.push(item);
  });
  return Array.from(weekMap.values()).sort((a, b) => b.key.localeCompare(a.key));
}

function renderWinLog() {
  const hist = STATE.predictions?.history || [];
  const strip = document.getElementById("scoreSummaryStrip");
  const weekScoreEl = document.getElementById("winLogWeekScoreStrip");
  const listEl = document.getElementById("historyList");
  const weekTitleEl = document.getElementById("winLogWeekTitle");
  const weekSubEl = document.getElementById("winLogWeekSub");
  const prevBtn = document.getElementById("prevWinWeekBtn");
  const nextBtn = document.getElementById("nextWinWeekBtn");

  // 1. All-Time Summary Strip
  const total = hist.length;
  const exact = hist.filter((h) => h.outcome === "EXACT").length;
  const rev = hist.filter((h) => h.outcome === "REVERSE").length;
  const anchor = hist.filter((h) => h.outcome === "ANCHOR_HIT").length;
  const anyWin = exact + rev + anchor;

  strip.innerHTML = `
    <div class="stat-box"><div class="muted small">All-Time Evaluated</div><div class="num">${total}</div></div>
    <div class="stat-box"><div class="muted small">All-Time Wins</div><div class="num" style="color:#34d399">${anyWin} (${total ? ((anyWin / total) * 100).toFixed(1) : 0}%)</div></div>
    <div class="stat-box"><div class="muted small">Exact + Reverse</div><div class="num" style="color:#fbbf24">${exact + rev}</div></div>
    <div class="stat-box"><div class="muted small">Anchor Hits</div><div class="num" style="color:#38bdf8">${anchor}</div></div>
  `;

  // 2. Week-by-Week Grouping
  STATE.winLogWeeks = groupWinLogByWeeks(hist);
  if (!STATE.winLogWeeks.length) {
    weekTitleEl.textContent = "No Evaluated Weeks Yet";
    weekSubEl.textContent = "";
    weekScoreEl.innerHTML = "";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    listEl.innerHTML = `<p class="muted">Every forecast is automatically graded here once its result is logged.</p>`;
    return;
  }

  if (STATE.winLogWeekIndex >= STATE.winLogWeeks.length) STATE.winLogWeekIndex = 0;
  if (STATE.winLogWeekIndex < 0) STATE.winLogWeekIndex = 0;

  const currentWeek = STATE.winLogWeeks[STATE.winLogWeekIndex];
  weekTitleEl.textContent = `Week of ${currentWeek.label}`;
  weekSubEl.textContent = `Week ${STATE.winLogWeeks.length - STATE.winLogWeekIndex} of ${STATE.winLogWeeks.length}`;
  prevBtn.disabled = STATE.winLogWeekIndex >= STATE.winLogWeeks.length - 1;
  nextBtn.disabled = STATE.winLogWeekIndex <= 0;

  // Weekly Scorecard
  const wkItems = currentWeek.items;
  const wkTotal = wkItems.length;
  const wkExact = wkItems.filter((h) => h.outcome === "EXACT").length;
  const wkRev = wkItems.filter((h) => h.outcome === "REVERSE").length;
  const wkAnchor = wkItems.filter((h) => h.outcome === "ANCHOR_HIT").length;
  const wkWins = wkExact + wkRev + wkAnchor;

  weekScoreEl.innerHTML = `
    <div class="wk-pill"><span>This Week Draws</span><b>${wkTotal}</b></div>
    <div class="wk-pill"><span>Weekly Wins</span><b style="color:#34d399">${wkWins}/${wkTotal} (${wkTotal ? ((wkWins / wkTotal) * 100).toFixed(1) : 0}%)</b></div>
    <div class="wk-pill"><span>Exact / Rev</span><b style="color:#fbbf24">${wkExact} / ${wkRev}</b></div>
    <div class="wk-pill"><span>Anchor Hits</span><b style="color:#38bdf8">${wkAnchor}</b></div>
  `;

  const filtered = STATE.activeFilter === "ALL" ? wkItems : wkItems.filter((h) => h.outcome === STATE.activeFilter);
  if (!filtered.length) {
    listEl.innerHTML = `<p class="muted">No draws match this filter in ${currentWeek.label}.</p>`;
    return;
  }

  listEl.innerHTML = filtered
    .map((h) => {
      const p = parseDateParts(h.date);
      const shortDate = h.short_date || p.shortDate;
      const dayFull = h.day_full || p.dayFull;
      const pairs = (h.top_pairs || []).map((x) => x.pair).join(", ");
      const actStr = String(h.actual || "").padStart(2, "0");

      const allModelsHtml = Object.entries(h.models || {})
        .map(([mName, m]) => {
          const mPairs = m.top_pairs || [];
          const sc = computeClientModelScore(m, actStr);

          let tagBadge = "";
          let rowCls = "";
          if (sc.hitExact) {
            tagBadge = `<span class="mini-hit gold">EXACT</span>`;
            rowCls = "hit-row";
          } else if (sc.hitRev) {
            tagBadge = `<span class="mini-hit emerald">REV</span>`;
            rowCls = "hit-row";
          } else if (sc.hitAnchor) {
            tagBadge = `<span class="mini-hit blue">ANCHOR</span>`;
            rowCls = "hit-row";
          } else if (sc.hitPos) {
            tagBadge = `<span class="mini-hit pos">POS</span>`;
            rowCls = "hit-row-soft";
          } else if (sc.hitAnyDigit) {
            tagBadge = `<span class="mini-hit digit">DIGIT</span>`;
            rowCls = "hit-row-soft";
          }

          const ptsBadge = sc.earned > 0
            ? `<span class="pts-pill earned">+${sc.earned.toFixed(1)} pts</span>`
            : `<span class="pts-pill zero">0.0 pts</span>`;

          return `
            <div class="winlog-model-item ${rowCls}">
              <div class="winlog-model-top">
                <strong>${mName.replace("_", " ")}</strong>
                <div class="winlog-badges">${tagBadge}${ptsBadge}</div>
              </div>
              <div class="muted small">
                Anchors: <b>${m.anchor_digit}</b> / <b>${m.backup_digit ?? "—"}</b> • Pairs: <b>${mPairs.join(", ")}</b>
              </div>
            </div>`;
        })
        .join("");

      const labelMap = {
        EXACT: "★ GOLD: EXACT",
        REVERSE: "◆ EMERALD: REVERSE (R)",
        ANCHOR_HIT: "● BLUE: ANCHOR",
        MISS: "MISS",
      };

      return `
        <div class="hist-row ${h.outcome}">
          <div class="hist-row-head">
            <div>
              <strong>${shortDate} (${dayFull}) • ${h.session}</strong> — Actual: <b style="font-size:16px">${actStr}</b>
              <div class="muted small">
                Consensus Anchors: <b>${h.primary_anchor?.digit}</b> & <b>${h.secondary_anchor?.digit}</b> • Consensus Pairs: <b>${pairs}</b>
              </div>
            </div>
            <span class="outcome-tag ${h.outcome}">${labelMap[h.outcome] || h.outcome}</span>
          </div>
          ${allModelsHtml ? `<div class="winlog-models-grid">${allModelsHtml}</div>` : ""}
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
      const st = stats[mName] || {
        evaluated: 0, total_points: 0, anchor_hits: 0, exact_hits: 0, reverse_hits: 0, pos_pair_hits: 0
      };
      const evalCnt = st.evaluated || 0;
      const totPts = Number(st.total_points || 0);
      const avgPts = evalCnt ? (totPts / evalCnt).toFixed(2) : "—";
      const rate = evalCnt ? ((st.anchor_hits / evalCnt) * 100).toFixed(1) + "%" : "—";

      return `
        <div class="model-stat-card">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong>${mName.replace("_", " ")}</strong>
            <span class="violet-chip">${(w * 100).toFixed(1)}% wt</span>
          </div>
          <div class="muted small" style="margin-top:7px">
            Score: <b>${totPts.toFixed(1)} pts</b> (Avg: <b>${avgPts}</b>/draw)<br/>
            Dual-Anchor Hit: <b>${rate}</b> (${st.anchor_hits}/${evalCnt})<br/>
            Exact: <b>${st.exact_hits}</b> • Rev: <b>${st.reverse_hits}</b> • Pos Pair: <b>${st.pos_pair_hits || 0}</b>
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
  document.getElementById("undoLastDrawBtn").addEventListener("click", handleUndoLastDraw);
  document.getElementById("refreshBtn").addEventListener("click", fetchRepoData);

  document.getElementById("prevWinWeekBtn").addEventListener("click", () => {
    if (STATE.winLogWeekIndex < STATE.winLogWeeks.length - 1) {
      STATE.winLogWeekIndex += 1;
      renderWinLog();
    }
  });
  document.getElementById("nextWinWeekBtn").addEventListener("click", () => {
    if (STATE.winLogWeekIndex > 0) {
      STATE.winLogWeekIndex -= 1;
      renderWinLog();
    }
  });

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
    fetchRepoData();
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
