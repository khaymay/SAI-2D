const REPO_OWNER = "saihtet2012";
const REPO_NAME = "SAI-2D";
const BRANCH = "main";
const WORKFLOW_FILE = "update.yml";

const MODEL_LABELS = {
  TabPFN_3.5: "TabPFN 3.5",
  LightGBM: "LightGBM",
  CatBoost: "CatBoost",
  L2_Logistic: "L2 Logistic",
  Bayesian_Markov: "Bayesian Markov",
};

let drawsData = [];
let predData = null;
let droughtMode = "any_pos_gaps";
let topWeekOffset = 0; // 0 = This Week, 1 = Last Week
let winlogWeekIdx = 0;
let winlogOutcomeFilter = "ALL";

function getPat() {
  return localStorage.getItem("sai2d_gh_pat") || "";
}

function parseDateMDY(str) {
  const [m, d, y] = str.split("/").map(Number);
  return new Date(y, m - 1, d);
}

function formatShortDate(dt) {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${dt.getDate()} ${months[dt.getMonth()]}`;
}

function getMondayOfWeek(dt) {
  const copy = new Date(dt.getFullYear(), dt.getMonth(), dt.getDate());
  const day = copy.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  copy.setDate(copy.getDate() + diff);
  return copy;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const obj = {};
    headers.forEach((h, i) => {
      obj[h.trim()] = (cols[i] || "").trim();
    });
    return obj;
  });
}

function toCSV(rows) {
  const headers = ["Session_ID", "Year", "Day_of_Year", "Date", "Is_PM", "Winning_Number"];
  const lines = [headers.join(",")];
  rows.forEach((r) => {
    lines.push(headers.map((h) => (r[h] === undefined || r[h] === null ? "" : r[h])).join(","));
  });
  return lines.join("\n") + "\n";
}

async function fetchData() {
  const ts = Date.now();
  const [csvRes, jsonRes] = await Promise.all([
    fetch(`data/draws.csv?v=${ts}`),
    fetch(`data/predictions.json?v=${ts}`),
  ]);
  const csvText = await csvRes.text();
  drawsData = parseCSV(csvText);
  predData = await jsonRes.json();
  renderAll();
}

function renderAll() {
  if (!predData) return;
  document.getElementById("lastUpdatedPill").textContent = predData.updated_at || "Ready";
  renderWeekStrip();
  renderDroughtRadar();
  renderUpcoming();
  renderHistory();
  renderWinLog();
  renderWeights();
  populateLoggerSlots();
}

/* 1. Slideable This Week / Last Week Winning Numbers Strip */
function renderWeekStrip() {
  const completed = drawsData.filter((r) => r.Winning_Number !== "");
  const referenceRow = completed.length ? completed[completed.length - 1] : drawsData[0];
  const refDate = parseDateMDY(referenceRow.Date);
  const thisMonday = getMondayOfWeek(refDate);

  const targetMonday = new Date(thisMonday);
  targetMonday.setDate(thisMonday.getDate() - topWeekOffset * 7);
  const targetFriday = new Date(targetMonday);
  targetFriday.setDate(targetMonday.getDate() + 4);

  document.getElementById("weekStripTitle").textContent =
    topWeekOffset === 0 ? "Winning Numbers • This Week" : "Winning Numbers • Last Week";
  document.getElementById("weekStripRange").textContent =
    `${formatShortDate(targetMonday)} – ${formatShortDate(targetFriday)} ${targetFriday.getFullYear()} (Swipe or toggle)`;

  document.getElementById("btnThisWeek").classList.toggle("active", topWeekOffset === 0);
  document.getElementById("btnLastWeek").classList.toggle("active", topWeekOffset === 1);

  const dayNames = ["Mon", "Tue", "Wed", "Thu", "Fri"];
  const holidaysMap = predData.thai_holidays || {};
  const grid = document.getElementById("weekStripGrid");
  grid.innerHTML = "";

  for (let i = 0; i < 5; i++) {
    const d = new Date(targetMonday);
    d.setDate(targetMonday.getDate() + i);
    const mdy = `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
    const holName = holidaysMap[mdy];
    const dayRows = drawsData.filter((r) => r.Date === mdy);

    const amRow = dayRows.find((r) => r.Is_PM === "0");
    const pmRow = dayRows.find((r) => r.Is_PM === "1");

    const renderSlot = (label, row) => {
      if (holName) {
        return `<div class="slot-chip holiday"><div class="slot-session">${label}</div><div class="slot-num holiday-txt">HOL</div></div>`;
      }
      if (!row || row.Winning_Number === "") {
        return `<div class="slot-chip"><div class="slot-session">${label}</div><div class="slot-num pending">--</div></div>`;
      }
      const numStr = String(parseInt(row.Winning_Number, 10)).padStart(2, "0");
      return `<div class="slot-chip"><div class="slot-session">${label}</div><div class="slot-num">${numStr}</div></div>`;
    };

    const col = document.createElement("div");
    col.className = "day-col";
    col.innerHTML = `
      <div class="day-col-head">${dayNames[i]}</div>
      <div class="day-col-date">${formatShortDate(d)}</div>
      ${renderSlot("AM", amRow)}
      ${renderSlot("PM", pmRow)}
    `;
    grid.appendChild(col);
  }
}

/* 2. Drought Radar */
function renderDroughtRadar() {
  const radar = (predData.drought_radar || {})[droughtMode] || {};
  const grid = document.getElementById("droughtGrid");
  grid.innerHTML = "";
  for (let d = 0; d <= 9; d++) {
    const gap = radar[String(d)] ?? 0;
    const isHot = gap >= 7;
    const cell = document.createElement("div");
    cell.className = `drought-cell ${isHot ? "hot" : ""}`;
    cell.innerHTML = `${d}<span>${gap}d</span>`;
    grid.appendChild(cell);
  }
}

/* 3. Upcoming Picks with Core Top-3 vs Extended 4-6 & Confidence Score */
function renderUpcoming() {
  const container = document.getElementById("upcomingContainer");
  container.innerHTML = "";
  const upcoming = predData.upcoming || [];

  upcoming.forEach((slot, idx) => {
    const isLive = idx === 0;
    const conf = slot.confidence || { score: 0, tier: "LOW", label: "LOW / SPLIT" };
    const corePairs = (slot.top_pairs || []).slice(0, 3);
    const extPairs = (slot.top_pairs || []).slice(3, 6);

    const renderPairPills = (list, isCore) =>
      list
        .map(
          (p) =>
            `<div class="pair-pill ${isCore ? "core" : ""}">${p.pair}<span>(${(p.prob * 100).toFixed(2)}%)</span></div>`
        )
        .join("");

    const modelRowsHtml = Object.entries(slot.models || {})
      .map(([mKey, mVal]) => {
        const pairsText = (mVal.top_pairs_prob || [])
          .map((tp) => `${tp.pair} (${(tp.prob * 100).toFixed(1)}%)`)
          .join(", ") || (mVal.top_pairs || []).join(", ");
        return `
          <div class="model-row">
            <div class="model-row-top">
              <span class="model-name">${MODEL_LABELS[mKey] || mKey}</span>
              <span class="model-detail">Anchor: <strong>${mVal.anchor_digit}</strong> / ${mVal.backup_digit}</span>
            </div>
            <div class="model-detail">Top Pairs: ${pairsText}</div>
          </div>
        `;
      })
      .join("");

    const card = document.createElement("article");
    card.className = `pred-card ${isLive ? "live-slot" : ""}`;
    card.innerHTML = `
      <div class="pred-card-head">
        <div class="pred-date-group">
          <span class="pred-date-title">${slot.short_date} (${slot.day_full})</span>
          <span class="badge ${isLive ? "badge-live" : "badge-chained"}">
            ${isLive ? "Next Live Draw" : `Chained #${idx + 1}`}
          </span>
        </div>
        <div class="pred-date-group">
          <span class="conf-badge conf-${conf.tier}">${conf.label} • ${conf.score}/100</span>
          <span class="badge badge-session">${slot.session}</span>
        </div>
      </div>

      <div class="anchors-row">
        <div class="anchor-box primary">
          <div class="anchor-label">Primary Anchor</div>
          <div class="anchor-digit">${slot.primary_anchor.digit}</div>
          <div class="anchor-meta">${(slot.primary_anchor.prob * 100).toFixed(1)}% • ${slot.primary_anchor.votes} Models</div>
        </div>
        <div class="anchor-box">
          <div class="anchor-label">Backup Anchor</div>
          <div class="anchor-digit">${slot.secondary_anchor.digit}</div>
          <div class="anchor-meta">${(slot.secondary_anchor.prob * 100).toFixed(1)}% • ${slot.secondary_anchor.votes} Models</div>
        </div>
      </div>

      <div class="tier-block">
        <div class="tier-header">
          <span class="tier-label-core">★ Core Top-3 Picks</span>
          <span class="tier-sub">Cross: ${(slot.intersection_pairs || []).join(" / ")}</span>
        </div>
        <div class="pairs-row-3">${renderPairPills(corePairs, true)}</div>
      </div>

      ${
        extPairs.length
          ? `<div class="tier-block">
              <div class="tier-header">
                <span class="tier-label-ext">Extended Coverage (Ranks 4–6)</span>
              </div>
              <div class="pairs-row-3">${renderPairPills(extPairs, false)}</div>
            </div>`
          : ""
      }

      <details ${isLive ? "open" : ""}>
        <summary class="breakdown-toggle">5-Model Breakdown & Probabilities</summary>
        <div class="model-rows">${modelRowsHtml}</div>
      </details>
    `;
    container.appendChild(card);
  });
}

/* 4. Full Winning History Tab */
function renderHistory() {
  const completed = drawsData.filter((r) => r.Winning_Number !== "").slice().reverse();
  const years = [...new Set(completed.map((r) => r.Year))];
  const yearSel = document.getElementById("historyYearSelect");
  const monthSel = document.getElementById("historyMonthSelect");

  if (!yearSel.options.length) {
    years.forEach((y) => {
      const opt = document.createElement("option");
      opt.value = y;
      opt.textContent = y;
      yearSel.appendChild(opt);
    });
    const months = ["All Months", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    months.forEach((m, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = m;
      monthSel.appendChild(opt);
    });
  }

  const selYear = yearSel.value || years[0];
  const selMonth = parseInt(monthSel.value || "0", 10);

  const filtered = completed.filter((r) => {
    if (r.Year !== selYear) return false;
    if (selMonth > 0) {
      const m = parseInt(r.Date.split("/")[0], 10);
      if (m !== selMonth) return false;
    }
    return true;
  });

  const list = document.getElementById("historyList");
  list.innerHTML = filtered
    .slice(0, 120)
    .map((r) => {
      const dt = parseDateMDY(r.Date);
      const dayName = dt.toLocaleDateString("en-US", { weekday: "short" });
      const sess = r.Is_PM === "1" ? "PM" : "AM";
      const numStr = String(parseInt(r.Winning_Number, 10)).padStart(2, "0");
      return `
        <div class="history-row">
          <span><strong>${formatShortDate(dt)} ${r.Year}</strong> (${dayName}) • ${sess}</span>
          <span class="slot-num">${numStr}</span>
        </div>
      `;
    })
    .join("");
}

/* 5. Win Log Tab with Weekly Pager, Color-Dotted Filter Chips & Actual Prob Lookup */
function groupHistoryByWeek(historyList) {
  const map = new Map();
  historyList.forEach((item) => {
    const dt = parseDateMDY(item.date);
    const mon = getMondayOfWeek(dt);
    const key = `${mon.getFullYear()}-${mon.getMonth() + 1}-${mon.getDate()}`;
    if (!map.has(key)) {
      const fri = new Date(mon);
      fri.setDate(mon.getDate() + 4);
      map.set(key, {
        key,
        monday: mon,
        label: `Week of ${formatShortDate(mon)} – ${formatShortDate(fri)} ${fri.getFullYear()}`,
        items: [],
      });
    }
    map.get(key).items.push(item);
  });
  return Array.from(map.values()).sort((a, b) => b.monday - a.monday);
}

function renderWinLog() {
  const hist = predData.history || [];
  const totalEval = hist.length;
  const exactCnt = hist.filter((x) => x.outcome === "EXACT").length;
  const revCnt = hist.filter((x) => x.outcome === "REVERSE").length;
  const ancCnt = hist.filter((x) => x.outcome === "ANCHOR_HIT").length;
  const totalWins = exactCnt + revCnt + ancCnt;
  const winRate = totalEval ? ((totalWins / totalEval) * 100).toFixed(1) : "0.0";

  document.getElementById("winlogAllTimeStats").innerHTML = `
    <div class="stat-box"><div class="stat-label">All-Time Evaluated</div><div class="stat-val">${totalEval}</div></div>
    <div class="stat-box"><div class="stat-label">All-Time Wins</div><div class="stat-val green">${totalWins} (${winRate}%)</div></div>
    <div class="stat-box"><div class="stat-label">Exact + Reverse</div><div class="stat-val gold">${exactCnt + revCnt}</div></div>
    <div class="stat-box"><div class="stat-label">Anchor Hits</div><div class="stat-val blue">${ancCnt}</div></div>
  `;

  const weeks = groupHistoryByWeek(hist);
  if (!weeks.length) return;
  if (winlogWeekIdx >= weeks.length) winlogWeekIdx = 0;

  const curWeek = weeks[winlogWeekIdx];
  document.getElementById("winlogWeekTitle").textContent = curWeek.label;
  document.getElementById("winlogWeekSub").textContent = `Week ${winlogWeekIdx + 1} of ${weeks.length}`;
  document.getElementById("winlogPrevWeek").disabled = winlogWeekIdx >= weeks.length - 1;
  document.getElementById("winlogNextWeek").disabled = winlogWeekIdx <= 0;

  const wItems = curWeek.items;
  const wExact = wItems.filter((x) => x.outcome === "EXACT").length;
  const wRev = wItems.filter((x) => x.outcome === "REVERSE").length;
  const wAnc = wItems.filter((x) => x.outcome === "ANCHOR_HIT").length;
  const wMiss = wItems.filter((x) => x.outcome === "MISS").length;
  const wWins = wExact + wRev + wAnc;
  const wRate = wItems.length ? ((wWins / wItems.length) * 100).toFixed(1) : "0.0";

  document.getElementById("winlogWeeklyStats").innerHTML = `
    <div class="stat-box"><div class="stat-label">This Week Draws</div><div class="stat-val">${wItems.length}</div></div>
    <div class="stat-box"><div class="stat-label">Weekly Wins</div><div class="stat-val green">${wWins}/${wItems.length} (${wRate}%)</div></div>
    <div class="stat-box"><div class="stat-label">Exact / Rev</div><div class="stat-val gold">${wExact}/${wRev}</div></div>
    <div class="stat-box"><div class="stat-label">Anchor Hits</div><div class="stat-val blue">${wAnc}</div></div>
  `;

  const chipDefs = [
    { key: "ALL", label: "All", count: wItems.length },
    { key: "EXACT", label: "Exact", count: wExact },
    { key: "REVERSE", label: "Reverse", count: wRev },
    { key: "ANCHOR_HIT", label: "Anchor", count: wAnc },
    { key: "MISS", label: "Miss", count: wMiss },
  ];

  document.getElementById("winlogFilterChips").innerHTML = chipDefs
    .map(
      (c) => `
      <button class="filter-chip ${winlogOutcomeFilter === c.key ? "active" : ""}" data-outcome="${c.key}">
        <span class="filter-dot dot-${c.key}"></span>
        <span>${c.label} (${c.count})</span>
      </button>
    `
    )
    .join("");

  const visibleItems =
    winlogOutcomeFilter === "ALL"
      ? wItems
      : wItems.filter((x) => x.outcome === winlogOutcomeFilter);

  const badgeTextMap = {
    EXACT: "★ GOLD: EXACT",
    REVERSE: "◆ EMERALD: REVERSE (R)",
    ANCHOR_HIT: "● BLUE: ANCHOR",
    MISS: "MISS",
  };

  document.getElementById("winlogCards").innerHTML = visibleItems
    .map((item) => {
      const pairsStr = (item.top_pairs || []).map((p) => p.pair).join(", ");
      const lookup = item.actual_prob_lookup || {};
      const lookupParts = Object.entries(lookup).map(([k, v]) => {
        const name = k === "Consensus" ? "Consensus" : MODEL_LABELS[k] || k;
        return `${name}: <strong>${(v.prob * 100).toFixed(2)}%</strong> (#${v.rank})`;
      });

      const modelsHtml = Object.entries(item.models || {})
        .map(([mKey, mVal]) => {
          const pts = mVal.earned_pts ?? 0;
          const pairsTxt = (mVal.top_pairs_prob || [])
            .map((tp) => `${tp.pair} (${(tp.prob * 100).toFixed(1)}%)`)
            .join(", ") || (mVal.top_pairs || []).join(", ");
          return `
            <div class="model-row">
              <div class="model-row-top">
                <span class="model-name">${MODEL_LABELS[mKey] || mKey}</span>
                <span class="pts-badge ${pts === 0 ? "zero" : ""}">${pts > 0 ? `+${pts.toFixed(1)} pts` : "0.0 pts"}</span>
              </div>
              <div class="model-detail">Anchors: ${mVal.anchor_digit} / ${mVal.backup_digit} • Pairs: ${pairsTxt}</div>
            </div>
          `;
        })
        .join("");

      return `
        <article class="winlog-card outcome-${item.outcome}">
          <div class="pred-date-title">${item.short_date} (${item.day_full}) • ${item.session} — Actual: ${item.actual}</div>
          <div class="model-detail">Consensus Anchors: ${item.primary_anchor.digit} & ${item.secondary_anchor.digit} • Pairs: ${pairsStr}</div>
          <div class="outcome-pill ${item.outcome}">${badgeTextMap[item.outcome] || item.outcome}</div>
          ${
            lookupParts.length
              ? `<div class="actual-lookup-box">Actual <strong>${item.actual}</strong> Model Probabilities: ${lookupParts.join(" • ")}</div>`
              : ""
          }
          <div class="model-rows">${modelsHtml}</div>
        </article>
      `;
    })
    .join("");
}

/* 6. 5-Model Committee Weights Tab */
function renderWeights() {
  const weights = predData.model_weights || {};
  const stats = predData.model_stats || {};
  const container = document.getElementById("weightsContainer");

  container.innerHTML = Object.keys(MODEL_LABELS)
    .map((mKey) => {
      const wt = ((weights[mKey] || 0.2) * 100).toFixed(1);
      const st = stats[mKey] || {};
      const ev = st.evaluated || 0;
      const avg = ev ? (st.total_points / ev).toFixed(2) : "0.00";
      const ancRate = ev ? ((st.anchor_hits / ev) * 100).toFixed(1) : "0.0";

      return `
        <div class="weight-card">
          <div class="weight-top">
            <span class="section-title">${MODEL_LABELS[mKey]}</span>
            <span class="weight-pill">${wt}% wt</span>
          </div>
          <div class="model-detail">Score: <strong>${(st.total_points || 0).toFixed(1)} pts</strong> (Avg: ${avg}/draw)</div>
          <div class="model-detail">Dual-Anchor Hit: <strong>${ancRate}%</strong> (${st.anchor_hits || 0}/${ev})</div>
          <div class="model-detail">Exact: <strong>${st.exact_hits || 0}</strong> • Rev: ${st.reverse_hits || 0} • Pos Pair: ${st.pos_pair_hits || 0}</div>
        </div>
      `;
    })
    .join("");
}

/* 7. Draw Logger + GitHub API Commit */
function populateLoggerSlots() {
  const sel = document.getElementById("slotSelect");
  const openRows = drawsData.filter((r) => r.Winning_Number === "").slice(0, 10);
  sel.innerHTML = openRows
    .map((r) => {
      const dt = parseDateMDY(r.Date);
      const sess = r.Is_PM === "1" ? "PM" : "AM";
      return `<option value="${r.Session_ID}">${formatShortDate(dt)} (${sess}) — Slot #${r.Session_ID}</option>`;
    })
    .join("");
}

async function commitDrawsCsv(updatedRows, message) {
  const pat = getPat();
  if (!pat) {
    document.getElementById("tokenModal").classList.remove("hidden");
    return false;
  }
  const statusEl = document.getElementById("loggerStatus");
  statusEl.textContent = "Saving to GitHub...";

  const apiUrl = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/contents/data/draws.csv`;
  const getRes = await fetch(apiUrl, {
    headers: { Authorization: `token ${pat}`, Accept: "application/vnd.github.v3+json" },
  });
  const fileData = await getRes.json();
  const encodedContent = btoa(unescape(encodeURIComponent(toCSV(updatedRows))));

  const putRes = await fetch(apiUrl, {
    method: "PUT",
    headers: {
      Authorization: `token ${pat}`,
      Accept: "application/vnd.github.v3+json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      content: encodedContent,
      sha: fileData.sha,
      branch: BRANCH,
    }),
  });

  if (!putRes.ok) {
    statusEl.textContent = "GitHub save failed. Check your token permissions.";
    return false;
  }
  statusEl.textContent = "Saved! GitHub Action is updating predictions (~45s)...";
  return true;
}

/* Event Wiring */
document.addEventListener("DOMContentLoaded", () => {
  fetchData();

  // Primary Navigation Tabs
  document.getElementById("primaryNav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-tab");
    if (!btn) return;
    document.querySelectorAll(".nav-tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });

  // This Week / Last Week Toggle + Swipe
  document.querySelectorAll(".week-toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      topWeekOffset = parseInt(btn.dataset.weekOffset, 10);
      renderWeekStrip();
    });
  });

  let touchStartX = 0;
  const stripCard = document.getElementById("weekStripCard");
  stripCard.addEventListener("touchstart", (e) => {
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });
  stripCard.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].screenX - touchStartX;
    if (dx > 45 && topWeekOffset === 0) {
      topWeekOffset = 1;
      renderWeekStrip();
    } else if (dx < -45 && topWeekOffset === 1) {
      topWeekOffset = 0;
      renderWeekStrip();
    }
  }, { passive: true });

  // Drought Radar Tabs
  document.getElementById("droughtTabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".drought-tab");
    if (!btn) return;
    document.querySelectorAll(".drought-tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    droughtMode = btn.dataset.mode;
    renderDroughtRadar();
  });

  // Win Log Week Pager & Filter Chips
  document.getElementById("winlogPrevWeek").addEventListener("click", () => {
    winlogWeekIdx += 1;
    renderWinLog();
  });
  document.getElementById("winlogNextWeek").addEventListener("click", () => {
    winlogWeekIdx = Math.max(0, winlogWeekIdx - 1);
    renderWinLog();
  });
  document.getElementById("winlogFilterChips").addEventListener("click", (e) => {
    const chip = e.target.closest(".filter-chip");
    if (!chip) return;
    winlogOutcomeFilter = chip.dataset.outcome;
    renderWinLog();
  });

  // History Filters
  document.getElementById("historyYearSelect").addEventListener("change", renderHistory);
  document.getElementById("historyMonthSelect").addEventListener("change", renderHistory);

  // Save Draw Result
  document.getElementById("saveDrawBtn").addEventListener("click", async () => {
    const sid = document.getElementById("slotSelect").value;
    const rawVal = document.getElementById("winningInput").value.trim();
    if (!/^\d{1,2}$/.test(rawVal)) {
      document.getElementById("loggerStatus").textContent = "Enter a valid 2-digit number (00–99).";
      return;
    }
    const numVal = String(parseInt(rawVal, 10));
    const updated = drawsData.map((r) => (r.Session_ID === sid ? { ...r, Winning_Number: numVal } : r));
    const ok = await commitDrawsCsv(updated, `Log draw result for Slot #${sid}: ${numVal.padStart(2, "0")}`);
    if (ok) {
      drawsData = updated;
      document.getElementById("winningInput").value = "";
      renderAll();
    }
  });

  // Undo Last Completed Draw
  document.getElementById("undoLastBtn").addEventListener("click", async () => {
    const completed = drawsData.filter((r) => r.Winning_Number !== "");
    if (!completed.length) return;
    const lastRow = completed[completed.length - 1];
    const updated = drawsData.map((r) =>
      r.Session_ID === lastRow.Session_ID ? { ...r, Winning_Number: "" } : r
    );
    const ok = await commitDrawsCsv(updated, `Undo last draw Slot #${lastRow.Session_ID}`);
    if (ok) {
      drawsData = updated;
      renderAll();
    }
  });

  // Refresh & Token Modal
  document.getElementById("refreshBtn").addEventListener("click", fetchData);
  document.getElementById("tokenBtn").addEventListener("click", () => {
    document.getElementById("patInput").value = getPat();
    document.getElementById("tokenModal").classList.remove("hidden");
  });
  document.getElementById("closeModalBtn").addEventListener("click", () => {
    document.getElementById("tokenModal").classList.add("hidden");
  });
  document.getElementById("savePatBtn").addEventListener("click", () => {
    localStorage.setItem("sai2d_gh_pat", document.getElementById("patInput").value.trim());
    document.getElementById("tokenModal").classList.add("hidden");
  });
});
