const STATE = {
  predictions: null,
  drawsCsvRaw: "",
  drawsRows: [],
  activeRadar: "any_pos_gaps",
  activeFilter: "ALL",
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
      populateSessionDropdown();
    }

    if (predRes.ok) {
      STATE.predictions = await predRes.json();
      syncEl.textContent = `Updated: ${STATE.predictions.updated_at || "Ready"}`;
      renderAll();
    } else {
      syncEl.textContent = "Waiting for first Action run";
      document.getElementById("upcomingGrid").innerHTML = `
        <div class="card">
          <h3>No predictions.json yet</h3>
          <p class="muted">Go to your GitHub repo → <b>Actions</b> → <b>SAI 2D Prediction Engine</b> → <b>Run workflow</b> to generate your initial 5-model forecast.</p>
        </div>`;
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
    const numStr = String(parseInt(last.Winning_Number, 10)).padStart(2, "0");
    document.getElementById("latestBadge").textContent =
      `Latest: #${last.Session_ID} (${last.Date} ${last.Is_PM === "1" ? "PM" : "AM"}) = ${numStr}`;
  }

  openRows.slice(0, 10).forEach((r) => {
    const opt = document.createElement("option");
    opt.value = r.Session_ID;
    opt.textContent = `Session #${r.Session_ID} — ${r.Date} (${r.Is_PM === "1" ? "PM 4:30" : "AM 12:01"})`;
    select.appendChild(opt);
  });
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
  feedback.textContent = `Committing Session #${sid} = ${numFormatted} to GitHub...`;

  try {
    const fileData = await githubGetFile("data/draws.csv");
    const latestCsv = decodeBase64Unicode(fileData.content);
    const rows = parseCsv(latestCsv);
    const target = rows.find((r) => r.Session_ID === String(sid));
    if (!target) throw new Error(`Session #${sid} not found in data/draws.csv`);

    target.Winning_Number = String(numInt);
    const updatedCsv = toCsv(rows);

    await githubPutFile(
      "data/draws.csv",
      updatedCsv,
      fileData.sha,
      `feat(draw): log Session #${sid} = ${numFormatted}`
    );

    document.getElementById("winningInput").value = "";
    feedback.textContent = `Saved Session #${sid} = ${numFormatted}! GitHub Actions is now running the 5-model update (~30s). Click Refresh shortly.`;
    STATE.drawsRows = rows;
    populateSessionDropdown();
  } catch (err) {
    feedback.textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
}

async function toggleLockPrediction(sessionId) {
  const feedback = document.getElementById("loggerFeedback");
  try {
    feedback.textContent = `Locking prediction snapshot for Session #${sessionId}...`;
    const fileData = await githubGetFile("data/predictions.json");
    const predObj = JSON.parse(decodeBase64Unicode(fileData.content));
    const slot = (predObj.upcoming || []).find((u) => Number(u.session_id) === Number(sessionId));
    if (!slot) throw new Error("Upcoming session not found.");
    slot.locked = !slot.locked;

    await githubPutFile(
      "data/predictions.json",
      JSON.stringify(predObj, null, 2),
      fileData.sha,
      `chore: ${slot.locked ? "lock" : "unlock"} prediction for Session #${sessionId}`
    );
    STATE.predictions = predObj;
    renderUpcoming();
    feedback.textContent = `Session #${sessionId} prediction ${slot.locked ? "LOCKED" : "unlocked"}.`;
  } catch (err) {
    feedback.textContent = `Lock failed: ${err.message}`;
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
          <div class="radar-gap">${gap}d gap</div>
        </div>`;
    })
    .join("");
}

function renderUpcoming() {
  const container = document.getElementById("upcomingGrid");
  const list = STATE.predictions?.upcoming || [];
  if (!list.length) {
    container.innerHTML = `<p class="muted">No upcoming predictions available.</p>`;
    return;
  }

  container.innerHTML = list
    .map((item) => {
      const p1 = item.primary_anchor;
      const p2 = item.secondary_anchor;
      const pairsHtml = (item.top_pairs || [])
        .map(
          (p) => `
          <div class="pair-badge" title="Supporters: ${(p.supporters || []).join(", ")}">
            ${p.pair} <span>(${(p.prob * 100).toFixed(2)}%)</span>
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
        <article class="pred-card ${item.locked ? "is-locked" : ""}">
          <div class="card-header" style="margin-bottom:0">
            <div>
              <strong>Session #${item.session_id}</strong>
              <div class="muted small">${item.date} (${item.day}) • <b>${item.session}</b></div>
            </div>
            <button class="btn btn-outline btn-sm" onclick="toggleLockPrediction(${item.session_id})">
              ${item.locked ? "🔒 Locked" : "🔓 Lock Pick"}
            </button>
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

          <div>
            <button class="btn btn-ghost btn-sm" onclick="document.getElementById('wheel-${item.session_id}').classList.toggle('hidden')">
              Toggle 19-Number Wheel (Digit ${p1.digit})
            </button>
            <div id="wheel-${item.session_id}" class="wheel-box hidden" style="margin-top:8px">
              ${(p1.wheel_19 || []).join("  •  ")}
            </div>
          </div>

          <details>
            <summary>5-Model Breakdown</summary>
            <div class="model-mini-list">${modelsHtml}</div>
          </details>
        </article>`;
    })
    .join("");
}

function renderHistory() {
  const hist = STATE.predictions?.history || [];
  const strip = document.getElementById("scoreSummaryStrip");
  const listEl = document.getElementById("historyList");

  const total = hist.length;
  const exact = hist.filter((h) => h.outcome === "EXACT").length;
  const rev = hist.filter((h) => h.outcome === "REVERSE").length;
  const anchor = hist.filter((h) => h.outcome === "ANCHOR_HIT").length;
  const anyWin = exact + rev + anchor;

  strip.innerHTML = `
    <div class="stat-box"><div class="muted small">Scored Sessions</div><div class="num">${total}</div></div>
    <div class="stat-box"><div class="muted small">Total Wins (Pair + Digit)</div><div class="num">${anyWin} (${total ? ((anyWin / total) * 100).toFixed(1) : 0}%)</div></div>
    <div class="stat-box"><div class="muted small">Exact + Reverse (R)</div><div class="num" style="color:#fbbf24">${exact + rev}</div></div>
    <div class="stat-box"><div class="muted small">Anchor Digit Hits</div><div class="num" style="color:#38bdf8">${anchor}</div></div>
  `;

  const filtered = STATE.activeFilter === "ALL" ? hist : hist.filter((h) => h.outcome === STATE.activeFilter);
  if (!filtered.length) {
    listEl.innerHTML = `<p class="muted">No scored predictions in this filter yet. Once you log a draw for an upcoming session, it will appear here with Gold/Emerald/Blue win highlights.</p>`;
    return;
  }

  listEl.innerHTML = filtered
    .map((h) => {
      const pairs = (h.top_pairs || []).map((p) => p.pair).join(", ");
      const labelMap = {
        EXACT: "★ GOLD: EXACT PAIR",
        REVERSE: "◆ EMERALD: REVERSE (R)",
        ANCHOR_HIT: "● BLUE: ANCHOR DIGIT",
        MISS: "MISS",
      };
      return `
        <div class="hist-row ${h.outcome}">
          <div>
            <strong>Session #${h.session_id}</strong> (${h.date} ${h.session}) — Actual: <b style="font-size:16px">${h.actual}</b>
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

function renderAll() {
  renderRadar();
  renderUpcoming();
  renderHistory();
  renderModels();
}

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  const cfg = getConfig();
  if (cfg.owner) document.getElementById("ghOwner").value = cfg.owner;
  if (cfg.repo) document.getElementById("ghRepo").value = cfg.repo;
  if (cfg.branch) document.getElementById("ghBranch").value = cfg.branch;
  if (cfg.token) document.getElementById("ghToken").value = cfg.token;

  document.getElementById("logDrawForm").addEventListener("submit", handleLogDraw);
  document.getElementById("refreshBtn").addEventListener("click", fetchRepoData);

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
      renderHistory();
    });
  });

  fetchRepoData();
});
