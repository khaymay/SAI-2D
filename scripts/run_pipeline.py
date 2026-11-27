import json
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import holidays
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

DRAWS_PATH = "data/draws.csv"
PRED_PATH = "data/predictions.json"
MODEL_NAMES = ["TabPFN_3.5", "LightGBM", "CatBoost", "L2_Logistic", "Bayesian_Markov"]
YANGON_TZ = timezone(timedelta(hours=6, minutes=30))
ROLLING_WINDOW = 600

# Precomputed (10, 10) lookup matrix of Brake = (t + u) % 10
BRAKE_IDX = (np.arange(10)[:, None] + np.arange(10)[None, :]) % 10


def get_thai_holiday_map(years: list[int]) -> dict[str, str]:
    """Returns {'M/D/YYYY': 'Holiday Name'} for Thailand / SET market holidays."""
    th_holidays = holidays.Thailand(years=years)
    out = {}
    for dt_obj, name in th_holidays.items():
        key = f"{dt_obj.month}/{dt_obj.day}/{dt_obj.year}"
        out[key] = str(name)
    return out


def format_short_date(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    return f"{dt.day} {dt.strftime('%b')}"


def format_day_name(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%A")


def format_day_short(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%a")


def default_model_stat() -> dict:
    return {
        "evaluated": 0,
        "total_points": 0.0,
        "anchor_hits": 0,
        "both_anchor_hits": 0,
        "exact_hits": 0,
        "reverse_hits": 0,
        "pos_pair_hits": 0,
        "any_pair_hits": 0,
    }


def load_existing_predictions() -> dict:
    if os.path.exists(PRED_PATH):
        with open(PRED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "app_name": "SAI 2D",
        "updated_at": None,
        "updated_at_iso": None,
        "thai_holidays": {},
        "model_weights": {m: 0.20 for m in MODEL_NAMES},
        "model_stats": {m: default_model_stat() for m in MODEL_NAMES},
        "drought_radar": {},
        "upcoming": [],
        "history": [],
    }


def ensure_future_slots(df: pd.DataFrame, th_map: dict[str, str], min_open_slots: int = 10) -> pd.DataFrame:
    """
    Removes any unplayed open rows that fall on Thai SET public holidays,
    then generates at least `min_open_slots` (10 = full 5-day week) valid trading slots.
    """
    is_open = df["Winning_Number"].isna()
    is_hol = df["Date"].astype(str).isin(th_map.keys())
    cleaned = df[~(is_open & is_hol)].copy().reset_index(drop=True)

    open_count = int(cleaned["Winning_Number"].isna().sum())
    if open_count >= min_open_slots and len(cleaned) == len(df):
        return cleaned

    rows = cleaned.to_dict("records")
    last = rows[-1]
    cur_sid = int(last["Session_ID"])
    cur_date = pd.to_datetime(last["Date"])
    cur_pm = int(last["Is_PM"])

    while sum(1 for r in rows if pd.isna(r["Winning_Number"])) < min_open_slots:
        cur_sid += 1
        if cur_pm == 0:
            cur_pm = 1
        else:
            cur_pm = 0
            cur_date += pd.Timedelta(days=1)
            while True:
                date_key = f"{cur_date.month}/{cur_date.day}/{cur_date.year}"
                if cur_date.weekday() < 5 and date_key not in th_map:
                    break
                cur_date += pd.Timedelta(days=1)

        date_str = f"{cur_date.month}/{cur_date.day}/{cur_date.year}"
        rows.append({
            "Session_ID": cur_sid,
            "Year": int(cur_date.year),
            "Day_of_Year": int(cur_date.dayofyear),
            "Date": date_str,
            "Is_PM": cur_pm,
            "Winning_Number": np.nan,
        })

    out = pd.DataFrame(rows)
    out["Session_ID"] = np.arange(1, len(out) + 1)
    out.to_csv(DRAWS_PATH, index=False)
    return out


def build_features(data: pd.DataFrame):
    """
    Returns (full_feature_df, tree_feature_cols, tabpfn_hybrid_cols).
    TabPFN-3.5 gets a 28-feature hybrid matrix (18 temporal/lag features + 10 DoW x Session
    any-position base rates) to prevent static anchor repetition while preserving token efficiency.
    """
    d = data.copy()
    d["Date_dt"] = pd.to_datetime(d["Date"])
    d["DayOfWeek"] = d["Date_dt"].dt.dayofweek
    nums = d["Winning_Number"].values
    dows = d["DayOfWeek"].values
    pms = d["Is_PM"].values
    n = len(d)

    d["sin_doy"] = np.sin(2 * np.pi * d["Day_of_Year"] / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * d["Day_of_Year"] / 365.25)

    for lag in [1, 2, 3, 4, 5, 10]:
        s = pd.Series(nums).shift(lag)
        d[f"lag_{lag}_num"] = s
        d[f"lag_{lag}_tens"] = np.floor(s / 10.0)
        d[f"lag_{lag}_units"] = s % 10
        d[f"lag_{lag}_brake"] = (d[f"lag_{lag}_tens"] + d[f"lag_{lag}_units"]) % 10

    d["lag1_t_x_pm"] = d["lag_1_tens"] * d["Is_PM"]
    d["lag1_u_x_pm"] = d["lag_1_units"] * d["Is_PM"]
    d["lag1_is_double"] = (d["lag_1_tens"] == d["lag_1_units"]).astype(float)

    tens_roll15, units_roll15 = np.zeros((n, 10)), np.zeros((n, 10))
    tens_roll50, units_roll50 = np.zeros((n, 10)), np.zeros((n, 10))
    recent_carry = np.zeros((n, 10))
    hist_t, hist_u = [], []

    dow_t_rate = np.zeros((n, 10))
    dow_u_rate = np.zeros((n, 10))
    dow_any_rate = np.zeros((n, 10))
    cnt_t = np.zeros((7, 2, 10))
    cnt_u = np.zeros((7, 2, 10))
    cnt_any = np.zeros((7, 2, 10))
    cnt_slot = np.zeros((7, 2))
    ALPHA_POS = 2.0
    ALPHA_ANY = 3.8
    PRIOR_N = 20.0

    for i in range(n):
        if hist_t:
            rt15, ru15 = hist_t[-15:], hist_u[-15:]
            rt50, ru50 = hist_t[-50:], hist_u[-50:]
            last_digits = set(hist_t[-2:] + hist_u[-2:])
            for k in range(10):
                tens_roll15[i, k] = rt15.count(k) / len(rt15)
                units_roll15[i, k] = ru15.count(k) / len(ru15)
                tens_roll50[i, k] = rt50.count(k) / len(rt50)
                units_roll50[i, k] = ru50.count(k) / len(ru50)
                recent_carry[i, k] = 1.0 if k in last_digits else 0.0

        w = int(dows[i])
        pm = int(pms[i])
        denom = cnt_slot[w, pm] + PRIOR_N
        dow_t_rate[i] = (cnt_t[w, pm] + ALPHA_POS) / denom
        dow_u_rate[i] = (cnt_u[w, pm] + ALPHA_POS) / denom
        dow_any_rate[i] = (cnt_any[w, pm] + ALPHA_ANY) / denom

        if not np.isnan(nums[i]):
            val = int(nums[i])
            t_val, u_val = val // 10, val % 10
            hist_t.append(t_val)
            hist_u.append(u_val)

            cnt_t[w, pm, t_val] += 1.0
            cnt_u[w, pm, u_val] += 1.0
            cnt_any[w, pm, t_val] += 1.0
            if u_val != t_val:
                cnt_any[w, pm, u_val] += 1.0
            cnt_slot[w, pm] += 1.0

    for k in range(10):
        d[f"tens_roll15_{k}"] = tens_roll15[:, k]
        d[f"units_roll15_{k}"] = units_roll15[:, k]
        d[f"tens_roll50_{k}"] = tens_roll50[:, k]
        d[f"units_roll50_{k}"] = units_roll50[:, k]
        d[f"recent_carry_{k}"] = recent_carry[:, k]
        d[f"dow_pm_tens_rate_{k}"] = dow_t_rate[:, k]
        d[f"dow_pm_units_rate_{k}"] = dow_u_rate[:, k]
        d[f"dow_pm_any_rate_{k}"] = dow_any_rate[:, k]

    ignore_cols = {"Session_ID", "Date", "Date_dt", "Winning_Number"}
    tree_cols = [c for c in d.columns if c not in ignore_cols]

    tabpfn_hybrid_cols = [
        "Is_PM", "DayOfWeek", "sin_doy", "cos_doy",
        "lag_1_tens", "lag_1_units", "lag_1_brake",
        "lag_2_tens", "lag_2_units", "lag_2_brake",
        "lag_3_tens", "lag_3_units", "lag_3_brake",
        "lag_4_num", "lag_5_num",
        "lag1_t_x_pm", "lag1_u_x_pm", "lag1_is_double",
    ] + [f"dow_pm_any_rate_{k}" for k in range(10)]

    return d, tree_cols, tabpfn_hybrid_cols


def compute_drought_radar(hist: pd.DataFrame) -> dict:
    last_sid = int(hist["Session_ID"].max())
    tens_s = hist["Winning_Number"].astype(int) // 10
    units_s = hist["Winning_Number"].astype(int) % 10
    brake_s = (tens_s + units_s) % 10

    return {
        "tens_gaps": {
            str(d): int(last_sid - hist.loc[tens_s == d, "Session_ID"].max())
            for d in range(10)
        },
        "units_gaps": {
            str(d): int(last_sid - hist.loc[units_s == d, "Session_ID"].max())
            for d in range(10)
        },
        "brake_gaps": {
            str(d): int(last_sid - hist.loc[brake_s == d, "Session_ID"].max())
            for d in range(10)
        },
        "any_pos_gaps": {
            str(d): int(last_sid - hist.loc[(tens_s == d) | (units_s == d), "Session_ID"].max())
            for d in range(10)
        },
    }


def combine_t_u_brake(pt: np.ndarray, pu: np.ndarray, pb: np.ndarray) -> np.ndarray:
    grid = (np.outer(pt, pu) * (pb[BRAKE_IDX] * 10.0)).flatten()
    grid = np.maximum(grid, 1e-8)
    return grid / grid.sum()


def predict_bayesian_pure_transition(hist: pd.DataFrame, target_row: pd.Series) -> np.ndarray:
    """Diverse Model 5: Pure Day-of-Week x Session + Intraday/Overnight Markov Transition."""
    nums = hist["Winning_Number"].astype(int).values
    tens = nums // 10
    units = nums % 10

    d_dt = pd.to_datetime(hist["Date"])
    dow_hist = d_dt.dt.dayofweek.values
    is_pm_hist = hist["Is_PM"].values
    t_dow = pd.to_datetime(target_row["Date"]).dayofweek
    t_pm = int(target_row["Is_PM"])

    mask = (dow_hist == t_dow) & (is_pm_hist == t_pm)
    sub_nums = nums[mask]
    p_exact_dow = (np.bincount(sub_nums, minlength=100) + 0.60) / (len(sub_nums) + 60.0)

    last_num = int(nums[-1])
    last_t, last_u = last_num // 10, last_num % 10
    t_mat = np.ones((10, 10)) * 2.0
    u_mat = np.ones((10, 10)) * 2.0
    trans_mask = is_pm_hist[1:] == t_pm
    np.add.at(t_mat, (tens[:-1][trans_mask], tens[1:][trans_mask]), 1.0)
    np.add.at(u_mat, (units[:-1][trans_mask], units[1:][trans_mask]), 1.0)
    t_mat /= t_mat.sum(axis=1, keepdims=True)
    u_mat /= u_mat.sum(axis=1, keepdims=True)
    p_markov = np.outer(t_mat[last_t], u_mat[last_u]).flatten()

    p_intraday = np.ones(100)
    if t_pm == 1:
        am_digits = {last_t, last_u}
        for num in range(100):
            if (num // 10) in am_digits or (num % 10) in am_digits:
                p_intraday[num] = 1.25
    p_intraday /= p_intraday.sum()

    grid = 0.45 * p_exact_dow + 0.40 * p_markov + 0.15 * p_intraday
    grid = np.maximum(grid, 1e-6)
    return grid / grid.sum()


def fit_ml_models(
    X_train_full: pd.DataFrame,
    X_train_hybrid: pd.DataFrame,
    y_num: np.ndarray,
):
    """Fits 3 heads (Tens, Units, Modulo-10 Brake); TabPFN-3.5 uses the 28-feature hybrid matrix."""
    y_t = y_num // 10
    y_u = y_num % 10
    y_b = (y_t + y_u) % 10
    models = {}

    lgb_heads = []
    for target in (y_t, y_u, y_b):
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=10, n_estimators=100,
            learning_rate=0.04, num_leaves=15, subsample_for_bin=500,
            min_child_samples=20, n_jobs=-1, random_state=42, verbose=-1
        )
        clf.fit(X_train_full, target)
        lgb_heads.append(clf)
    models["LightGBM"] = tuple(lgb_heads)

    cb_heads = []
    for target in (y_t, y_u, y_b):
        clf = CatBoostClassifier(
            iterations=150, learning_rate=0.05, depth=4,
            loss_function="MultiClass", thread_count=-1, random_seed=42, verbose=0
        )
        clf.fit(X_train_full, target)
        cb_heads.append(clf)
    models["CatBoost"] = tuple(cb_heads)

    lr_heads = []
    for target in (y_t, y_u, y_b):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=600))
        clf.fit(X_train_full, target)
        lr_heads.append(clf)
    models["L2_Logistic"] = tuple(lr_heads)

    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if token:
        try:
            from tabpfn_client import TabPFNClassifier, set_access_token
            from tabpfn_client.api_models import ModelVersion
            set_access_token(token)
            tp_heads = []
            for target in (y_t, y_u, y_b):
                clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)
                clf.fit(X_train_hybrid, target)
                tp_heads.append(clf)
            models["TabPFN_3.5"] = tuple(tp_heads)
        except Exception as exc:
            print(f"[WARN] TabPFN-3.5 skipped during fit: {exc}")

    return models


def grid_to_anchor_probs(p_100: np.ndarray) -> np.ndarray:
    mat = p_100.reshape(10, 10)
    p_t = mat.sum(axis=1)
    p_u = mat.sum(axis=0)
    return p_t + p_u - (p_t * p_u)


def evaluate_model_draw(m_info: dict, actual: int) -> dict:
    act_t, act_u = actual // 10, actual % 10
    act_str = f"{actual:02d}"
    rev_str = f"{act_u}{act_t}"
    act_digits = {act_t, act_u}

    a1 = int(m_info["anchor_digit"])
    a2 = int(m_info.get("backup_digit", -1))
    hit_a1 = a1 in act_digits
    hit_a2 = a2 in act_digits

    if hit_a1 and hit_a2 and a1 != a2:
        anchor_pts = 1.5
    elif hit_a1 or hit_a2:
        anchor_pts = 1.0
    else:
        anchor_pts = 0.0

    pair_pts = 0.0
    exact_cnt = 0
    rev_cnt = 0
    pos_cnt = 0
    any_cnt = 0

    for pair_str in m_info.get("top_pairs", [])[:3]:
        p_val = int(pair_str)
        pt, pu = p_val // 10, p_val % 10
        if pair_str == act_str:
            pair_pts += 3.0
            exact_cnt += 1
        elif pair_str == rev_str and act_str != rev_str:
            pair_pts += 2.0
            rev_cnt += 1
        elif pt == act_t or pu == act_u:
            pair_pts += 0.2
            pos_cnt += 1
        elif pt in act_digits or pu in act_digits:
            pair_pts += 0.1
            any_cnt += 1

    earned = round(anchor_pts + pair_pts, 2)
    return {
        "earned": earned,
        "anchor_pts": round(anchor_pts, 2),
        "pair_pts": round(pair_pts, 2),
        "hit_any_anchor": bool(hit_a1 or hit_a2),
        "hit_both_anchors": bool(hit_a1 and hit_a2 and a1 != a2),
        "exact_cnt": exact_cnt,
        "rev_cnt": rev_cnt,
        "pos_cnt": pos_cnt,
        "any_cnt": any_cnt,
    }


def apply_weight_guardrails(raw_weights: dict[str, float], floor: float = 0.14, cap: float = 0.28) -> dict[str, float]:
    """
    Clips committee weights to [14%, 28%] so no single model can override a 3-model majority,
    and re-normalizes so weights sum cleanly to 1.0.
    """
    w = {m: float(v) for m, v in raw_weights.items()}
    s = sum(w.values())
    w = {m: v / s for m, v in w.items()}

    for _ in range(10):
        excess = 0.0
        deficit = 0.0
        free_keys = []
        for m, val in w.items():
            if val > cap:
                excess += val - cap
                w[m] = cap
            elif val < floor:
                deficit += floor - val
                w[m] = floor
            else:
                free_keys.append(m)

        net = excess - deficit
        if abs(net) < 1e-9 or not free_keys:
            break
        free_sum = sum(w[m] for m in free_keys)
        if free_sum <= 0:
            break
        for m in free_keys:
            w[m] += net * (w[m] / free_sum)

    final_s = sum(w.values())
    return {m: round(w[m] / final_s, 4) for m in raw_weights}


def grade_consensus_item(item: dict, row: pd.Series) -> dict:
    """Computes consensus outcome, actual string, anchor flags, and full grid actual lookup."""
    actual = int(row["Winning_Number"])
    act_str = f"{actual:02d}"
    rev_str = f"{actual % 10}{actual // 10}"
    act_digits = {actual // 10, actual % 10}

    top_pairs = [p["pair"] for p in item.get("top_pairs", [])]
    anchor1 = int(item["primary_anchor"]["digit"])
    anchor2 = int(item["secondary_anchor"]["digit"])

    if act_str in top_pairs[:6]:
        outcome = "EXACT"
    elif rev_str in top_pairs[:6] and act_str != rev_str:
        outcome = "REVERSE"
    elif anchor1 in act_digits or anchor2 in act_digits:
        outcome = "ANCHOR_HIT"
    else:
        outcome = "MISS"

    item["short_date"] = format_short_date(str(row["Date"]))
    item["day_full"] = format_day_name(str(row["Date"]))
    item["day"] = format_day_short(str(row["Date"]))
    item["actual"] = act_str
    item["outcome"] = outcome
    item["hit_primary_anchor"] = bool(anchor1 in act_digits)
    item["hit_secondary_anchor"] = bool(anchor2 in act_digits)

    # Lookup actual winning number's predicted probability & rank across stored grid_100
    actual_lookup = {}
    c_grid = item.get("consensus_grid_100")
    if isinstance(c_grid, list) and len(c_grid) == 100:
        c_prob = float(c_grid[actual])
        c_rank = int(sum(1 for p in c_grid if p > c_prob) + 1)
        actual_lookup["Consensus"] = {"prob": round(c_prob, 4), "rank": c_rank}

    for m_name, m_info in item.get("models", {}).items():
        m_grid = m_info.get("grid_100")
        if isinstance(m_grid, list) and len(m_grid) == 100:
            m_prob = float(m_grid[actual])
            m_rank = int(sum(1 for p in m_grid if p > m_prob) + 1)
            actual_lookup[m_name] = {"prob": round(m_prob, 4), "rank": m_rank}

    if actual_lookup:
        item["actual_prob_lookup"] = actual_lookup

    return item


def score_completed_sessions(df: pd.DataFrame, state: dict) -> dict:
    """
    Grades newly completed draws, removes undone entries, re-grades existing history
    against draws.csv, and rebuilds guarded [14% - 28%] committee weights.
    """
    completed_rows = df[df["Winning_Number"].notna()].copy()
    completed_map = {
        int(r["Session_ID"]): r
        for _, r in completed_rows.iterrows()
    }

    existing_hist = {
        int(item["session_id"]): item
        for item in state.get("history", [])
        if int(item["session_id"]) in completed_map
    }
    upcoming_list = state.get("upcoming", [])
    still_upcoming = []

    for pred in upcoming_list:
        sid = int(pred["session_id"])
        if sid in completed_map and sid not in existing_hist:
            existing_hist[sid] = grade_consensus_item(pred, completed_map[sid])
        elif sid not in completed_map:
            still_upcoming.append(pred)

    rebuilt_stats = {m: default_model_stat() for m in MODEL_NAMES}
    sorted_hist = sorted(existing_hist.values(), key=lambda x: x["session_id"], reverse=True)[:250]

    for item in sorted_hist:
        row = completed_map[int(item["session_id"])]
        grade_consensus_item(item, row)
        actual = int(row["Winning_Number"])

        for m_name, m_info in item.get("models", {}).items():
            st = rebuilt_stats.setdefault(m_name, default_model_stat())
            ev = evaluate_model_draw(m_info, actual)

            st["evaluated"] += 1
            st["total_points"] = round(st["total_points"] + ev["earned"], 2)
            if ev["hit_any_anchor"]:
                st["anchor_hits"] += 1
            if ev["hit_both_anchors"]:
                st["both_anchor_hits"] += 1
            st["exact_hits"] += ev["exact_cnt"]
            st["reverse_hits"] += ev["rev_cnt"]
            st["pos_pair_hits"] += ev["pos_cnt"]
            st["any_pair_hits"] += ev["any_cnt"]

            m_info["earned_pts"] = ev["earned"]
            m_info["score_detail"] = ev

    state["model_stats"] = rebuilt_stats

    raw_weights = {}
    for m in MODEL_NAMES:
        st = state["model_stats"].get(m, default_model_stat())
        avg_pts = (st["total_points"] + 8.0 * 0.55) / (st["evaluated"] + 8.0)
        raw_weights[m] = avg_pts ** 1.5

    state["model_weights"] = apply_weight_guardrails(raw_weights, floor=0.14, cap=0.28)
    state["history"] = sorted_hist
    state["upcoming"] = still_upcoming
    return state


def select_majority_protected_anchors(
    p_anchor_consensus: np.ndarray,
    model_top2_map: dict[str, list[int]],
    min_votes: int = 2,
) -> tuple[int, int, dict[int, int]]:
    """
    Selects Primary (d1) and Backup (d2) anchors using Majority-Vote Protection:
    digits supported by >= 2 models in their top-2 anchors take priority over
    single-model (1/5) spikes, ordered by blended anchor probability.
    """
    vote_counts = {d: 0 for d in range(10)}
    for digits in model_top2_map.values():
        for d in set(digits[:2]):
            vote_counts[int(d)] += 1

    ranked_all = [int(x) for x in np.argsort(p_anchor_consensus)[::-1]]
    majority_digits = [d for d in ranked_all if vote_counts[d] >= min_votes]
    solo_digits = [d for d in ranked_all if vote_counts[d] < min_votes]

    ordered = majority_digits + solo_digits
    return ordered[0], ordered[1], vote_counts


def compute_confidence_score(
    d1_votes: int,
    d2_votes: int,
    total_models: int,
    d1: int,
    d2: int,
    model_top2_map: dict[str, list[int]],
    model_stats: dict[str, dict],
    core_top3_prob_sum: float,
    step_idx: int,
) -> dict:
    """
    0-100 Confidence Score combining 4 signals:
      1. Committee Anchor Agreement (35 pts)
      2. Top-2 Leaderboard Leaders Alignment (25 pts)
      3. Core Top-3 Probability Lift vs. 3.0% Random Baseline (25 pts)
      4. Forecast Horizon: Live Lag vs. Chained Slot (15 pts)
    """
    best_votes = max(d1_votes, d2_votes)
    if best_votes >= 5:
        agree_pts = 35
    elif best_votes == 4:
        agree_pts = 28
    elif best_votes == 3:
        agree_pts = 20
    elif best_votes == 2 and min(d1_votes, d2_votes) >= 2:
        agree_pts = 14
    else:
        agree_pts = 6

    ranked_leaders = sorted(
        MODEL_NAMES,
        key=lambda m: model_stats.get(m, {}).get("total_points", 0.0),
        reverse=True,
    )[:2]
    consensus_set = {d1, d2}
    leader_hits = sum(
        1 for m in ranked_leaders
        if any(dig in consensus_set for dig in model_top2_map.get(m, []))
    )
    leader_pts = 25 if leader_hits == 2 else (13 if leader_hits == 1 else 0)

    # Random baseline for 3 pairs is 0.0300 (3.0%); lift of 5.5x (16.5%+) earns full 25 pts
    lift_ratio = core_top3_prob_sum / 0.03
    lift_pts = int(round(min(25.0, max(0.0, (lift_ratio - 1.5) * 6.25))))

    horizon_pts = 15 if step_idx == 0 else (4 if step_idx == 1 else 0)

    total_score = int(min(100, max(0, agree_pts + leader_pts + lift_pts + horizon_pts)))

    if total_score >= 75:
        tier = "HIGH"
        label = "HIGH CONVICTION"
    elif total_score >= 50:
        tier = "MODERATE"
        label = "MODERATE"
    else:
        tier = "LOW"
        label = "LOW / SPLIT"

    return {
        "score": total_score,
        "tier": tier,
        "label": label,
        "breakdown": {
            "agreement_pts": agree_pts,
            "leaders_pts": leader_pts,
            "lift_pts": lift_pts,
            "horizon_pts": horizon_pts,
            "top3_prob_sum": round(core_top3_prob_sum, 4),
            "leaders_aligned": ranked_leaders,
        },
    }


def main():
    df = pd.read_csv(DRAWS_PATH)
    now_mmt = datetime.now(YANGON_TZ)
    years_to_check = sorted(set(df["Year"].dropna().astype(int).tolist() + [now_mmt.year, now_mmt.year + 1]))
    th_map = get_thai_holiday_map(years_to_check)

    df = ensure_future_slots(df, th_map, min_open_slots=10)
    state = load_existing_predictions()
    state["thai_holidays"] = th_map
    state = score_completed_sessions(df, state)

    hist = df[df["Winning_Number"].notna()].copy()
    hist["Winning_Number"] = hist["Winning_Number"].astype(int)
    state["drought_radar"] = compute_drought_radar(hist)

    feat_df, tree_cols, hybrid_cols = build_features(df)
    completed_feats = feat_df[(feat_df["Session_ID"] >= 15) & (feat_df["Winning_Number"].notna())]
    train_df = completed_feats.tail(ROLLING_WINDOW)

    X_train_full = train_df[tree_cols].fillna(train_df[tree_cols].mean())
    X_train_hybrid = train_df[hybrid_cols].fillna(train_df[hybrid_cols].mean())
    y_train_num = train_df["Winning_Number"].astype(int).values

    ml_models = fit_ml_models(X_train_full, X_train_hybrid, y_train_num)

    # Autoregressive Sequential Chaining across the next 10 open trading slots
    sim_df = df.copy()
    open_sids = sim_df.loc[sim_df["Winning_Number"].isna(), "Session_ID"].astype(int).tolist()[:10]
    new_upcoming = []
    chained_pairs_so_far = []
    tabpfn_quota_exhausted = False

    for step_idx, sid in enumerate(open_sids):
        step_feats, _, _ = build_features(sim_df)
        row_mask = step_feats["Session_ID"] == sid
        target_row = sim_df.loc[row_mask].iloc[0]

        X_next_full = step_feats.loc[row_mask, tree_cols].fillna(X_train_full.mean())
        X_next_hybrid = step_feats.loc[row_mask, hybrid_cols].fillna(X_train_hybrid.mean())

        sim_hist_600 = sim_df[sim_df["Winning_Number"].notna()].tail(ROLLING_WINDOW).copy()
        model_grids = {
            "Bayesian_Markov": predict_bayesian_pure_transition(sim_hist_600, target_row)
        }

        for m_name, (clf_t, clf_u, clf_b) in ml_models.items():
            if m_name == "TabPFN_3.5" and tabpfn_quota_exhausted:
                continue
            X_in = X_next_hybrid if m_name == "TabPFN_3.5" else X_next_full
            try:
                pt = clf_t.predict_proba(X_in)[0]
                pu = clf_u.predict_proba(X_in)[0]
                pb = clf_b.predict_proba(X_in)[0]
                model_grids[m_name] = combine_t_u_brake(pt, pu, pb)
            except Exception as exc:
                print(f"[WARN] {m_name} skipped at step {step_idx + 1}: {exc}")
                if m_name == "TabPFN_3.5":
                    tabpfn_quota_exhausted = True

        step_weights = {m: state["model_weights"].get(m, 0.20) for m in model_grids}
        w_sum = sum(step_weights.values())
        step_weights = {m: w / w_sum for m, w in step_weights.items()}

        p_consensus = np.zeros(100)
        for m_name, grid in model_grids.items():
            p_consensus += step_weights[m_name] * grid
        p_consensus /= p_consensus.sum()

        model_breakdown = {}
        model_top2_map = {}
        for m_name, grid in model_grids.items():
            m_anc = grid_to_anchor_probs(grid)
            m_top_digits = [int(x) for x in np.argsort(m_anc)[::-1][:2]]
            model_top2_map[m_name] = m_top_digits
            top_indices = np.argsort(grid)[::-1][:3]
            m_top_pairs = [f"{int(idx):02d}" for idx in top_indices]
            m_top_pairs_prob = [
                {"pair": f"{int(idx):02d}", "prob": round(float(grid[idx]), 4)}
                for idx in top_indices
            ]
            model_breakdown[m_name] = {
                "anchor_digit": m_top_digits[0],
                "backup_digit": m_top_digits[1],
                "anchor_prob": round(float(m_anc[m_top_digits[0]]), 4),
                "top_pairs": m_top_pairs,
                "top_pairs_prob": m_top_pairs_prob,
                "grid_100": [round(float(x), 4) for x in grid],
            }

        p_anchor_consensus = grid_to_anchor_probs(p_consensus)
        d1, d2, vote_counts = select_majority_protected_anchors(
            p_anchor_consensus, model_top2_map, min_votes=2
        )
        d1_votes = vote_counts[d1]
        d2_votes = vote_counts[d2]

        anchor_filtered_pairs = []
        for idx in np.argsort(p_consensus)[::-1]:
            t, u = idx // 10, idx % 10
            if t in (d1, d2) or u in (d1, d2):
                supporters = [
                    m for m, info in model_breakdown.items()
                    if f"{idx:02d}" in info["top_pairs"] or info["anchor_digit"] in (t, u)
                ]
                rank_num = len(anchor_filtered_pairs) + 1
                anchor_filtered_pairs.append({
                    "pair": f"{idx:02d}",
                    "reverse": f"{u}{t}",
                    "prob": round(float(p_consensus[idx]), 4),
                    "tier": "CORE" if rank_num <= 3 else "EXTENDED",
                    "contains_primary": bool(t == d1 or u == d1),
                    "supporters": supporters,
                })
            if len(anchor_filtered_pairs) >= 6:
                break

        core_top3_prob_sum = sum(p["prob"] for p in anchor_filtered_pairs[:3])
        confidence = compute_confidence_score(
            d1_votes=d1_votes,
            d2_votes=d2_votes,
            total_models=len(model_grids),
            d1=d1,
            d2=d2,
            model_top2_map=model_top2_map,
            model_stats=state.get("model_stats", {}),
            core_top3_prob_sum=core_top3_prob_sum,
            step_idx=step_idx,
        )

        top1_pair_str = anchor_filtered_pairs[0]["pair"]
        sim_df.loc[row_mask, "Winning_Number"] = float(int(top1_pair_str))

        raw_date = str(target_row["Date"])
        new_upcoming.append({
            "session_id": sid,
            "date": raw_date,
            "short_date": format_short_date(raw_date),
            "day": format_day_short(raw_date),
            "day_full": format_day_name(raw_date),
            "session": "PM" if int(target_row["Is_PM"]) == 1 else "AM",
            "forecast_mode": "LIVE_LAG" if step_idx == 0 else "CHAINED",
            "chained_from": list(chained_pairs_so_far[-2:]),
            "confidence": confidence,
            "primary_anchor": {
                "digit": d1,
                "prob": round(float(p_anchor_consensus[d1]), 4),
                "votes": f"{d1_votes}/{len(model_grids)}",
            },
            "secondary_anchor": {
                "digit": d2,
                "prob": round(float(p_anchor_consensus[d2]), 4),
                "votes": f"{d2_votes}/{len(model_grids)}",
            },
            "intersection_pairs": [f"{d1}{d2}", f"{d2}{d1}"],
            "top_pairs": anchor_filtered_pairs,
            "consensus_grid_100": [round(float(x), 4) for x in p_consensus],
            "models": model_breakdown,
        })
        chained_pairs_so_far.append(top1_pair_str)

    state["upcoming"] = new_upcoming
    state["updated_at_iso"] = now_mmt.isoformat()
    state["updated_at"] = f"{now_mmt.day} {now_mmt.strftime('%b')} • {now_mmt.strftime('%I:%M %p').lstrip('0')} MMT"
    state.pop("latest_session_id", None)

    with open(PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print("[OK] SAI 2D predictions, confidence scores, grid_100 archive & Win Log synced.")


if __name__ == "__main__":
    main()
