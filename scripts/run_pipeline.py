import json
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
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


def format_short_date(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    return f"{dt.day} {dt.strftime('%b')}"


def format_day_name(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%A")


def format_day_short(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime("%a")


def load_existing_predictions() -> dict:
    if os.path.exists(PRED_PATH):
        with open(PRED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "app_name": "SAI 2D",
        "updated_at": None,
        "updated_at_iso": None,
        "model_weights": {m: 0.20 for m in MODEL_NAMES},
        "model_stats": {
            m: {"evaluated": 0, "anchor_hits": 0, "exact_hits": 0, "reverse_hits": 0}
            for m in MODEL_NAMES
        },
        "drought_radar": {},
        "upcoming": [],
        "history": [],
    }


def ensure_future_slots(df: pd.DataFrame, min_open_slots: int = 8) -> pd.DataFrame:
    open_count = int(df["Winning_Number"].isna().sum())
    if open_count >= min_open_slots:
        return df

    rows = df.to_dict("records")
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
            while cur_date.weekday() >= 5:
                cur_date += pd.Timedelta(days=1)

        rows.append({
            "Session_ID": cur_sid,
            "Year": int(cur_date.year),
            "Day_of_Year": int(cur_date.dayofyear),
            "Date": f"{cur_date.month}/{cur_date.day}/{cur_date.year}",
            "Is_PM": cur_pm,
            "Winning_Number": np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(DRAWS_PATH, index=False)
    return out


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Leak-free momentum + Day-of-Week x AM/PM positional rate features."""
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

    return d


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


def predict_bayesian_pure_transition(
    hist: pd.DataFrame, target_row: pd.Series, is_immediate_next: bool = True
) -> np.ndarray:
    """Diverse Model 1: Pure Day-of-Week x Session + Intraday/Overnight Markov Transition."""
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

    if not is_immediate_next:
        weights = 0.985 ** np.arange(len(nums) - 1, -1, -1)
        p_t = (np.bincount(tens, weights=weights, minlength=10) + 1.5)
        p_u = (np.bincount(units, weights=weights, minlength=10) + 1.5)
        p_t /= p_t.sum()
        p_u /= p_u.sum()
        p_mom = np.outer(p_t, p_u).flatten()
        grid = 0.55 * p_exact_dow + 0.45 * p_mom
        grid = np.maximum(grid, 1e-6)
        return grid / grid.sum()

    last_num = int(nums[-1])
    last_t, last_u = last_num // 10, last_num % 10
    t_mat = np.ones((10, 10)) * 2.0
    u_mat = np.ones((10, 10)) * 2.0
    trans_mask = (is_pm_hist[1:] == t_pm)
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


def fit_ml_models(X_train: pd.DataFrame, y_num: np.ndarray):
    """Fits 3 heads (Tens, Units, Modulo-10 Brake) on the active 600-draw regime window."""
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
        clf.fit(X_train, target)
        lgb_heads.append(clf)
    models["LightGBM"] = tuple(lgb_heads)

    cb_heads = []
    for target in (y_t, y_u, y_b):
        clf = CatBoostClassifier(
            iterations=150, learning_rate=0.05, depth=4,
            loss_function="MultiClass", thread_count=-1, random_seed=42, verbose=0
        )
        clf.fit(X_train, target)
        cb_heads.append(clf)
    models["CatBoost"] = tuple(cb_heads)

    lr_heads = []
    for target in (y_t, y_u, y_b):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=600))
        clf.fit(X_train, target)
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
                clf.fit(X_train, target)
                tp_heads.append(clf)
            models["TabPFN_3.5"] = tuple(tp_heads)
        except Exception as exc:
            print(f"[WARN] TabPFN-3.5 skipped due to API error: {exc}")

    return models


def grid_to_anchor_probs(p_100: np.ndarray) -> np.ndarray:
    mat = p_100.reshape(10, 10)
    p_t = mat.sum(axis=1)
    p_u = mat.sum(axis=0)
    return p_t + p_u - (p_t * p_u)


def score_completed_sessions(df: pd.DataFrame, state: dict) -> dict:
    """Grades completed draws and updates multi-tier composite weights (3=Exact, 2=Reverse, 1=Anchor)."""
    completed_rows = df[df["Winning_Number"].notna()].copy()
    completed_map = {
        int(r["Session_ID"]): r
        for _, r in completed_rows.iterrows()
    }

    existing_hist = {int(item["session_id"]): item for item in state.get("history", [])}
    upcoming_list = state.get("upcoming", [])
    still_upcoming = []

    for pred in upcoming_list:
        sid = int(pred["session_id"])
        if sid in completed_map and sid not in existing_hist:
            row = completed_map[sid]
            actual = int(row["Winning_Number"])
            act_str = f"{actual:02d}"
            rev_str = f"{actual % 10}{actual // 10}"
            act_digits = {actual // 10, actual % 10}

            top_pairs = [p["pair"] for p in pred.get("top_pairs", [])]
            anchor1 = int(pred["primary_anchor"]["digit"])
            anchor2 = int(pred["secondary_anchor"]["digit"])

            if act_str in top_pairs[:6]:
                outcome = "EXACT"
            elif rev_str in top_pairs[:6] and act_str != rev_str:
                outcome = "REVERSE"
            elif anchor1 in act_digits or anchor2 in act_digits:
                outcome = "ANCHOR_HIT"
            else:
                outcome = "MISS"

            for m_name, m_info in pred.get("models", {}).items():
                if m_name not in state["model_stats"]:
                    state["model_stats"][m_name] = {
                        "evaluated": 0, "anchor_hits": 0, "exact_hits": 0, "reverse_hits": 0
                    }
                st = state["model_stats"][m_name]
                st["evaluated"] += 1
                m_anchor = int(m_info["anchor_digit"])
                m_pairs = m_info.get("top_pairs", [])
                if m_anchor in act_digits:
                    st["anchor_hits"] += 1
                if act_str in m_pairs[:3]:
                    st["exact_hits"] += 1
                elif rev_str in m_pairs[:3] and act_str != rev_str:
                    st["reverse_hits"] += 1

            pred["short_date"] = format_short_date(str(row["Date"]))
            pred["day_full"] = format_day_name(str(row["Date"]))
            pred["day"] = format_day_short(str(row["Date"]))
            pred["actual"] = act_str
            pred["outcome"] = outcome
            pred["hit_primary_anchor"] = bool(anchor1 in act_digits)
            pred["hit_secondary_anchor"] = bool(anchor2 in act_digits)
            existing_hist[sid] = pred
        elif sid not in completed_map:
            still_upcoming.append(pred)

    raw_weights = {}
    for m in MODEL_NAMES:
        st = state["model_stats"].get(
            m, {"evaluated": 0, "anchor_hits": 0, "exact_hits": 0, "reverse_hits": 0}
        )
        composite_pts = (
            3.0 * st.get("exact_hits", 0)
            + 2.0 * st.get("reverse_hits", 0)
            + 1.0 * st.get("anchor_hits", 0)
        )
        smoothed_pts = (composite_pts + 20.0 * 0.25) / (st.get("evaluated", 0) + 20.0)
        raw_weights[m] = smoothed_pts ** 2

    total_w = sum(raw_weights.values())
    state["model_weights"] = {m: round(raw_weights[m] / total_w, 4) for m in MODEL_NAMES}
    state["history"] = sorted(existing_hist.values(), key=lambda x: x["session_id"], reverse=True)[:150]
    state["upcoming"] = still_upcoming
    return state


def main():
    df = pd.read_csv(DRAWS_PATH)
    df = ensure_future_slots(df, min_open_slots=8)
    state = load_existing_predictions()
    state = score_completed_sessions(df, state)

    hist = df[df["Winning_Number"].notna()].copy()
    hist["Winning_Number"] = hist["Winning_Number"].astype(int)
    state["drought_radar"] = compute_drought_radar(hist)

    feat_df = build_features(df)
    ignore_cols = {"Session_ID", "Date", "Date_dt", "Winning_Number"}
    feature_cols = [c for c in feat_df.columns if c not in ignore_cols]

    # Train exclusively on the most recent 600 completed draws (validated winner C2)
    completed_feats = feat_df[(feat_df["Session_ID"] >= 15) & (feat_df["Winning_Number"].notna())]
    train_df = completed_feats.tail(ROLLING_WINDOW)
    X_train = train_df[feature_cols]
    y_train_num = train_df["Winning_Number"].astype(int).values

    ml_models = fit_ml_models(X_train, y_train_num)
    active_weights = {
        m: state["model_weights"].get(m, 0.20)
        for m in MODEL_NAMES
        if m in ml_models or m == "Bayesian_Markov"
    }
    w_sum = sum(active_weights.values())
    active_weights = {m: w / w_sum for m, w in active_weights.items()}

    open_sids = df.loc[df["Winning_Number"].isna(), "Session_ID"].astype(int).tolist()[:8]
    recent_hist_600 = hist.tail(ROLLING_WINDOW).copy()
    lag_cols = [c for c in feature_cols if c.startswith("lag_")]
    neutral_lag_means = X_train[lag_cols].mean()

    # Confirmed feature state right after the latest real draw
    first_open_mask = feat_df["Session_ID"] == open_sids[0]
    confirmed_state_row = feat_df.loc[first_open_mask, feature_cols].iloc[0].copy()

    new_upcoming = []
    for step_idx, sid in enumerate(open_sids):
        is_immediate = (step_idx == 0)
        row_mask = feat_df["Session_ID"] == sid
        target_row = df.loc[row_mask].iloc[0]

        if is_immediate:
            X_next = feat_df.loc[row_mask, feature_cols].fillna(X_train.mean())
        else:
            # For Slots 2-8, use known calendar + current confirmed momentum with neutral lag means
            X_next = feat_df.loc[row_mask, feature_cols].copy()
            for col in feature_cols:
                if col not in {"Year", "Day_of_Year", "Is_PM", "DayOfWeek", "sin_doy", "cos_doy"}:
                    X_next[col] = confirmed_state_row[col]
            for lcol in lag_cols:
                X_next[lcol] = neutral_lag_means[lcol]

        model_grids = {
            "Bayesian_Markov": predict_bayesian_pure_transition(
                recent_hist_600, target_row, is_immediate_next=is_immediate
            )
        }

        for m_name, (clf_t, clf_u, clf_b) in ml_models.items():
            pt = clf_t.predict_proba(X_next)[0]
            pu = clf_u.predict_proba(X_next)[0]
            pb = clf_b.predict_proba(X_next)[0]
            model_grids[m_name] = combine_t_u_brake(pt, pu, pb)

        p_consensus = np.zeros(100)
        for m_name, grid in model_grids.items():
            p_consensus += active_weights[m_name] * grid
        p_consensus /= p_consensus.sum()

        p_anchor_consensus = grid_to_anchor_probs(p_consensus)
        ranked_digits = np.argsort(p_anchor_consensus)[::-1]
        d1, d2 = int(ranked_digits[0]), int(ranked_digits[1])

        model_breakdown = {}
        d1_votes, d2_votes = 0, 0
        for m_name, grid in model_grids.items():
            m_anc = grid_to_anchor_probs(grid)
            m_top_digits = [int(x) for x in np.argsort(m_anc)[::-1][:2]]
            if d1 in m_top_digits:
                d1_votes += 1
            if d2 in m_top_digits:
                d2_votes += 1
            m_top_pairs = [f"{idx:02d}" for idx in np.argsort(grid)[::-1][:3]]
            model_breakdown[m_name] = {
                "anchor_digit": m_top_digits[0],
                "backup_digit": m_top_digits[1],
                "anchor_prob": round(float(m_anc[m_top_digits[0]]), 4),
                "top_pairs": m_top_pairs,
            }

        anchor_filtered_pairs = []
        for idx in np.argsort(p_consensus)[::-1]:
            t, u = idx // 10, idx % 10
            if t in (d1, d2) or u in (d1, d2):
                supporters = [
                    m for m, info in model_breakdown.items()
                    if f"{idx:02d}" in info["top_pairs"] or info["anchor_digit"] in (t, u)
                ]
                anchor_filtered_pairs.append({
                    "pair": f"{idx:02d}",
                    "reverse": f"{u}{t}",
                    "prob": round(float(p_consensus[idx]), 4),
                    "contains_primary": bool(t == d1 or u == d1),
                    "supporters": supporters,
                })
            if len(anchor_filtered_pairs) >= 6:
                break

        raw_date = str(target_row["Date"])
        new_upcoming.append({
            "session_id": sid,
            "date": raw_date,
            "short_date": format_short_date(raw_date),
            "day": format_day_short(raw_date),
            "day_full": format_day_name(raw_date),
            "session": "PM" if int(target_row["Is_PM"]) == 1 else "AM",
            "forecast_mode": "LIVE_LAG" if is_immediate else "BASE_CALENDAR",
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
            "models": model_breakdown,
        })

    now_mmt = datetime.now(YANGON_TZ)
    state["upcoming"] = new_upcoming
    state["updated_at_iso"] = now_mmt.isoformat()
    state["updated_at"] = f"{now_mmt.day} {now_mmt.strftime('%b')} • {now_mmt.strftime('%I:%M %p').lstrip('0')} MMT"
    state.pop("latest_session_id", None)

    with open(PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print("[OK] SAI 2D Rolling-600 3-Head Pure-Transition predictions updated.")


if __name__ == "__main__":
    main()
