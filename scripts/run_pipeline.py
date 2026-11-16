import json
import os
from datetime import datetime, timezone
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


def load_existing_predictions() -> dict:
    if os.path.exists(PRED_PATH):
        with open(PRED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "app_name": "SAI 2D",
        "updated_at": None,
        "latest_session_id": 0,
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
    """Automatically appends future weekday AM/PM slots if open rows drop below min_open_slots."""
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
            while cur_date.weekday() >= 5:  # Skip Sat/Sun
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
    d = data.copy()
    d["Date_dt"] = pd.to_datetime(d["Date"])
    d["DayOfWeek"] = d["Date_dt"].dt.dayofweek
    nums = d["Winning_Number"].values
    n = len(d)

    d["sin_doy"] = np.sin(2 * np.pi * d["Day_of_Year"] / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * d["Day_of_Year"] / 365.25)

    for lag in [1, 2, 3, 4, 5, 10]:
        s = pd.Series(nums).shift(lag)
        d[f"lag_{lag}_num"] = s
        d[f"lag_{lag}_tens"] = np.floor(s / 10.0)
        d[f"lag_{lag}_units"] = s % 10
        d[f"lag_{lag}_brake"] = (d[f"lag_{lag}_tens"] + d[f"lag_{lag}_units"]) % 10

    tens_gap, units_gap = np.zeros((n, 10)), np.zeros((n, 10))
    tens_roll50, units_roll50 = np.zeros((n, 10)), np.zeros((n, 10))
    cur_t_gap, cur_u_gap = np.zeros(10), np.zeros(10)
    hist_t, hist_u = [], []

    for i in range(n):
        tens_gap[i] = cur_t_gap
        units_gap[i] = cur_u_gap
        if hist_t:
            rt, ru = hist_t[-50:], hist_u[-50:]
            for k in range(10):
                tens_roll50[i, k] = rt.count(k) / len(rt)
                units_roll50[i, k] = ru.count(k) / len(ru)
        if not np.isnan(nums[i]):
            val = int(nums[i])
            t_val, u_val = val // 10, val % 10
            cur_t_gap += 1
            cur_u_gap += 1
            cur_t_gap[t_val] = 0
            cur_u_gap[u_val] = 0
            hist_t.append(t_val)
            hist_u.append(u_val)

    for k in range(10):
        d[f"tens_gap_{k}"] = tens_gap[:, k]
        d[f"units_gap_{k}"] = units_gap[:, k]
        d[f"tens_roll50_{k}"] = tens_roll50[:, k]
        d[f"units_roll50_{k}"] = units_roll50[:, k]

    return d


def compute_drought_radar(hist: pd.DataFrame) -> dict:
    last_sid = int(hist["Session_ID"].max())
    tens_s = hist["Winning_Number"].astype(int) // 10
    units_s = hist["Winning_Number"].astype(int) % 10
    brake_s = (tens_s + units_s) % 10

    tens_gaps = {
        str(d): int(last_sid - hist.loc[tens_s == d, "Session_ID"].max())
        for d in range(10)
    }
    units_gaps = {
        str(d): int(last_sid - hist.loc[units_s == d, "Session_ID"].max())
        for d in range(10)
    }
    brake_gaps = {
        str(d): int(last_sid - hist.loc[brake_s == d, "Session_ID"].max())
        for d in range(10)
    }
    any_pos_gaps = {
        str(d): int(last_sid - hist.loc[(tens_s == d) | (units_s == d), "Session_ID"].max())
        for d in range(10)
    }
    return {
        "as_of_session": last_sid,
        "tens_gaps": tens_gaps,
        "units_gaps": units_gaps,
        "brake_gaps": brake_gaps,
        "any_pos_gaps": any_pos_gaps,
    }


def predict_bayesian_markov(hist: pd.DataFrame, target_row: pd.Series) -> np.ndarray:
    """Conjugate Dirichlet-Markov + Overdue + Brake + Session Polarity model."""
    n_hist = len(hist)
    nums = hist["Winning_Number"].astype(int).values
    tens = nums // 10
    units = nums % 10
    brakes = (tens + units) % 10

    # Exponentially decayed positional priors (gamma = 0.994)
    weights = 0.994 ** np.arange(n_hist - 1, -1, -1)
    p_t = np.ones(10) * 2.0
    p_u = np.ones(10) * 2.0
    p_b = np.ones(10) * 2.0
    for idx in range(n_hist):
        p_t[tens[idx]] += weights[idx]
        p_u[units[idx]] += weights[idx]
        p_b[brakes[idx]] += weights[idx]
    p_t /= p_t.sum()
    p_u /= p_u.sum()
    p_b /= p_b.sum()

    # Session & Day-of-Week affinity
    d_dt = pd.to_datetime(hist["Date"])
    dow_hist = d_dt.dt.dayofweek.values
    is_pm_hist = hist["Is_PM"].values
    t_dow = pd.to_datetime(target_row["Date"]).dayofweek
    t_pm = int(target_row["Is_PM"])

    mask = (dow_hist == t_dow) & (is_pm_hist == t_pm)
    sub_nums = nums[mask]
    p_exact_dow = (np.bincount(sub_nums, minlength=100) + 0.35) / (len(sub_nums) + 35.0)

    # First-order Markov transition from immediately prior session
    last_num = int(nums[-1])
    last_t, last_u = last_num // 10, last_num % 10
    t_mat = np.ones((10, 10)) * 2.0
    u_mat = np.ones((10, 10)) * 2.0
    for i in range(n_hist - 1):
        t_mat[tens[i], tens[i + 1]] += 1.0
        u_mat[units[i], units[i + 1]] += 1.0
    t_mat /= t_mat.sum(axis=1, keepdims=True)
    u_mat /= u_mat.sum(axis=1, keepdims=True)
    p_markov = np.outer(t_mat[last_t], u_mat[last_u]).flatten()

    # Overdue pressure multiplier
    last_sid = int(hist["Session_ID"].max())
    gap_t = np.array([last_sid - hist.loc[tens == k, "Session_ID"].max() for k in range(10)])
    gap_u = np.array([last_sid - hist.loc[units == k, "Session_ID"].max() for k in range(10)])
    gap_b = np.array([last_sid - hist.loc[brakes == k, "Session_ID"].max() for k in range(10)])

    w_t = 1.0 + 0.35 * np.clip((gap_t - 8) / 20.0, 0, 1.5)
    w_u = 1.0 + 0.35 * np.clip((gap_u - 8) / 20.0, 0, 1.5)
    w_b = 1.0 + 0.25 * np.clip((gap_b - 8) / 20.0, 0, 1.5)

    grid = np.zeros(100)
    for num in range(100):
        t, u = num // 10, num % 10
        b = (t + u) % 10
        pos_score = (p_t[t] * w_t[t]) * (p_u[u] * w_u[u]) * (p_b[b] * w_b[b] * 10.0)
        grid[num] = 0.40 * pos_score + 0.30 * p_markov[num] + 0.30 * p_exact_dow[num]

    grid = np.maximum(grid, 1e-6)
    return grid / grid.sum()


def fit_ml_models(X_train: pd.DataFrame, y_num: np.ndarray):
    y_t = y_num // 10
    y_u = y_num % 10
    models = {}

    # 1. LightGBM
    lgb_t = lgb.LGBMClassifier(
        objective="multiclass", num_class=10, n_estimators=120,
        learning_rate=0.04, num_leaves=15, subsample_for_bin=500,
        min_child_samples=25, random_state=42, verbose=-1
    )
    lgb_u = lgb.LGBMClassifier(
        objective="multiclass", num_class=10, n_estimators=120,
        learning_rate=0.04, num_leaves=15, subsample_for_bin=500,
        min_child_samples=25, random_state=42, verbose=-1
    )
    lgb_t.fit(X_train, y_t)
    lgb_u.fit(X_train, y_u)
    models["LightGBM"] = (lgb_t, lgb_u)

    # 2. CatBoost
    cb_t = CatBoostClassifier(
        iterations=250, learning_rate=0.04, depth=4,
        loss_function="MultiClass", random_seed=42, verbose=0
    )
    cb_u = CatBoostClassifier(
        iterations=250, learning_rate=0.04, depth=4,
        loss_function="MultiClass", random_seed=42, verbose=0
    )
    cb_t.fit(X_train, y_t)
    cb_u.fit(X_train, y_u)
    models["CatBoost"] = (cb_t, cb_u)

    # 3. L2-Logistic Regression
    lr_t = make_pipeline(StandardScaler(), LogisticRegression(C=0.04, max_iter=1000))
    lr_u = make_pipeline(StandardScaler(), LogisticRegression(C=0.04, max_iter=1000))
    lr_t.fit(X_train, y_t)
    lr_u.fit(X_train, y_u)
    models["L2_Logistic"] = (lr_t, lr_u)

    # 4. TabPFN-3.5 (optional cloud API via TABPFN_TOKEN)
    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if token:
        try:
            from tabpfn_client import TabPFNClassifier, set_access_token
            from tabpfn_client.api_models import ModelVersion
            set_access_token(token)
            tp_t = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)
            tp_u = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)
            tp_t.fit(X_train, y_t)
            tp_u.fit(X_train, y_u)
            models["TabPFN_3.5"] = (tp_t, tp_u)
        except Exception as exc:
            print(f"[WARN] TabPFN-3.5 skipped due to API error: {exc}")

    return models


def grid_to_anchor_probs(p_100: np.ndarray) -> np.ndarray:
    """Computes P(Digit d appears in Tens OR Units) for d=0..9."""
    mat = p_100.reshape(10, 10)
    p_t = mat.sum(axis=1)
    p_u = mat.sum(axis=0)
    return p_t + p_u - (p_t * p_u)


def score_completed_sessions(df: pd.DataFrame, state: dict) -> dict:
    """Grades previously locked predictions against newly logged actual winning numbers."""
    completed_map = {
        int(r["Session_ID"]): int(r["Winning_Number"])
        for _, r in df[df["Winning_Number"].notna()].iterrows()
    }

    existing_hist = {int(item["session_id"]): item for item in state.get("history", [])}
    upcoming_list = state.get("upcoming", [])
    still_upcoming = []

    for pred in upcoming_list:
        sid = int(pred["session_id"])
        if sid in completed_map and sid not in existing_hist:
            actual = completed_map[sid]
            act_str = f"{actual:02d}"
            rev_str = f"{actual % 10}{actual // 10}"
            act_digits = {actual // 10, actual % 10}

            top_pairs = [p["pair"] for p in pred.get("top_pairs", [])]
            anchor1 = int(pred["primary_anchor"]["digit"])
            anchor2 = int(pred["secondary_anchor"]["digit"])

            if act_str in top_pairs[:5]:
                outcome = "EXACT"
            elif rev_str in top_pairs[:5] and act_str != rev_str:
                outcome = "REVERSE"
            elif anchor1 in act_digits or anchor2 in act_digits:
                outcome = "ANCHOR_HIT"
            else:
                outcome = "MISS"

            # Update per-model rolling stats
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

            pred["actual"] = act_str
            pred["outcome"] = outcome
            pred["hit_primary_anchor"] = bool(anchor1 in act_digits)
            pred["hit_secondary_anchor"] = bool(anchor2 in act_digits)
            existing_hist[sid] = pred
        elif sid not in completed_map:
            still_upcoming.append(pred)

    # Recompute adaptive model weights from anchor hit rate (smoothed around 19% baseline)
    raw_weights = {}
    for m in MODEL_NAMES:
        st = state["model_stats"].get(m, {"evaluated": 0, "anchor_hits": 0})
        evals = st["evaluated"]
        hits = st["anchor_hits"]
        # Bayesian smoothed hit rate with 20 pseudo-observations at 0.19 baseline
        smoothed_rate = (hits + 20 * 0.19) / (evals + 20.0)
        raw_weights[m] = smoothed_rate ** 2
    total_w = sum(raw_weights.values())
    state["model_weights"] = {m: round(raw_weights[m] / total_w, 4) for m in MODEL_NAMES}
    state["history"] = sorted(existing_hist.values(), key=lambda x: x["session_id"], reverse=True)[:120]
    state["upcoming"] = still_upcoming
    return state


def main():
    df = pd.read_csv(DRAWS_PATH)
    df = ensure_future_slots(df, min_open_slots=8)
    state = load_existing_predictions()
    state = score_completed_sessions(df, state)

    hist = df[df["Winning_Number"].notna()].copy()
    hist["Winning_Number"] = hist["Winning_Number"].astype(int)
    state["latest_session_id"] = int(hist["Session_ID"].max())
    state["drought_radar"] = compute_drought_radar(hist)

    feat_df = build_features(df)
    ignore_cols = {"Session_ID", "Date", "Date_dt", "Winning_Number"}
    feature_cols = [c for c in feat_df.columns if c not in ignore_cols]

    train_df = feat_df[(feat_df["Session_ID"] >= 15) & (feat_df["Winning_Number"].notna())]
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

    # Preserve locked predictions if user manually locked an upcoming slot, otherwise refresh
    locked_map = {
        int(p["session_id"]): p
        for p in state.get("upcoming", [])
        if p.get("locked", False)
    }

    sim_df = df.copy()
    open_sids = sim_df.loc[sim_df["Winning_Number"].isna(), "Session_ID"].astype(int).tolist()[:8]
    new_upcoming = []

    for sid in open_sids:
        if sid in locked_map:
            new_upcoming.append(locked_map[sid])
            top_pair_val = int(locked_map[sid]["top_pairs"][0]["pair"])
            sim_df.loc[sim_df["Session_ID"] == sid, "Winning_Number"] = float(top_pair_val)
            continue

        step_feats = build_features(sim_df)
        row_mask = step_feats["Session_ID"] == sid
        X_next = step_feats.loc[row_mask, feature_cols]
        target_row = sim_df.loc[row_mask].iloc[0]

        sim_hist = sim_df[sim_df["Winning_Number"].notna()].copy()
        model_grids = {}
        model_grids["Bayesian_Markov"] = predict_bayesian_markov(sim_hist, target_row)

        for m_name, (clf_t, clf_u) in ml_models.items():
            pt = clf_t.predict_proba(X_next)[0]
            pu = clf_u.predict_proba(X_next)[0]
            model_grids[m_name] = np.outer(pt, pu).flatten()

        # Weighted consensus 100-cell probability grid
        p_consensus = np.zeros(100)
        for m_name, grid in model_grids.items():
            p_consensus += active_weights[m_name] * grid
        p_consensus /= p_consensus.sum()

        # Digit-First Funnel: extract Consensus Anchor Digits & model agreement
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

        # Filter 00-99 pairs strictly through Primary & Secondary Anchor Digits
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

        # 19-number wheel for Primary Anchor Digit d1
        wheel_19 = sorted({
            f"{d1}{x}" for x in range(10)
        } | {
            f"{x}{d1}" for x in range(10)
        })

        # Propagate top consensus pair for multi-step lags
        sim_df.loc[row_mask, "Winning_Number"] = float(int(anchor_filtered_pairs[0]["pair"]))

        day_name = pd.to_datetime(target_row["Date"]).strftime("%a")
        new_upcoming.append({
            "session_id": sid,
            "date": str(target_row["Date"]),
            "day": day_name,
            "session": "PM" if int(target_row["Is_PM"]) == 1 else "AM",
            "locked": False,
            "primary_anchor": {
                "digit": d1,
                "prob": round(float(p_anchor_consensus[d1]), 4),
                "votes": f"{d1_votes}/{len(model_grids)}",
                "wheel_19": wheel_19,
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

    state["upcoming"] = new_upcoming
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with open(PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"[OK] SAI 2D predictions updated through Session {open_sids[-1]}.")


if __name__ == "__main__":
    main()
