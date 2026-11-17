import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

DRAWS_PATH = "data/draws.csv"
MODELS = ["LightGBM", "CatBoost", "L2_Logistic", "Bayesian_Markov"]

POWER_PAIRS = {5, 50, 16, 61, 27, 72, 38, 83, 49, 94}
NATKHAT_PAIRS = {7, 70, 18, 81, 24, 42, 35, 53, 69, 96}
BROTHER_PAIRS = {1, 10, 12, 21, 23, 32, 34, 43, 45, 54, 56, 65, 67, 76, 78, 87, 89, 98, 9, 90}


def get_family_id(num: int) -> int:
    if num // 10 == num % 10:
        return 1  # Double / Apuu
    if num in POWER_PAIRS:
        return 2  # Power
    if num in NATKHAT_PAIRS:
        return 3  # Natkhat
    if num in BROTHER_PAIRS:
        return 4  # Brother / Nyiko
    return 0      # Neutral


def build_feature_sets(data: pd.DataFrame):
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

    # Transition-type & intraday/overnight interaction features
    d["is_intraday_am_to_pm"] = (d["Is_PM"] == 1).astype(float)
    d["is_overnight_pm_to_am"] = ((d["Is_PM"] == 0) & (d["DayOfWeek"] != 0)).astype(float)
    d["is_monday_open"] = ((d["Is_PM"] == 0) & (d["DayOfWeek"] == 0)).astype(float)
    d["lag1_t_x_pm"] = d["lag_1_tens"] * d["Is_PM"]
    d["lag1_u_x_pm"] = d["lag_1_units"] * d["Is_PM"]

    tens_roll15, units_roll15 = np.zeros((n, 10)), np.zeros((n, 10))
    tens_roll50, units_roll50 = np.zeros((n, 10)), np.zeros((n, 10))
    brake_roll15, brake_roll50 = np.zeros((n, 10)), np.zeros((n, 10))
    fam_roll15 = np.zeros((n, 5))
    lag1_fam = np.zeros(n)
    high_tens_r15, odd_tens_r15 = np.zeros(n), np.zeros(n)
    high_units_r15, odd_units_r15 = np.zeros(n), np.zeros(n)
    recent_carry = np.zeros((n, 10))

    hist_t, hist_u, hist_b, hist_fam = [], [], [], []

    dow_t_rate = np.zeros((n, 10))
    dow_u_rate = np.zeros((n, 10))
    dow_any_rate = np.zeros((n, 10))
    cnt_t = np.zeros((7, 2, 10))
    cnt_u = np.zeros((7, 2, 10))
    cnt_any = np.zeros((7, 2, 10))
    cnt_slot = np.zeros((7, 2))

    for i in range(n):
        if hist_t:
            rt15, ru15, rb15 = hist_t[-15:], hist_u[-15:], hist_b[-15:]
            rt50, ru50, rb50 = hist_t[-50:], hist_u[-50:], hist_b[-50:]
            rf15 = hist_fam[-15:]
            last_digits = set(hist_t[-2:] + hist_u[-2:])
            for k in range(10):
                tens_roll15[i, k] = rt15.count(k) / len(rt15)
                units_roll15[i, k] = ru15.count(k) / len(ru15)
                tens_roll50[i, k] = rt50.count(k) / len(rt50)
                units_roll50[i, k] = ru50.count(k) / len(ru50)
                brake_roll15[i, k] = rb15.count(k) / len(rb15)
                brake_roll50[i, k] = rb50.count(k) / len(rb50)
                recent_carry[i, k] = 1.0 if k in last_digits else 0.0
            for f_id in range(5):
                fam_roll15[i, f_id] = rf15.count(f_id) / len(rf15)
            lag1_fam[i] = hist_fam[-1]
            high_tens_r15[i] = sum(1 for x in rt15 if x >= 5) / len(rt15)
            odd_tens_r15[i] = sum(1 for x in rt15 if x % 2 == 1) / len(rt15)
            high_units_r15[i] = sum(1 for x in ru15 if x >= 5) / len(ru15)
            odd_units_r15[i] = sum(1 for x in ru15 if x % 2 == 1) / len(ru15)

        w, pm = int(dows[i]), int(pms[i])
        denom = cnt_slot[w, pm] + 20.0
        dow_t_rate[i] = (cnt_t[w, pm] + 2.0) / denom
        dow_u_rate[i] = (cnt_u[w, pm] + 2.0) / denom
        dow_any_rate[i] = (cnt_any[w, pm] + 3.8) / denom

        if not np.isnan(nums[i]):
            val = int(nums[i])
            tv, uv = val // 10, val % 10
            bv = (tv + uv) % 10
            fv = get_family_id(val)
            hist_t.append(tv)
            hist_u.append(uv)
            hist_b.append(bv)
            hist_fam.append(fv)
            cnt_t[w, pm, tv] += 1.0
            cnt_u[w, pm, uv] += 1.0
            cnt_any[w, pm, tv] += 1.0
            if uv != tv:
                cnt_any[w, pm, uv] += 1.0
            cnt_slot[w, pm] += 1.0

    base_cols = ["Year", "Day_of_Year", "Is_PM", "DayOfWeek", "sin_doy", "cos_doy"]
    for lag in [1, 2, 3, 4, 5, 10]:
        base_cols += [f"lag_{lag}_num", f"lag_{lag}_tens", f"lag_{lag}_units", f"lag_{lag}_brake"]

    for k in range(10):
        d[f"tens_roll15_{k}"] = tens_roll15[:, k]
        d[f"units_roll15_{k}"] = units_roll15[:, k]
        d[f"tens_roll50_{k}"] = tens_roll50[:, k]
        d[f"units_roll50_{k}"] = units_roll50[:, k]
        d[f"recent_carry_{k}"] = recent_carry[:, k]
        d[f"dow_pm_tens_rate_{k}"] = dow_t_rate[:, k]
        d[f"dow_pm_units_rate_{k}"] = dow_u_rate[:, k]
        d[f"dow_pm_any_rate_{k}"] = dow_any_rate[:, k]
        base_cols += [
            f"tens_roll15_{k}", f"units_roll15_{k}", f"tens_roll50_{k}", f"units_roll50_{k}",
            f"recent_carry_{k}", f"dow_pm_tens_rate_{k}", f"dow_pm_units_rate_{k}", f"dow_pm_any_rate_{k}",
        ]

    d["lag1_fam"] = lag1_fam
    d["high_tens_r15"] = high_tens_r15
    d["odd_tens_r15"] = odd_tens_r15
    d["high_units_r15"] = high_units_r15
    d["odd_units_r15"] = odd_units_r15

    ext_cols = list(base_cols) + [
        "is_intraday_am_to_pm", "is_overnight_pm_to_am", "is_monday_open",
        "lag1_t_x_pm", "lag1_u_x_pm", "lag1_fam",
        "high_tens_r15", "odd_tens_r15", "high_units_r15", "odd_units_r15",
    ]

    for k in range(10):
        d[f"brake_roll15_{k}"] = brake_roll15[:, k]
        d[f"brake_roll50_{k}"] = brake_roll50[:, k]
        ext_cols += [f"brake_roll15_{k}", f"brake_roll50_{k}"]
    for f_id in range(5):
        d[f"fam_roll15_{f_id}"] = fam_roll15[:, f_id]
        ext_cols.append(f"fam_roll15_{f_id}")

    return d, base_cols, ext_cols


def predict_bayesian_markov(hist: pd.DataFrame, target_row: pd.Series) -> np.ndarray:
    n_hist = len(hist)
    nums = hist["Winning_Number"].astype(int).values
    tens = nums // 10
    units = nums % 10
    brakes = (tens + units) % 10

    weights = 0.985 ** np.arange(n_hist - 1, -1, -1)
    p_t = np.ones(10) * 1.5
    p_u = np.ones(10) * 1.5
    p_b = np.ones(10) * 1.5
    for idx in range(n_hist):
        p_t[tens[idx]] += weights[idx]
        p_u[units[idx]] += weights[idx]
        p_b[brakes[idx]] += weights[idx]
    p_t /= p_t.sum()
    p_u /= p_u.sum()
    p_b /= p_b.sum()

    d_dt = pd.to_datetime(hist["Date"])
    dow_hist = d_dt.dt.dayofweek.values
    is_pm_hist = hist["Is_PM"].values
    t_dow = pd.to_datetime(target_row["Date"]).dayofweek
    t_pm = int(target_row["Is_PM"])

    mask = (dow_hist == t_dow) & (is_pm_hist == t_pm)
    sub_nums = nums[mask]
    p_exact_dow = (np.bincount(sub_nums, minlength=100) + 0.35) / (len(sub_nums) + 35.0)

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

    grid = np.zeros(100)
    for num in range(100):
        t, u = num // 10, num % 10
        b = (t + u) % 10
        pos_score = p_t[t] * p_u[u] * (p_b[b] * 10.0)
        grid[num] = 0.40 * pos_score + 0.35 * p_exact_dow[num] + 0.25 * p_markov[num]

    grid = np.maximum(grid, 1e-6)
    return grid / grid.sum()


def fit_three_heads(X_train: pd.DataFrame, y_num: np.ndarray):
    y_t = y_num // 10
    y_u = y_num % 10
    y_b = (y_t + y_u) % 10
    fitted = {}

    for m_name in ["LightGBM", "CatBoost", "L2_Logistic"]:
        heads = []
        for target in (y_t, y_u, y_b):
            if m_name == "LightGBM":
                clf = lgb.LGBMClassifier(
                    objective="multiclass", num_class=10, n_estimators=120,
                    learning_rate=0.04, num_leaves=15, subsample_for_bin=500,
                    min_child_samples=25, random_state=42, verbose=-1
                )
            elif m_name == "CatBoost":
                clf = CatBoostClassifier(
                    iterations=250, learning_rate=0.04, depth=4,
                    loss_function="MultiClass", random_seed=42, verbose=0
                )
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.04, max_iter=1000))
            clf.fit(X_train, target)
            heads.append(clf)
        fitted[m_name] = tuple(heads)
    return fitted


def combine_t_u_brake(pt: np.ndarray, pu: np.ndarray, pb: np.ndarray) -> np.ndarray:
    grid = np.zeros(100)
    for num in range(100):
        t, u = num // 10, num % 10
        b = (t + u) % 10
        grid[num] = pt[t] * pu[u] * (pb[b] * 10.0)
    grid = np.maximum(grid, 1e-8)
    return grid / grid.sum()


def calibrate_sharpness(grid: np.ndarray, temp: float = 0.85) -> np.ndarray:
    g = np.maximum(grid, 1e-8)
    log_p = np.log(g)
    log_p = (log_p - log_p.mean()) / (log_p.std() + 1e-8)
    scaled = np.exp(log_p * temp)
    return scaled / scaled.sum()


def grid_to_anchor_probs(p_100: np.ndarray) -> np.ndarray:
    mat = p_100.reshape(10, 10)
    p_t = mat.sum(axis=1)
    p_u = mat.sum(axis=0)
    return p_t + p_u - (p_t * p_u)


def evaluate_grid(p_consensus: np.ndarray, actual: int):
    p_anc = grid_to_anchor_probs(p_consensus)
    ranked_d = np.argsort(p_anc)[::-1]
    d1, d2 = int(ranked_d[0]), int(ranked_d[1])

    top_pairs = []
    for idx in np.argsort(p_consensus)[::-1]:
        t, u = idx // 10, idx % 10
        if t in (d1, d2) or u in (d1, d2):
            top_pairs.append(int(idx))
        if len(top_pairs) >= 6:
            break

    act_digits = {actual // 10, actual % 10}
    rev_num = (actual % 10) * 10 + (actual // 10)

    hit_d1 = int(d1 in act_digits)
    hit_dual = int(d1 in act_digits or d2 in act_digits)
    hit_exact = int(actual in top_pairs)
    hit_exact_rev = int(actual in top_pairs or rev_num in top_pairs)
    nll = -np.log(max(float(p_consensus[actual]), 1e-6))
    return hit_d1, hit_dual, hit_exact, hit_exact_rev, nll


def main():
    t0 = time.time()
    df = pd.read_csv(DRAWS_PATH)
    df = df[df["Winning_Number"].notna()].copy().reset_index(drop=True)
    df["Winning_Number"] = df["Winning_Number"].astype(int)

    feat_df, base_cols, ext_cols = build_feature_sets(df)
    n_total = len(df)
    eval_window = min(300, n_total - 200)
    start_idx = n_total - eval_window
    step_size = 15  # Retrain every 15 draws across the 300-draw out-of-sample window

    configs = [
        "V1_Current_Baseline",
        "V2_Plus_Brake_Head",
        "V3_Plus_Transitions_Families",
        "V4_Plus_Calibration_MultiTier",
    ]
    metrics = {
        c: {"d1": 0, "dual": 0, "exact": 0, "exact_rev": 0, "nll": 0.0, "n": 0}
        for c in configs
    }

    # Multi-tier adaptive points for V4 (3=Exact, 2=Reverse, 1=Anchor)
    v4_points = {m: 0.0 for m in MODELS}
    v4_evals = {m: 0.0 for m in MODELS}

    for fold_start in range(start_idx, n_total, step_size):
        fold_end = min(fold_start + step_size, n_total)
        train_mask = (feat_df.index >= 14) & (feat_df.index < fold_start)
        y_train = df.loc[train_mask, "Winning_Number"].values

        models_base = fit_three_heads(feat_df.loc[train_mask, base_cols], y_train)
        models_ext = fit_three_heads(feat_df.loc[train_mask, ext_cols], y_train)

        for idx in range(fold_start, fold_end):
            actual = int(df.loc[idx, "Winning_Number"])
            hist_slice = df.iloc[:idx]
            target_row = df.iloc[idx]
            p_bayes = predict_bayesian_markov(hist_slice, target_row)

            X_base_next = feat_df.loc[[idx], base_cols]
            X_ext_next = feat_df.loc[[idx], ext_cols]

            # V1: Baseline (2 heads: Tens x Units, base features, equal blend)
            grids_v1 = {"Bayesian_Markov": p_bayes}
            for m_name, (ct, cu, _) in models_base.items():
                pt = ct.predict_proba(X_base_next)[0]
                pu = cu.predict_proba(X_base_next)[0]
                grids_v1[m_name] = np.outer(pt, pu).flatten()
            p_v1 = sum(grids_v1.values()) / len(grids_v1)

            # V2: + 3rd Brake Head (Tens x Units x Brake, base features, equal blend)
            grids_v2 = {"Bayesian_Markov": p_bayes}
            for m_name, (ct, cu, cb) in models_base.items():
                pt = ct.predict_proba(X_base_next)[0]
                pu = cu.predict_proba(X_base_next)[0]
                pb = cb.predict_proba(X_base_next)[0]
                grids_v2[m_name] = combine_t_u_brake(pt, pu, pb)
            p_v2 = sum(grids_v2.values()) / len(grids_v2)

            # V3: + Transition splits & 2D Number-Family features
            grids_v3 = {"Bayesian_Markov": p_bayes}
            for m_name, (ct, cu, cb) in models_ext.items():
                pt = ct.predict_proba(X_ext_next)[0]
                pu = cu.predict_proba(X_ext_next)[0]
                pb = cb.predict_proba(X_ext_next)[0]
                grids_v3[m_name] = combine_t_u_brake(pt, pu, pb)
            p_v3 = sum(grids_v3.values()) / len(grids_v3)

            # V4: + Temperature Sharpness Calibration & Multi-Tier Auto-Weighting
            raw_w = {}
            for m in MODELS:
                avg_pts = (v4_points[m] + 20.0 * 0.25) / (v4_evals[m] + 20.0)
                raw_w[m] = avg_pts ** 2
            w_sum = sum(raw_w.values())
            w_v4 = {m: raw_w[m] / w_sum for m in MODELS}

            p_v4 = np.zeros(100)
            for m in MODELS:
                cal_g = calibrate_sharpness(grids_v3[m], temp=0.85)
                p_v4 += w_v4[m] * cal_g
            p_v4 /= p_v4.sum()

            # Update V4 online multi-tier scores using actual result
            act_digits = {actual // 10, actual % 10}
            rev_num = (actual % 10) * 10 + (actual // 10)
            for m in MODELS:
                g = grids_v3[m]
                m_anc = int(np.argmax(grid_to_anchor_probs(g)))
                m_top3 = [int(x) for x in np.argsort(g)[::-1][:3]]
                pts = 0.0
                if actual in m_top3:
                    pts = 3.0
                elif rev_num != actual and rev_num in m_top3:
                    pts = 2.0
                elif m_anc in act_digits:
                    pts = 1.0
                v4_points[m] += pts
                v4_evals[m] += 1.0

            for cfg_name, p_grid in [
                ("V1_Current_Baseline", p_v1),
                ("V2_Plus_Brake_Head", p_v2),
                ("V3_Plus_Transitions_Families", p_v3),
                ("V4_Plus_Calibration_MultiTier", p_v4),
            ]:
                d1, dual, ex, ex_r, nll = evaluate_grid(p_grid, actual)
                metrics[cfg_name]["d1"] += d1
                metrics[cfg_name]["dual"] += dual
                metrics[cfg_name]["exact"] += ex
                metrics[cfg_name]["exact_rev"] += ex_r
                metrics[cfg_name]["nll"] += nll
                metrics[cfg_name]["n"] += 1

    elapsed = time.time() - t0
    lines = [
        f"### SAI 2D Walk-Forward Backtest ({eval_window} Out-of-Sample Draws • Completed in {elapsed:.1f}s)",
        "",
        "| Configuration | Primary Anchor (19.0% base) | Dual Anchor (34.4% base) | Top-6 Exact (6.0% base) | Top-6 Exact + Rev (11.4% base) | Log-Loss (Lower = Better) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in configs:
        m = metrics[c]
        n = m["n"]
        lines.append(
            f"| **{c}** | {m['d1']}/{n} (**{m['d1']/n*100:.1f}%**) | "
            f"{m['dual']}/{n} (**{m['dual']/n*100:.1f}%**) | "
            f"{m['exact']}/{n} (**{m['exact']/n*100:.1f}%**) | "
            f"{m['exact_rev']}/{n} (**{m['exact_rev']/n*100:.1f}%**) | "
            f"{m['nll']/n:.4f} |"
        )

    report = "\n".join(lines)
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
