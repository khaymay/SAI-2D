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
BRAKE_IDX = (np.arange(10)[:, None] + np.arange(10)[None, :]) % 10


def build_features(data: pd.DataFrame):
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

    # Intraday (AM -> PM) vs Overnight / Weekend (PM -> AM) transition signals
    d["is_intraday_am_to_pm"] = (d["Is_PM"] == 1).astype(float)
    d["is_overnight_pm_to_am"] = ((d["Is_PM"] == 0) & (d["DayOfWeek"] != 0)).astype(float)
    d["is_monday_open"] = ((d["Is_PM"] == 0) & (d["DayOfWeek"] == 0)).astype(float)
    d["lag1_t_x_pm"] = d["lag_1_tens"] * d["Is_PM"]
    d["lag1_u_x_pm"] = d["lag_1_units"] * d["Is_PM"]
    d["lag1_b_x_pm"] = d["lag_1_brake"] * d["Is_PM"]
    d["lag1_is_double"] = (d["lag_1_tens"] == d["lag_1_units"]).astype(float)

    tens_roll15, units_roll15 = np.zeros((n, 10)), np.zeros((n, 10))
    tens_roll50, units_roll50 = np.zeros((n, 10)), np.zeros((n, 10))
    recent_carry = np.zeros((n, 10))
    lag1_digit_flag = np.zeros((n, 10))
    lag2_digit_flag = np.zeros((n, 10))
    hist_t, hist_u = [], []

    dow_t_rate = np.zeros((n, 10))
    dow_u_rate = np.zeros((n, 10))
    dow_any_rate = np.zeros((n, 10))
    cnt_t = np.zeros((7, 2, 10))
    cnt_u = np.zeros((7, 2, 10))
    cnt_any = np.zeros((7, 2, 10))
    cnt_slot = np.zeros((7, 2))

    for i in range(n):
        if hist_t:
            rt15, ru15 = hist_t[-15:], hist_u[-15:]
            rt50, ru50 = hist_t[-50:], hist_u[-50:]
            d_lag1 = {hist_t[-1], hist_u[-1]}
            d_lag2 = {hist_t[-2], hist_u[-2]} if len(hist_t) >= 2 else set()
            last_digits = d_lag1 | d_lag2
            for k in range(10):
                tens_roll15[i, k] = rt15.count(k) / len(rt15)
                units_roll15[i, k] = ru15.count(k) / len(ru15)
                tens_roll50[i, k] = rt50.count(k) / len(rt50)
                units_roll50[i, k] = ru50.count(k) / len(ru50)
                recent_carry[i, k] = 1.0 if k in last_digits else 0.0
                lag1_digit_flag[i, k] = 1.0 if k in d_lag1 else 0.0
                lag2_digit_flag[i, k] = 1.0 if k in d_lag2 else 0.0

        w, pm = int(dows[i]), int(pms[i])
        denom = cnt_slot[w, pm] + 20.0
        dow_t_rate[i] = (cnt_t[w, pm] + 2.0) / denom
        dow_u_rate[i] = (cnt_u[w, pm] + 2.0) / denom
        dow_any_rate[i] = (cnt_any[w, pm] + 3.8) / denom

        if not np.isnan(nums[i]):
            val = int(nums[i])
            tv, uv = val // 10, val % 10
            hist_t.append(tv)
            hist_u.append(uv)
            cnt_t[w, pm, tv] += 1.0
            cnt_u[w, pm, uv] += 1.0
            cnt_any[w, pm, tv] += 1.0
            if uv != tv:
                cnt_any[w, pm, uv] += 1.0
            cnt_slot[w, pm] += 1.0

    # Baseline columns (current production feature set)
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
        d[f"lag1_digit_{k}"] = lag1_digit_flag[:, k]
        d[f"lag2_digit_{k}"] = lag2_digit_flag[:, k]
        base_cols += [
            f"tens_roll15_{k}", f"units_roll15_{k}", f"tens_roll50_{k}", f"units_roll50_{k}",
            f"recent_carry_{k}", f"dow_pm_tens_rate_{k}", f"dow_pm_units_rate_{k}", f"dow_pm_any_rate_{k}",
        ]

    # Diverse non-linear lag + intraday/overnight features for Binary Anchor Specialists
    specialist_cols = [
        "Is_PM", "DayOfWeek", "sin_doy", "cos_doy",
        "is_intraday_am_to_pm", "is_overnight_pm_to_am", "is_monday_open",
        "lag1_t_x_pm", "lag1_u_x_pm", "lag1_b_x_pm", "lag1_is_double",
    ]
    for lag in [1, 2, 3, 4, 5]:
        specialist_cols += [f"lag_{lag}_tens", f"lag_{lag}_units", f"lag_{lag}_brake"]
    for k in range(10):
        specialist_cols += [
            f"lag1_digit_{k}", f"lag2_digit_{k}",
            f"dow_pm_any_rate_{k}", f"tens_roll15_{k}", f"units_roll15_{k}",
        ]

    return d, base_cols, specialist_cols


def predict_bayesian_pure_transition(hist: pd.DataFrame, target_row: pd.Series) -> np.ndarray:
    """Diverse Model 1: Pure Day-of-Week x Session + Intraday/Overnight Markov Transition (no roll15 duplication)."""
    nums = hist["Winning_Number"].astype(int).values
    tens = nums // 10
    units = nums % 10

    d_dt = pd.to_datetime(hist["Date"])
    dow_hist = d_dt.dt.dayofweek.values
    is_pm_hist = hist["Is_PM"].values
    t_dow = pd.to_datetime(target_row["Date"]).dayofweek
    t_pm = int(target_row["Is_PM"])

    # 1. Exact Day-of-Week x Session empirical distribution
    mask = (dow_hist == t_dow) & (is_pm_hist == t_pm)
    sub_nums = nums[mask]
    p_exact_dow = (np.bincount(sub_nums, minlength=100) + 0.35) / (len(sub_nums) + 35.0)

    # 2. Separate Intraday (AM->PM) vs Overnight (PM->AM) 1st-order transition matrices
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

    # 3. Same-day AM digit carryover boost when predicting PM
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


def fit_three_head_models(X_train: pd.DataFrame, y_num: np.ndarray):
    y_t = y_num // 10
    y_u = y_num % 10
    y_b = (y_t + y_u) % 10
    fitted = {}

    for m_name in ["LightGBM", "CatBoost", "L2_Logistic"]:
        heads = []
        for target in (y_t, y_u, y_b):
            if m_name == "LightGBM":
                clf = lgb.LGBMClassifier(
                    objective="multiclass", num_class=10, n_estimators=100,
                    learning_rate=0.04, num_leaves=15, subsample_for_bin=500,
                    min_child_samples=20, n_jobs=-1, random_state=42, verbose=-1
                )
            elif m_name == "CatBoost":
                clf = CatBoostClassifier(
                    iterations=130, learning_rate=0.05, depth=4,
                    loss_function="MultiClass", thread_count=-1, random_seed=42, verbose=0
                )
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
            clf.fit(X_train, target)
            heads.append(clf)
        fitted[m_name] = tuple(heads)
    return fitted


def fit_binary_anchor_specialists(X_train: pd.DataFrame, y_num: np.ndarray):
    """Trains 10 One-vs-Rest binary classifiers (digit k in Tens OR Units) for LGBM and CatBoost."""
    y_t = y_num // 10
    y_u = y_num % 10
    lgb_bins = []
    cb_bins = []

    for k in range(10):
        y_bin = ((y_t == k) | (y_u == k)).astype(int)
        clf_l = lgb.LGBMClassifier(
            objective="binary", n_estimators=90, learning_rate=0.04,
            num_leaves=12, min_child_samples=20, n_jobs=-1, random_state=42 + k, verbose=-1
        )
        clf_l.fit(X_train, y_bin)
        lgb_bins.append(clf_l)

        clf_c = CatBoostClassifier(
            iterations=110, learning_rate=0.05, depth=4,
            loss_function="Logloss", thread_count=-1, random_seed=42 + k, verbose=0
        )
        clf_c.fit(X_train, y_bin)
        cb_bins.append(clf_c)

    return lgb_bins, cb_bins


def combine_t_u_brake(pt: np.ndarray, pu: np.ndarray, pb: np.ndarray) -> np.ndarray:
    grid = (np.outer(pt, pu) * (pb[BRAKE_IDX] * 10.0)).flatten()
    grid = np.maximum(grid, 1e-8)
    return grid / grid.sum()


def anchor_probs_to_100_grid(p_anc_10: np.ndarray) -> np.ndarray:
    """Projects 10 any-position anchor probabilities onto the 100-pair grid."""
    grid = np.zeros(100)
    for num in range(100):
        t, u = num // 10, num % 10
        if t == u:
            grid[num] = p_anc_10[t] * 0.55
        else:
            grid[num] = 0.5 * (p_anc_10[t] + p_anc_10[u])
    grid = np.maximum(grid, 1e-8)
    return grid / grid.sum()


def grid_to_anchor_probs(p_100: np.ndarray) -> np.ndarray:
    mat = p_100.reshape(10, 10)
    p_t = mat.sum(axis=1)
    p_u = mat.sum(axis=0)
    return p_t + p_u - (p_t * p_u)


def evaluate_predictions(p_100: np.ndarray, actual: int, override_anchors=None):
    if override_anchors is not None:
        d1, d2 = override_anchors
    else:
        p_anc = grid_to_anchor_probs(p_100)
        ranked_d = np.argsort(p_anc)[::-1]
        d1, d2 = int(ranked_d[0]), int(ranked_d[1])

    top_pairs = []
    for idx in np.argsort(p_100)[::-1]:
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
    nll = -np.log(max(float(p_100[actual]), 1e-6))
    return hit_d1, hit_dual, hit_exact, hit_exact_rev, nll


def main():
    t0 = time.time()
    df = pd.read_csv(DRAWS_PATH)
    df = df[df["Winning_Number"].notna()].copy().reset_index(drop=True)
    df["Winning_Number"] = df["Winning_Number"].astype(int)

    feat_df, base_cols, spec_cols = build_features(df)
    n_total = len(df)
    eval_window = min(300, n_total - 200)
    start_idx = n_total - eval_window
    step_size = 50
    ROLLING_WINDOW = 600

    configs = [
        "C1_Prod_FullHist_3Head",
        "C2_Rolling600_RegimeWindow",
        "C3_Binary_Anchor_Specialists",
        "C4_Full_Diverse_Ensemble",
    ]
    metrics = {
        c: {"d1": 0, "dual": 0, "exact": 0, "exact_rev": 0, "nll": 0.0, "n": 0}
        for c in configs
    }

    for fold_start in range(start_idx, n_total, step_size):
        fold_end = min(fold_start + step_size, n_total)

        # Full history vs Rolling 600-draw window
        full_mask = (feat_df.index >= 14) & (feat_df.index < fold_start)
        roll_start = max(14, fold_start - ROLLING_WINDOW)
        roll_mask = (feat_df.index >= roll_start) & (feat_df.index < fold_start)

        models_full = fit_three_head_models(
            feat_df.loc[full_mask, base_cols], df.loc[full_mask, "Winning_Number"].values
        )
        models_roll = fit_three_head_models(
            feat_df.loc[roll_mask, base_cols], df.loc[roll_mask, "Winning_Number"].values
        )
        lgb_bins, cb_bins = fit_binary_anchor_specialists(
            feat_df.loc[roll_mask, spec_cols], df.loc[roll_mask, "Winning_Number"].values
        )

        X_base_slice = feat_df.loc[fold_start:fold_end - 1, base_cols]
        X_spec_slice = feat_df.loc[fold_start:fold_end - 1, spec_cols]

        probs_full = {
            m: tuple(clf.predict_proba(X_base_slice) for clf in heads)
            for m, heads in models_full.items()
        }
        probs_roll = {
            m: tuple(clf.predict_proba(X_base_slice) for clf in heads)
            for m, heads in models_roll.items()
        }

        # Batch predict binary any-position probabilities: shape (fold_len, 10)
        p_anc_lgb = np.column_stack([clf.predict_proba(X_spec_slice)[:, 1] for clf in lgb_bins])
        p_anc_cb = np.column_stack([clf.predict_proba(X_spec_slice)[:, 1] for clf in cb_bins])

        for offset, idx in enumerate(range(fold_start, fold_end)):
            actual = int(df.loc[idx, "Winning_Number"])
            hist_roll = df.iloc[roll_start:idx]
            target_row = df.iloc[idx]
            p_bayes = predict_bayesian_pure_transition(hist_roll, target_row)

            # C1: Current Production (Full History + 3-Head)
            g_c1 = [p_bayes]
            for m_name, (pt_a, pu_a, pb_a) in probs_full.items():
                g_c1.append(combine_t_u_brake(pt_a[offset], pu_a[offset], pb_a[offset]))
            p_c1 = sum(g_c1) / len(g_c1)

            # C2: Rolling 600-Draw Window + 3-Head
            g_c2 = [p_bayes]
            for m_name, (pt_a, pu_a, pb_a) in probs_roll.items():
                g_c2.append(combine_t_u_brake(pt_a[offset], pu_a[offset], pb_a[offset]))
            p_c2 = sum(g_c2) / len(g_c2)

            # C3: Binary Any-Position Anchor Specialists (LGBM + CatBoost One-vs-Rest)
            p_bin_anc = 0.5 * p_anc_lgb[offset] + 0.5 * p_anc_cb[offset]
            top2_bin = np.argsort(p_bin_anc)[::-1][:2]
            d1_c3, d2_c3 = int(top2_bin[0]), int(top2_bin[1])
            p_c3_grid = 0.5 * anchor_probs_to_100_grid(p_bin_anc) + 0.5 * p_c2

            # C4: Full Diverse Ensemble (Binary Specialists for Anchors + Transition Markov + Rolling 3-Head)
            p_grid_anc = grid_to_anchor_probs(p_c2)
            p_ens_anc = 0.55 * (p_bin_anc / p_bin_anc.sum() * 1.9) + 0.45 * p_grid_anc
            top2_ens = np.argsort(p_ens_anc)[::-1][:2]
            d1_c4, d2_c4 = int(top2_ens[0]), int(top2_ens[1])
            p_c4_grid = 0.35 * anchor_probs_to_100_grid(p_bin_anc) + 0.65 * p_c2

            for cfg_name, p_grid, ov_anc in [
                ("C1_Prod_FullHist_3Head", p_c1, None),
                ("C2_Rolling600_RegimeWindow", p_c2, None),
                ("C3_Binary_Anchor_Specialists", p_c3_grid, (d1_c3, d2_c3)),
                ("C4_Full_Diverse_Ensemble", p_c4_grid, (d1_c4, d2_c4)),
            ]:
                d1, dual, ex, ex_r, nll = evaluate_predictions(p_grid, actual, override_anchors=ov_anc)
                metrics[cfg_name]["d1"] += d1
                metrics[cfg_name]["dual"] += dual
                metrics[cfg_name]["exact"] += ex
                metrics[cfg_name]["exact_rev"] += ex_r
                metrics[cfg_name]["nll"] += nll
                metrics[cfg_name]["n"] += 1

    elapsed = time.time() - t0
    lines = [
        f"### SAI 2D Diverse Ensemble Backtest ({eval_window} Out-of-Sample Draws • Completed in {elapsed:.1f}s)",
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
