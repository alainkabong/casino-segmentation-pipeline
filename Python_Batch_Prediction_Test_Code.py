import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

# =====================================================================
# Configuration
# =====================================================================
NEW_DATA_PATH = r"C:\Users\MwambKabong\Downloads\Casino_Segementation_Pipeline\casino_table_final_october_panthera_only.csv"
ARTIFACT_PATH = r"C:\Users\MwambKabong\Downloads\Casino_Segementation_Pipeline\artifacts\deployment_artifacts.joblib"
BASELINE_PATH = r"C:\Users\MwambKabong\Downloads\Casino_Segementation_Pipeline\artifacts\training_baseline.json"
OUTPUT_DIR = r"C:\Users\MwambKabong\Downloads\Casino_Segementation_Pipeline\artifacts\batch_predictions"


RANDOM_STATE = 42
BUSINESS_PROTECTED_FEATURES = ["Cash_Stake_Casino", "Cash_Stake_Sports", "Total_Deposit_Amount", "bonus_dependency", "tenure_days"]


NGR_RECENT_WINDOW_DAYS = 10
NGR_RECENT_COL = f"NGR_last_{NGR_RECENT_WINDOW_DAYS}_days"


def banner(msg):
    print("\n" + "=" * 76)
    print(msg)
    print("=" * 76)


def subbanner(msg):
    print(f"\n\u2500\u2500 {msg} \u2500\u2500")


# =====================================================================
# STEP 1: Loading and normalise the raw extract
# =====================================================================
def load_and_normalize(path):
    banner(f"STEP 1: Loading and normalising {os.path.basename(path)}")
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    _EXPECTED_COLUMN_CASE = [
        "Customer_ID", "Game_ID", "Global_Customer_ID", "Country_Code", "Language_Code", "Site", "Gender",
        "Total_Deposit_Count", "Total_Deposit_Amount", "Cash_Balance", "Bonus_Balance", "Registration_Device",
        "Acquisition_Channel", "Flags", "Allow_Transactional_Comms", "Allow_Marketing_Comms",
        "Allow_Email_Comms", "Allow_SMS_Comms", "Allow_Bonuses", "Is_Ibex_AB", "DOB", "Reg_Date",
        "First_Deposit_Date", "Last_Deposit_Date", "Last_Login_Date", "Device", "Deposit_Count",
        "Deposit_Amount", "Withdrawal_Amount", "Cash_Stake_Casino", "Cash_Return_Casino", "Cash_Stake_Sports",
        "Cash_Return_Sports", "GGR", "Bonus_Stake", "Bonus_Stake_bets", "Bonus_Return", "Bonus_Return_bets",
        "Bonus_Given", "Bonus_To_Cash", "NGR", "Tax", "Payment_Fee", "Game_Provider_Fee",
        "Jackpot_Contribution", "Revenue_Share", "Casino_Income", "Closing_Cash_Balance",
        "Closing_Bonus_Balance", "Activity_Date", "Bet_Count", "Cash_Stake", "Cash_Return", "Bet_Date",
        "Game_Type", "Game_Name", "Supplier",
    ]
    _lower_to_expected = {c.lower(): c for c in _EXPECTED_COLUMN_CASE}
    _rename_map = {c: _lower_to_expected[c.lower()] for c in df.columns
                   if c.lower() in _lower_to_expected and c != _lower_to_expected[c.lower()]}
    if _rename_map:
        print(f"Normalised {len(_rename_map)} column name(s) to expected casing.")
        df = df.rename(columns=_rename_map)

    _alternate_names = {
        "activity_bonus_stake": "Bonus_Stake", "activity_bonus_return": "Bonus_Return",
        "bets_bonus_stake": "Bonus_Stake_bets", "bets_bonus_return": "Bonus_Return_bets",
    }
    _alt_rename_map = {c: _alternate_names[c.lower()] for c in df.columns
                        if c.lower() in _alternate_names and _alternate_names[c.lower()] not in df.columns}
    if _alt_rename_map:
        print(f"Renamed {len(_alt_rename_map)} alternate-named column(s): {_alt_rename_map}")
        df = df.rename(columns=_alt_rename_map)

    if "Customer_ID" not in df.columns:
        raise ValueError("Customer_ID not found after normalisation -- check the extract's column names.")

    return df


# =====================================================================
# STEP 2: Row-level cleanup, date parsing
# =====================================================================
def aggregate_to_customer_level(df):
    banner("STEP 2: Aggregating raw rows to customer level")

    df = df.replace(["NULL", "null", "None", "", " "], np.nan)
    date_columns = ["DOB", "Reg_Date", "First_Deposit_Date", "Last_Deposit_Date",
                     "Last_Login_Date", "Activity_Date", "Bet_Date"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

    snapshot_date = df[["Reg_Date", "First_Deposit_Date", "Last_Login_Date", "Last_Deposit_Date",
                         "Activity_Date", "Bet_Date"]].max().max().normalize() + pd.Timedelta(days=1)
    print(f"snapshot_date for this extract: {snapshot_date}")

    if "Cash_Return" not in df.columns and "Cash_Return_Casino" in df.columns and "Cash_Return_Sports" in df.columns:
        df["Cash_Return"] = df["Cash_Return_Casino"].fillna(0) + df["Cash_Return_Sports"].fillna(0)
    if all(c in df.columns for c in ["Cash_Stake", "Cash_Stake_Casino", "Cash_Stake_Sports"]):
        mask = df["Cash_Stake"].isna() & df["Cash_Stake_Casino"].notna() & df["Cash_Stake_Sports"].notna()
        df.loc[mask, "Cash_Stake"] = df.loc[mask, "Cash_Stake_Casino"] + df.loc[mask, "Cash_Stake_Sports"]
    
    if all(c in df.columns for c in ["Cash_Stake_Casino", "Cash_Return_Casino"]):
        df["GGR"] = df["Cash_Stake_Casino"] - df["Cash_Return_Casino"]

    ngr_cost_cols = ["Bonus_Given", "Revenue_Share", "Tax", "Payment_Fee", "Game_Provider_Fee"]
    for col in ngr_cost_cols:
        if col not in df.columns:
            df[col] = 0
    
    if "NGR" in df.columns and "GGR" in df.columns:
        mask = df["NGR"].isna() & df["GGR"].notna()
        df.loc[mask, "NGR"] = (df.loc[mask, "GGR"] - df.loc[mask, "Bonus_Given"].fillna(0)
                                - df.loc[mask, "Revenue_Share"].fillna(0) - df.loc[mask, "Tax"].fillna(0)
                                - df.loc[mask, "Payment_Fee"].fillna(0) - df.loc[mask, "Game_Provider_Fee"].fillna(0))

    if "Last_Login_Date" in df.columns:
        df["never_login_flag"] = df["Last_Login_Date"].isna().astype(int)
        df["days_since_login"] = (snapshot_date - df["Last_Login_Date"]).dt.days
        df["days_since_login"] = df["days_since_login"].fillna(df["days_since_login"].max() + 1)
    if "Last_Deposit_Date" in df.columns:
        df["never_deposit_flag"] = df["Last_Deposit_Date"].isna().astype(int)
        df["days_since_deposit"] = (snapshot_date - df["Last_Deposit_Date"]).dt.days
        df["days_since_deposit"] = df["days_since_deposit"].fillna(df["days_since_deposit"].max() + 1)
    df = df.drop(columns=["DOB", "Device"], errors="ignore")

    zero_fill_cols = ["Bonus_Stake", "Bonus_Given", "Tax", "Payment_Fee", "Game_Provider_Fee",
                       "Jackpot_Contribution", "Revenue_Share", "Casino_Income",
                       "Closing_Cash_Balance", "Closing_Bonus_Balance"]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "Flags" in df.columns:
        n_null_flags = df["Flags"].isna().sum()
        if n_null_flags == len(df):
            print(f"\u26a0\ufe0f  'Flags' (VIP tier source) is 100% null in this extract -- every customer will")
            print(f"default to 'Non-VIP' per the pipeline's own business rule. This is expected behaviour,")
            print(f"not a bug, but worth knowing if you were expecting real VIP-tier information here.")
        df["Flags"] = df["Flags"].fillna("Non-VIP")

    print(f"Row-level nulls remaining after cleanup: {df.isnull().sum().sum():,}")

    subbanner("Part B: customer-level aggregation")

    def safe_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else "Unknown"

    
    first_cols = ["Global_Customer_ID", "Country_Code", "Language_Code", "Site", "Gender",
                  "Registration_Device", "Acquisition_Channel", "Allow_Transactional_Comms",
                  "Allow_Marketing_Comms", "Allow_Email_Comms", "Allow_SMS_Comms", "Allow_Bonuses",
                  "Is_Ibex_AB", "Reg_Date", "First_Deposit_Date", "Last_Login_Date", "Last_Deposit_Date",
                  "Total_Deposit_Count", "Total_Deposit_Amount", "Cash_Balance", "Bonus_Balance", "Flags",
                  "days_since_login", "days_since_deposit", "never_login_flag", "never_deposit_flag"]
    sum_cols = ["Deposit_Count", "Deposit_Amount", "Withdrawal_Amount", "Cash_Stake_Casino", "Cash_Return_Casino",
                "Cash_Stake_Sports", "Cash_Return_Sports", "GGR", "NGR", "Bet_Count", "Cash_Stake", "Cash_Return",
                "Bonus_Stake", "Bonus_Stake_bets", "Bonus_Return", "Bonus_Return_bets", "Bonus_Given",
                "Bonus_To_Cash", "Tax", "Payment_Fee", "Game_Provider_Fee", "Jackpot_Contribution",
                "Revenue_Share", "Casino_Income"]
    agg_dict = {c: "first" for c in first_cols if c in df.columns}
    agg_dict.update({c: "sum" for c in sum_cols if c in df.columns})
    customer_df = df.groupby("Customer_ID").agg(agg_dict)
    customer_df["active_days"] = df.groupby("Customer_ID")["Activity_Date"].nunique()

    if "Game_ID" in df.columns:
        customer_df["distinct_games_played"] = df.groupby("Customer_ID")["Game_ID"].nunique()
    else:
        print("Game_ID not present in this extract -- distinct_games_played cannot be computed, skipped.")
    if "Game_Type" in df.columns:
        customer_df["favorite_game_type"] = df.groupby("Customer_ID")["Game_Type"].agg(safe_mode)
    elif "favorite_game_type" in df.columns:
        customer_df["favorite_game_type"] = df.groupby("Customer_ID")["favorite_game_type"].first()
    if "Supplier" in df.columns:
        customer_df["favorite_supplier"] = df.groupby("Customer_ID")["Supplier"].agg(safe_mode)
    elif "favorite_supplier" in df.columns:
        customer_df["favorite_supplier"] = df.groupby("Customer_ID")["favorite_supplier"].first()
    
    if "Game_Name" in df.columns:
        customer_df["favorite_game_name"] = df.groupby("Customer_ID")["Game_Name"].agg(safe_mode)
    elif "favorite_game_name" in df.columns:
        customer_df["favorite_game_name"] = df.groupby("Customer_ID")["favorite_game_name"].first()

    customer_df = customer_df.reset_index()

    
    if "Flags" in customer_df.columns:
        vip_excluded_mask = customer_df["Flags"].isin(["_silver_vip", "_gold_vip", "_platinum_vip"])
        n_vip_excluded = int(vip_excluded_mask.sum())
        if n_vip_excluded:
            print(f"Dropping {n_vip_excluded:,} silver/gold/platinum VIP customers from the modelling "
                  f"population (Non-VIP and _bronze_vip are kept).")
            customer_df = customer_df.loc[~vip_excluded_mask].reset_index(drop=True)

    vip_order = {"Non-VIP": 0, "_bronze_vip": 1}
    customer_df["vip_rank"] = customer_df["Flags"].map(vip_order)
    customer_df = customer_df.rename(columns={"Flags": "vip_tier"})
    customer_df["tenure_days"] = (snapshot_date - customer_df["Reg_Date"]).dt.days
    print(f"Aggregated to {len(customer_df):,} unique customers.")

    subbanner("Part C: derived ratio features")
    customer_df["total_stake_verified"] = customer_df["Cash_Stake_Casino"] + customer_df["Cash_Stake_Sports"]
    
    customer_df["theoretical_ggr"] = customer_df["Cash_Stake_Casino"] * 0.03
    customer_df["return_rate"] = np.where(customer_df["total_stake_verified"] > 0,
                                           customer_df["Cash_Return"] / customer_df["total_stake_verified"], np.nan)
    customer_df["casino_affinity"] = np.where(customer_df["total_stake_verified"] > 0,
                                               customer_df["Cash_Stake_Casino"] / customer_df["total_stake_verified"], np.nan)
    customer_df["bonus_dependency"] = np.where(customer_df["total_stake_verified"] > 0,
                                                customer_df["Bonus_Stake"] / customer_df["total_stake_verified"], np.nan)
   
    customer_df["bonus_per_bet"] = np.where(customer_df["Bet_Count"] > 0,
                                             customer_df["Bonus_Stake"] / customer_df["Bet_Count"], np.nan)
    customer_df["stake_per_active_day"] = np.where(customer_df["active_days"] > 0,
                                                    customer_df["Cash_Stake_Casino"] / customer_df["active_days"], np.nan)
    customer_df["avg_stake_per_bet"] = np.where(customer_df["Bet_Count"] > 0,
                                                 customer_df["Cash_Stake_Casino"] / customer_df["Bet_Count"], np.nan)
    customer_df["deposits_per_active_day"] = np.where(customer_df["active_days"] > 0,
                                                       customer_df["Deposit_Count"] / customer_df["active_days"], np.nan)

  
    subbanner(f"Computing {NGR_RECENT_COL} (recent-window profitability)")
    
    window_end = df["Activity_Date"].max().normalize() + pd.Timedelta(days=1)
    window_start = window_end - pd.Timedelta(days=NGR_RECENT_WINDOW_DAYS)
    recent_mask = df["Activity_Date"] >= window_start
    print(f"Window: {window_start.date()} to {(window_end - pd.Timedelta(days=1)).date()} "
          f"({NGR_RECENT_WINDOW_DAYS} days, ending on the last activity date) -- "
          f"{recent_mask.sum():,} / {len(df):,} rows fall inside it.")
    ngr_recent = df.loc[recent_mask].groupby("Customer_ID")["NGR"].sum()
    customer_df[NGR_RECENT_COL] = customer_df["Customer_ID"].map(ngr_recent).fillna(0.0)
    print(f"{NGR_RECENT_COL}: mean={customer_df[NGR_RECENT_COL].mean():.2f}  "
          f"median={customer_df[NGR_RECENT_COL].median():.2f}  "
          f"customers with $0 (no recent activity): {(customer_df[NGR_RECENT_COL] == 0).sum():,}")

    subbanner("Zero-cash-funded cohort split (business rule)")
    pre_activation_mask = customer_df["total_stake_verified"] == 0
    pre_activation_df = customer_df.loc[pre_activation_mask].copy()
    modelling_df = customer_df.loc[~pre_activation_mask].copy().reset_index(drop=True)
    print(f"Zero-cash-funded cohort (no cash stake on casino or sports): {len(pre_activation_df):,} "
          f"({len(pre_activation_df) / len(customer_df) * 100:.2f}%) -- routed OUT of scoring, matching")
    print(f"the training-time rule exactly (these never receive a segment_id).")
    print(f"Modelling population: {len(modelling_df):,} ({len(modelling_df) / len(customer_df) * 100:.2f}%)")

    return modelling_df, pre_activation_df, snapshot_date


# =====================================================================
# STEP 3: Batch scoring 
# =====================================================================
def compute_bonus_worthiness_features(new_df):
    
    cols = ["Customer_ID", "Total_Deposit_Amount", "Cash_Stake_Casino", "total_stake_verified",
            "tenure_days", "days_since_login", "days_since_deposit",
            "bonus_dependency", "bonus_per_bet", "Bonus_To_Cash", "Bonus_Given",
            "return_rate", "casino_affinity", "NGR"]
    missing = [c for c in cols if c not in new_df.columns]
    if missing:
        
        print(f"\u26a0\ufe0f  compute_bonus_worthiness_features(): expected column(s) missing from input data:")
        print(f"{missing} -- the corresponding worthiness component(s) will be constant/uninformative for")
        print(f"every customer, not a real per-customer signal. Check aggregate_to_customer_level()'s own")
        print(f"first_cols/sum_cols if this is unexpected.")
    present = [c for c in cols if c in new_df.columns]
    return new_df[present].copy()


def run_diagnostic_pca(features_df):
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    banner("Diagnostic PCA: true dimensionality of the underlying features")
    f = features_df
    bonus_given = f.get("Bonus_Given", pd.Series(0, index=f.index)).replace(0, np.nan)
    bonus_conversion = (f.get("Bonus_To_Cash", pd.Series(0, index=f.index)) / bonus_given).fillna(0.0)

    raw = pd.DataFrame({
        "Total_Deposit_Amount": f.get("Total_Deposit_Amount", pd.Series(0, index=f.index)),
        "Cash_Stake_Casino": f.get("Cash_Stake_Casino", pd.Series(0, index=f.index)),
        "recency_login_inv": -f.get("days_since_login", pd.Series(0, index=f.index)),
        "recency_deposit_inv": -f.get("days_since_deposit", pd.Series(0, index=f.index)),
        "bonus_dependency": f.get("bonus_dependency", pd.Series(0, index=f.index)),
        "bonus_per_bet": f.get("bonus_per_bet", pd.Series(0, index=f.index)),
        "bonus_conversion": bonus_conversion,
    }).fillna(0)

    X = StandardScaler().fit_transform(raw)
    pca = PCA()
    pca.fit(X)
    print(f"Explained variance ratio per component:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {v*100:.1f}%  (eigenvalue={pca.explained_variance_[i]:.3f}, "
              f"{'keep -- Kaiser criterion' if pca.explained_variance_[i] > 1 else 'below Kaiser threshold'})")
    n_keep = int((pca.explained_variance_ > 1).sum())
    print(f"{n_keep} component(s) clear the Kaiser criterion (eigenvalue > 1) -- suggests this many")
    print(f"genuinely independent dimensions actually exist in this run's data, not assumed to be 4 just")
    print(f"because there are 4 named business components below. ONLY these {n_keep} are used for the")
    print(f"importance calculation below -- the rest are noise-level directions, not real signal.")

    subbanner("Loadings (first 4 PCs)")
    loadings_display = pd.DataFrame(pca.components_[:4].T, index=raw.columns,
                                     columns=[f"PC{i+1}" for i in range(min(4, pca.n_components_))])
    print(loadings_display.round(3).to_string())
    print(f"\n Read this as: features loading heavily on the SAME PC are entangled in the real data;")
    print(f"features that load almost purely on their OWN PC (near-zero elsewhere) are genuinely")
    print(f"independent -- exactly what informs which components get orthogonalized against which")
    print(f"below, rather than assuming a fixed structure regardless of what this run's data shows.")

    subbanner(f"Group importance (communality), using only the {n_keep} retained component(s)")
    loadings_kept = pca.components_.T[:, :n_keep]
    ev_kept = pca.explained_variance_[:n_keep]
    communality = pd.Series((loadings_kept ** 2 * ev_kept[np.newaxis, :]).sum(axis=1), index=raw.columns)
    print(communality.round(3).to_string())

    groups = {
        "value": ["Total_Deposit_Amount", "Cash_Stake_Casino"],
        "recency": ["recency_login_inv", "recency_deposit_inv"],
        "responsiveness": ["bonus_dependency", "bonus_per_bet"],
        "engagement": ["bonus_conversion"],
    }
    group_importance = {g: float(communality[cols].sum()) for g, cols in groups.items()}
    total = sum(group_importance.values())
    pca_derived_weights = {g: v / total for g, v in group_importance.items()}
    print(f"\n PCA-derived target weights (genuinely data-driven, not hand-picked):")
    for g, v in pca_derived_weights.items():
        print(f"     {g:<16} {v*100:.1f}%")

    return pca_derived_weights


def score_bonus_worthiness(features_df, pca_derived_weights=None):
    
    f = features_df.copy()

    def pct_rank(s):
        return s.rank(pct=True, na_option="bottom").fillna(0.0)

    def orthogonalize(target, *already_used):
        
        if not already_used:
            return target
        X = np.column_stack([np.ones(len(target))] + [a.values for a in already_used])
        coef, _, _, _ = np.linalg.lstsq(X, target.values, rcond=None)
        residual = target.values - X @ coef
        return pct_rank(pd.Series(residual, index=target.index))

    value_score = 0.5 * pct_rank(f.get("Total_Deposit_Amount", pd.Series(0, index=f.index))) + \
                  0.5 * pct_rank(f.get("Cash_Stake_Casino", pd.Series(0, index=f.index)))

   
    recency_raw = 0.5 * pct_rank(-f.get("days_since_login", pd.Series(0, index=f.index))) + \
                  0.5 * pct_rank(-f.get("days_since_deposit", pd.Series(0, index=f.index)))
    recency_score = orthogonalize(recency_raw, value_score)

   
    
    bonus_dep = f.get("bonus_dependency", pd.Series(0, index=f.index)).fillna(0)
    ngr = f.get("NGR", pd.Series(0, index=f.index)).fillna(0)
    is_zero = bonus_dep == 0
    nonzero_mask = ~is_zero

    dependency_shaped = pd.Series(0.5, index=f.index)
    N_BINS = 10
    if nonzero_mask.sum() > N_BINS * 2:
        nonzero_pct = pct_rank(bonus_dep[nonzero_mask])
        bins = pd.qcut(nonzero_pct, N_BINS, labels=False, duplicates="drop")
        bin_ngr_means = pd.Series(ngr[nonzero_mask].values, index=nonzero_pct.index).groupby(bins).mean()
        zero_ngr = ngr[is_zero].mean() if is_zero.any() else np.nan

        if len(bin_ngr_means) > 1 and bin_ngr_means.notna().any():
            all_group_ngr = bin_ngr_means.copy()
            if not pd.isna(zero_ngr):
                all_group_ngr.loc["zero"] = zero_ngr
            optimal_ngr = all_group_ngr.max()
            worst_ngr = all_group_ngr.min()
            ngr_range = max(optimal_ngr - worst_ngr, 1e-9)

            
            optimal_bin = bin_ngr_means.idxmax()
            n_actual_bins = int(bins.max()) + 1
            optimal_bin_pct_center = (optimal_bin + 0.5) / n_actual_bins
            nonzero_shaped = 1.0 - (nonzero_pct - optimal_bin_pct_center).abs()
            dependency_shaped.loc[nonzero_mask] = nonzero_shaped.values

           
            if not pd.isna(zero_ngr):
                zero_score = float(np.clip((zero_ngr - worst_ngr) / ngr_range, 0.0, 1.0))
                dependency_shaped.loc[is_zero] = zero_score

    dependency_shaped = pct_rank(dependency_shaped)

    responsiveness_raw = 0.6 * dependency_shaped + \
                          0.4 * pct_rank(f.get("bonus_per_bet", pd.Series(0, index=f.index)))
    responsiveness_score = orthogonalize(responsiveness_raw, value_score, recency_score)

    bonus_given = f.get("Bonus_Given", pd.Series(0, index=f.index)).replace(0, np.nan)
    conversion_ratio = (f.get("Bonus_To_Cash", pd.Series(0, index=f.index)) / bonus_given).fillna(0.0)
    
    engagement_score = pct_rank(conversion_ratio)

    
    if pca_derived_weights is not None:
        target_weights = pca_derived_weights
        print(f"Using PCA-derived target weights (from run_diagnostic_pca()'s communality analysis on")
        print(f"THIS run's real data): {({k: round(v,3) for k,v in target_weights.items()})}")
    else:
        target_weights = {"value": 0.25, "recency": 0.25, "responsiveness": 0.25, "engagement": 0.25}
        print(f"\u26a0\ufe0f  No PCA-derived weights supplied -- falling back to an even 25/25/25/25 split (an")
        print(f"honest 'no informed opinion' default, not a disguised hand-picked one). Call")
        print(f"run_diagnostic_pca() first and pass its result in for real, data-driven targets.")
    raw_components = {"value": value_score, "recency": recency_score,
                       "responsiveness": responsiveness_score, "engagement": engagement_score}
    variances = {name: s.var() for name, s in raw_components.items()}

    subbanner("Recalibrating component weights to match the PCA-DERIVED variance contribution")
    MIN_VARIANCE = 1e-6
    adjusted_weights = {}
    for name, target in target_weights.items():
        v = max(variances[name], MIN_VARIANCE)
        adjusted_weights[name] = float(np.sqrt(target / v))
    print(f"{'Component':<16} {'Target %':>10} {'Variance':>12} {'Adjusted coef':>15}")
    for name in target_weights:
        flag = ""
        if variances[name] < 0.01:
            flag = "\u26a0\ufe0f near-degenerate component -- adjusted coefficient amplifies a signal that's"
        print(f"{name:<16} {target_weights[name]*100:>9.1f}% {variances[name]:>12.5f} "
              f"{adjusted_weights[name]:>15.3f}{flag}")
        if flag:
            print(f"mostly tied/constant across the population -- see engagement's own")
            print(f"99% zero-inflation note above; a handful of customers with real signal")
            print(f"now swing much harder on the final score than the rest of the population.")

    composite = (adjusted_weights["value"] * value_score + adjusted_weights["recency"] * recency_score +
                 adjusted_weights["responsiveness"] * responsiveness_score +
                 adjusted_weights["engagement"] * engagement_score)
    
    composite_pct = pct_rank(composite) * 100

    
    achieved_contrib = {name: (adjusted_weights[name] ** 2) * variances[name] for name in target_weights}
    achieved_total = sum(achieved_contrib.values())
    subbanner("Achieved variance contribution after recalibration (should now match the target)")
    for name in target_weights:
        print(f"{name:<16} target={target_weights[name]*100:>5.1f}%   achieved={achieved_contrib[name]/achieved_total*100:>5.1f}%")

    return pd.DataFrame({
        "Customer_ID": f["Customer_ID"] if "Customer_ID" in f.columns else f.index,
        "value_score": np.round(value_score, 4), "recency_score": np.round(recency_score, 4),
        "responsiveness_score": np.round(responsiveness_score, 4), "engagement_score": np.round(engagement_score, 4),
        "bonus_worthiness_score": np.round(composite_pct, 2),
    })


def compute_business_rule_amount(new_df, min_amount=5.0, max_amount=500.0):
    
    tenure = new_df.get("tenure_days", pd.Series(1, index=new_df.index)).clip(lower=1)
    deposit_velocity = new_df.get("Total_Deposit_Amount", 0).fillna(0) / tenure
    stake_velocity = new_df.get("Cash_Stake_Casino", 0).fillna(0) / tenure
    value_basis = deposit_velocity + stake_velocity
    log_value = np.log1p(value_basis.clip(lower=0))
    pct = log_value.rank(pct=True, na_option="bottom").fillna(0.0)
    raw = min_amount + pct * (max_amount - min_amount)
    return np.round(raw, 2)


def compute_profitability_rule_amount(new_df, min_amount=5.0, max_amount=500.0):
    
    ngr_recent = new_df.get(NGR_RECENT_COL, pd.Series(0, index=new_df.index)).fillna(0).clip(lower=0)
    log_runrate = np.log1p(ngr_recent)
    pct = log_runrate.rank(pct=True, na_option="bottom").fillna(0.0)
    raw = min_amount + pct * (max_amount - min_amount)
    return np.round(raw, 2)


def compute_segment_aware_multiplier(results_df, segment_col="sub_segment_id", fallback_col="segment_id"):
    
    key_col = results_df[segment_col].fillna(results_df[fallback_col].astype(str)) \
        if segment_col in results_df.columns else results_df[fallback_col].astype(str)
    population_mean = results_df["bonus_worthiness_score"].mean()
    segment_means = results_df.groupby(key_col)["bonus_worthiness_score"].mean()
    multiplier_by_segment = (segment_means / population_mean).clip(lower=0.5, upper=2.0)
    multiplier = key_col.map(multiplier_by_segment)
    return multiplier, key_col, multiplier_by_segment


def combine_bonus_rules(new_df, results_df, weight_business=1/3, weight_profitability=1/3,
                         weight_segment=1/3, min_amount=0.125244, max_amount=31.310880, worthiness_threshold=70.0,
                         worthiness_tiers=None):
    
    
    if worthiness_tiers is None:
        
        
       
        worthiness_tiers = [
            (70.0, 83.64, 0.125244, 1.541),
            (83.64, 90.45, 1.541, 5.404),
            (90.45, 100.0001, 5.404, 31.310880),
        ]

    banner("Combined Bonus Amount: Business Rule + Profitability Rule + Segment-Aware Rule")
    total_weight = weight_business + weight_profitability + weight_segment
    if abs(total_weight - 1.0) > 1e-6:
        print(f"\u26a0\ufe0f  Weights sum to {total_weight:.4f}, not 1.0 -- normalising so they do.")
        weight_business, weight_profitability, weight_segment = (
            weight_business / total_weight, weight_profitability / total_weight, weight_segment / total_weight)

    amount_business = compute_business_rule_amount(new_df, min_amount=min_amount, max_amount=max_amount)
    amount_profitability = compute_profitability_rule_amount(new_df, min_amount=min_amount, max_amount=max_amount)
    segment_multiplier, segment_key, multiplier_by_segment = compute_segment_aware_multiplier(results_df)
    reference_base = (min_amount + max_amount) / 2.0
    amount_segment = np.round(np.clip(segment_multiplier * reference_base, min_amount, max_amount), 2)

    def pct_rank(s):
        return s.rank(pct=True, na_option="bottom").fillna(0.0)

    ranked_rules = {"business": pct_rank(amount_business), "profitability": pct_rank(amount_profitability),
                     "segment": pct_rank(amount_segment)}
    target = {"business": weight_business, "profitability": weight_profitability, "segment": weight_segment}
    variances = {name: s.var() for name, s in ranked_rules.items()}
    MIN_VARIANCE = 1e-6
    scale = {name: float(np.sqrt(target[name] / max(variances[name], MIN_VARIANCE))) for name in target}
    scaled = {name: ranked_rules[name] * scale[name] for name in target}
    combined_rank = sum(scaled.values())

    subbanner("Rule weighting (documented policy, not a fitted result)")
    print(f"Business rule (Deposit_Velocity + Stake_Velocity, past economic importance):")
    print(f"{weight_business*100:5.1f}% of the combined amount  --  mean=${amount_business.mean():.2f}")
    print(f"Profitability rule (NGR over the last {NGR_RECENT_WINDOW_DAYS} days, expected future value):")
    print(f"{weight_profitability*100:5.1f}% of the combined amount  --  mean=${amount_profitability.mean():.2f}")
    print(f"Segment-aware rule (segment/sub-segment policy coefficient):")
    print(f"{weight_segment*100:5.1f}% of the combined amount  --  mean=${amount_segment.mean():.2f}")

    subbanner("Achieved variance contribution after recalibration (should now match the target %)")
    achieved_var = {name: scaled[name].var() for name in target}
    achieved_total = sum(achieved_var.values())
    for name in target:
        print(f"{name:<14} target={target[name]*100:5.1f}%   achieved={achieved_var[name]/achieved_total*100:5.1f}%")

    subbanner("Segment-aware rule: actual coefficient by sub-segment")
    for seg, coef in multiplier_by_segment.sort_values(ascending=False).items():
        print(f"{seg:<8} coefficient={coef:.4f}x  (mean bonus_worthiness_score in this sub-segment: "
              f"{results_df.loc[segment_key == seg, 'bonus_worthiness_score'].mean():.1f})")

    subbanner(f"Worthiness gating (threshold: {worthiness_threshold})")
    worthiness = results_df["bonus_worthiness_score"]
    deserves_bonus = worthiness >= worthiness_threshold
    print(f"{deserves_bonus.sum():,} / {len(deserves_bonus):,} customers ({deserves_bonus.mean()*100:.2f}%) "
          f"score at or above {worthiness_threshold} and get a real bonus_amount_policy.")
    print(f"{(~deserves_bonus).sum():,} customers ({(~deserves_bonus).mean()*100:.2f}%) score below it and are")
    print(f"flagged 'No bonus - below worthiness threshold', not given a small nonzero amount anyway.")

    subbanner("Dynamic tiered scale (each tier's ceiling grows with worthiness)")
    final_amount = pd.Series(np.nan, index=new_df.index)
    tier_label = pd.Series("", index=new_df.index, dtype=object)
   
    tier_score_lower = pd.Series(np.nan, index=new_df.index)
    tier_score_upper = pd.Series(np.nan, index=new_df.index)
    tier_bonus_lower = pd.Series(np.nan, index=new_df.index)
    tier_bonus_upper = pd.Series(np.nan, index=new_df.index)
    for lo, hi, dmin, dmax in worthiness_tiers:
        in_tier = deserves_bonus & (worthiness >= lo) & (worthiness < hi)
        n_in_tier = in_tier.sum()
        if n_in_tier == 0:
            print(f"[{lo:.0f},{hi:.0f}) -> ${dmin:.0f}-${dmax:.0f}: 0 customers")
            continue
        within_tier_pct = combined_rank[in_tier].rank(pct=True, na_option="bottom").fillna(0.0)
        tier_amount = np.round(dmin + within_tier_pct * (dmax - dmin), 2)
        final_amount.loc[in_tier] = tier_amount
        tier_label.loc[in_tier] = f"[{lo:.0f},{hi:.0f})->${dmin:.0f}-${dmax:.0f}"
        tier_score_lower.loc[in_tier] = lo
        tier_score_upper.loc[in_tier] = hi
        tier_bonus_lower.loc[in_tier] = dmin
        tier_bonus_upper.loc[in_tier] = dmax
        print(f"[{lo:.0f},{hi:.0f}) -> ${dmin:.0f}-${dmax:.0f}: {n_in_tier:,} customers, "
              f"actual mean=${tier_amount.mean():.2f}")

    final_basis = np.where(deserves_bonus,
                            "policy (business + profitability + segment-aware, tiered by worthiness: " +
                            tier_label.astype(str) + ")",
                            f"No bonus - below worthiness threshold ({worthiness_threshold})")

    return pd.DataFrame({
        "Customer_ID": new_df["Customer_ID"].values,
        "bonus_amount_business_rule": amount_business.values,
        "bonus_amount_profitability_rule": amount_profitability.values,
        "segment_multiplier": np.round(segment_multiplier.values, 4),
        "bonus_amount_segment_rule": amount_segment.values,
        "deserves_bonus": deserves_bonus.values,
        "worthiness_tier": tier_label.values,
        "worthiness_tier_score_lower": tier_score_lower.values,
        "worthiness_tier_score_upper": tier_score_upper.values,
        "worthiness_tier_bonus_lower": tier_bonus_lower.values,
        "worthiness_tier_bonus_upper": tier_bonus_upper.values,
        "bonus_amount_policy": final_amount.values,
        "amount_basis": final_basis,
        "policy_weights": [f"business={weight_business:.2f}/profitability={weight_profitability:.2f}/"
                            f"segment={weight_segment:.2f}"] * len(new_df),})



GAME_TO_PRODUCT = {
    "Aviator": {"label": "Free Flights - Aviator", "floor": 2.0, "mt_rate": 63.87556},
    # next game goes here when it's ready -- it will likely share this rate (same market, same currency),
    # but isn't forced to: a future game can carry its own mt_rate here if it ever needs one.
}

FLIGHT_OVERPAY_THRESHOLD = 0.5


COUNT_FLOOR = 1
COUNT_CEILING = 25
SIZE_CEILING = 1000.0

FREE_FLIGHT_MIN_THEORETICAL_GGR = 2.0
FREE_FLIGHT_MIN_STAKE_MT = 2.0



AVIATOR_FLIGHT_TEMPLATES = [
    (2.0, 4.0), (2.0, 10.0), (3.0, 8.0), (2.0, 15.0), (4.0, 9.0), (2.0, 19.0), (4.0, 13.0),
    (5.0, 12.0), (7.0, 9.0), (8.0, 9.0), (12.0, 7.0), (6.0, 16.0), (10.0, 11.0), (5.0, 25.0),
    (12.0, 12.0), (50.0, 2.0), (50.0, 5.0), (100.0, 1.0), (100.0, 3.0), (250.0, 4.0),
    (500.0, 4.0), (1000.0, 2.0),
]


def apply_aviator_flight_structure(general_out, stake_lookup, game_to_product=GAME_TO_PRODUCT,
                                    overpay_threshold=FLIGHT_OVERPAY_THRESHOLD,
                                    min_theoretical_ggr=FREE_FLIGHT_MIN_THEORETICAL_GGR,
                                    min_stake_mt=FREE_FLIGHT_MIN_STAKE_MT,
                                    flight_templates=AVIATOR_FLIGHT_TEMPLATES):
    
    out = general_out.copy()
    out["bonus_product_type"] = None
    out["bonus_flight_count"] = np.nan
    out["bonus_flight_size"] = np.nan
    out["bonus_structured_total"] = np.nan
    out["bonus_amount_policy_mt"] = np.nan
    out["bonus_structure_note"] = None

    favorite_game_name = out["Customer_ID"].map(stake_lookup["favorite_game_name"])
    avg_stake_per_bet = out["Customer_ID"].map(stake_lookup["avg_stake_per_bet"])

    
    default_game = next(iter(game_to_product))
    row_floor = pd.Series(game_to_product[default_game]["floor"], index=out.index)
    row_label = pd.Series(game_to_product[default_game]["label"], index=out.index)
    row_rate = pd.Series(game_to_product[default_game].get("mt_rate", 1.0), index=out.index)

    gate = out["deserves_bonus"].fillna(False)
    has_stake = avg_stake_per_bet.notna() & (avg_stake_per_bet > 0)

    
    stake_mt = avg_stake_per_bet * row_rate
    approved_mt = out["bonus_amount_policy"] * row_rate

   
    fails_ggr = out["theoretical_ggr"] < min_theoretical_ggr
    fails_stake_low = stake_mt < min_stake_mt
    meets_ff_criteria = ~(fails_ggr | fails_stake_low)

    eligible = gate & meets_ff_criteria

  
    effective_stake_mt = stake_mt.where(has_stake, 0.0)

    templates = np.array(flight_templates, dtype=float)      # shape (22, 2): [flight_value, count]
    template_totals = templates[:, 0] * templates[:, 1]       # shape (22,)

    approved_arr = approved_mt.to_numpy()[:, None]             # shape (n, 1)
    stake_arr = effective_stake_mt.to_numpy()[:, None]         # shape (n, 1)
    value_diff = np.abs(template_totals[None, :] - approved_arr)           # shape (n, 22)
    stake_diff = np.abs(templates[None, :, 0] - stake_arr)                  # shape (n, 22), tiebreaker

   
    combined_key = value_diff * 1e6 + stake_diff
    best_idx = np.argmin(combined_key, axis=1)

    count = pd.Series(templates[best_idx, 1], index=out.index)
    size_final = pd.Series(templates[best_idx, 0], index=out.index)
    delivered = count * size_final

    deviates_notably = eligible & ((delivered > approved_mt * (1 + overpay_threshold)) |
                                    (delivered < approved_mt * (1 - overpay_threshold)))

    structure_this = eligible

    out.loc[structure_this, "bonus_product_type"] = row_label[structure_this]
    out.loc[structure_this, "bonus_flight_count"] = count[structure_this]
    out.loc[structure_this, "bonus_flight_size"] = size_final[structure_this]
    out.loc[structure_this, "bonus_structured_total"] = delivered[structure_this]
    out.loc[structure_this, "bonus_amount_policy_mt"] = approved_mt[structure_this]
    out.loc[deviates_notably, "bonus_structure_note"] = (
        f"Structured, but the closest available real template still differs from the approved amount by "
        f"more than {overpay_threshold*100:.0f}% -- the real Free Flight catalogue has uneven coverage, "
        f"and this customer's approved amount falls in a gap between templates.")

    _size_floor_note = game_to_product[default_game]["floor"]
    no_stake_case = eligible & ~has_stake
    out.loc[no_stake_case, "bonus_structure_note"] = (
        "Structured to the closest-value real template by approved amount alone -- no real stake history "
        "was available to use as a tiebreaker.")

    
    not_ff_eligible = gate & ~meets_ff_criteria
    reason_parts = pd.Series([[] for _ in range(len(out))], index=out.index)
    for mask, text in [
        (fails_ggr, f"theoretical GGR below ${min_theoretical_ggr:.0f}"),
        (fails_stake_low, f"average stake below {min_stake_mt:.0f} MT"),
    ]:
        idx = out.index[mask & not_ff_eligible]
        for i in idx:
            reason_parts.loc[i].append(text)
    combined_reasons = reason_parts.loc[not_ff_eligible].apply(lambda parts: ", ".join(parts))
    out.loc[not_ff_eligible, "bonus_structure_note"] = (
        "Qualifies for a bonus, but does not meet the Free Flight eligibility criteria (" +
        combined_reasons + ") -- not structured into flights; deserves_bonus and bonus_amount_policy "
        "are unaffected.")

    n_structured = int(structure_this.sum())
    n_no_stake = int(no_stake_case.sum())
    n_overpay = int(deviates_notably.sum())
    n_not_ff_eligible = int(not_ff_eligible.sum())
    by_game = out.loc[structure_this, "bonus_product_type"].value_counts().to_dict()
    print(f"Flight structuring: {n_structured:,} customers structured ({by_game}). Of qualifying "
          f"customers (worthiness >= 70), {n_not_ff_eligible:,} don't meet the Free Flight eligibility "
          f"criteria (theoretical GGR >= ${min_theoretical_ggr:.0f}, stake >= {min_stake_mt:.0f} MT) and "
          f"keep deserves_bonus=True without a flight structure. Of those structured, {n_no_stake:,} had "
          f"no real stake to use as a tiebreaker and {n_overpay:,} fall in a gap in the real template "
          f"catalogue, deviating from their approved amount by more than {overpay_threshold*100:.0f}% "
          f"(flagged in bonus_structure_note, not excluded).")
    return out


def batch_score(new_df, artifact):
    
    row = new_df.copy()
    for c, enc in artifact["cat_encoders"].items():
        if c not in row.columns:
            continue
        if enc["method"] == "onehot":
            unseen = set(row[c].dropna().unique()) - set(enc["levels"])
            if unseen:
                print(f"\u26a0\ufe0f  '{c}' has {len(unseen)} value(s) not seen during training "
                      f"(e.g. {list(unseen)[:3]}) -- encoded as all-zero (reference category), the")
                print(f"correct, safe behaviour for an unseen category, not an error.")
            for level in enc["levels"]:
                row[f"{c}_{level}"] = (row[c] == level).astype(float)
            row = row.drop(columns=[c])
        elif enc["method"] == "frequency":
            collapsed = row[c].where(row[c].isin(enc["freq_map"].keys()), "Other")
            row[f"{c}_freq"] = collapsed.map(enc["freq_map"]).fillna(0.0)
            row = row.drop(columns=[c])

    for c in artifact["feat_cols_post_encoding"]:
        if c not in row.columns:
            row[c] = np.nan
    row = row[artifact["feat_cols_post_encoding"]]

    row_imp = row.copy()
    for c, spec in artifact["impute_fill_values"].items():
        if c in row_imp.columns:
            row_imp[c] = row_imp[c].fillna(spec["value"])
    for c in artifact["log_transformed_cols"]:
        row_imp[c] = np.log1p(row_imp[c])
    for c in artifact["signed_log_transformed_cols"]:
        row_imp[c] = np.sign(row_imp[c]) * np.log1p(np.abs(row_imp[c]))

    row_scaled = pd.DataFrame(artifact["scaler"].transform(row_imp), columns=artifact["feat_cols_post_encoding"],
                               index=row.index)
    row_final = row_scaled[artifact["feat_cols_final"]]

    primary_model = artifact["primary_model"]
    centroids = np.asarray(primary_model["centroids"])
    dists = np.linalg.norm(row_final.values[:, None, :] - centroids[None, :, :], axis=2)
    pred_labels = np.argmin(dists, axis=1)
    sorted_d = np.sort(dists, axis=1)
    confidence = np.where(sorted_d.shape[1] > 1, 1 - sorted_d[:, 0] / (sorted_d[:, 1] + 1e-9), 1.0)
    personas = np.array([artifact["persona_names"].get(int(l), f"Segment {l}") for l in pred_labels])

    out = pd.DataFrame({"segment_id": pred_labels, "persona": personas,
                         "assignment_confidence": np.round(confidence, 4)}, index=row.index)

    sub_cluster_models = artifact.get("sub_cluster_models") or {}
    behavioral_scaler = artifact.get("behavioral_scaler")
    if sub_cluster_models and behavioral_scaler is not None:
        print(f"   Sub-cluster models available for parent segment(s) "
              f"{[int(c) for c in sub_cluster_models.keys()]} -- "
              f"predicting sub-segments too.")
        beh = pd.DataFrame(index=new_df.index)
        bet_count = new_df.get("Bet_Count", pd.Series(0, index=new_df.index)).replace(0, np.nan)
        active_days = new_df.get("active_days", pd.Series(0, index=new_df.index)).replace(0, np.nan)
        beh["bonus_per_bet"] = new_df.get("Bonus_Stake", 0) / bet_count
        beh["stake_per_active_day"] = new_df.get("Cash_Stake_Casino", 0) / active_days
        beh["avg_stake_per_bet"] = new_df.get("Cash_Stake_Casino", 0) / bet_count
        beh["deposits_per_active_day"] = new_df.get("Deposit_Count", 0) / active_days
        for c, spec in artifact["behavioral_impute_fill_values"].items():
            beh[c] = beh[c].fillna(spec["value"])
        for c in artifact["behavioral_log_transformed"]:
            beh[c] = np.log1p(beh[c])
        for c in artifact["behavioral_signed_log_transformed"]:
            beh[c] = np.sign(beh[c]) * np.log1p(np.abs(beh[c]))
        BEHAVIORAL_SHAPE_FEATURES = ["bonus_per_bet", "stake_per_active_day", "avg_stake_per_bet",
                                      "deposits_per_active_day"]
        beh_scaled = pd.DataFrame(behavioral_scaler.transform(beh), columns=BEHAVIORAL_SHAPE_FEATURES,
                                   index=row.index)
        expanded = pd.concat([row_final, beh_scaled], axis=1)[artifact["feat_final_expanded"]]

        sub_segment_id = np.full(len(row), None, dtype=object)
        sub_persona = np.full(len(row), None, dtype=object)
        
        sub_segment_confidence = np.full(len(row), np.nan, dtype=float)
        for parent_c, sub_model in sub_cluster_models.items():
            parent_mask = (pred_labels == int(parent_c))
            if parent_mask.sum() == 0:
                continue
            sub_centroids = np.asarray(sub_model["sub_centroids"])
            sub_dists = np.linalg.norm(expanded.values[parent_mask][:, None, :] - sub_centroids[None, :, :], axis=2)
            sub_idx = np.argmin(sub_dists, axis=1)
            if sub_dists.shape[1] > 1:
                sorted_sub_d = np.sort(sub_dists, axis=1)
                sub_conf_here = 1 - sorted_sub_d[:, 0] / (sorted_sub_d[:, 1] + 1e-9)
            else:
                sub_conf_here = np.ones(sub_dists.shape[0])
            labels_here = np.array([f"{parent_c}{chr(97 + i)}" for i in sub_idx])
            names_here = np.array([sub_model["sub_persona_names"].get(i, f"Sub-segment {parent_c}{chr(97+i)}")
                                    for i in sub_idx])
            sub_segment_id[parent_mask] = labels_here
            sub_persona[parent_mask] = names_here
            sub_segment_confidence[parent_mask] = np.round(sub_conf_here, 4)
        out["sub_segment_id"] = sub_segment_id
        out["sub_persona"] = sub_persona
        out["sub_segment_confidence"] = sub_segment_confidence
    else:
        print(f"No sub_cluster_models in this artefact (produced before sub-cluster persistence existed, or")
        print(f"no cluster genuinely sub-divided at training time) -- top-level segments only.")

    
    if "Customer_ID" in new_df.columns:
        worthiness_features = compute_bonus_worthiness_features(new_df)
        pca_weights = run_diagnostic_pca(worthiness_features)
        worthiness = score_bonus_worthiness(worthiness_features, pca_derived_weights=pca_weights)
        worthiness = worthiness.set_index(new_df.index)
        for c in ["value_score", "recency_score", "responsiveness_score", "engagement_score",
                  "bonus_worthiness_score"]:
            out[c] = worthiness[c]
        
        out["worthiness_weights"] = "/".join(f"{g}={v*100:.1f}%" for g, v in pca_weights.items())

    return out


# =====================================================================
# STEP 4: Drift check against the saved training baseline
# =====================================================================
def check_drift_against_baseline(new_raw_df, artifact, baseline, preprocess_fn):
    banner("STEP 4: Drift Check -- September vs Saved Training Baseline")
    primary_model = artifact["primary_model"]
    centroids = np.asarray(primary_model["centroids"])
    k = baseline["k"]
    print(f"Baseline: {baseline['n_training_customers']:,} training customers as of "
          f"{baseline.get('training_snapshot_date', 'unknown date')}, k={k}")
    print(f"New data: {len(new_raw_df):,} customers")

    X_new = preprocess_fn(new_raw_df, artifact)
    scored = batch_score(new_raw_df, artifact)
    new_labels = scored["segment_id"].values

    subbanner("Type 1: Feature Drift (PSI) vs saved baseline")
    PSI_STABLE, PSI_WARNING = 0.10, 0.25
    psi_results = {}
    for c in baseline["feat_cols_final"]:
        edges = np.asarray(baseline["psi_bins"][c]["edges"], dtype=float)
        ref_pct = np.clip(np.asarray(baseline["psi_bins"][c]["ref_pct"], dtype=float), 1e-6, None)
        cur_counts, _ = np.histogram(X_new[c].values, bins=edges)
        cur_pct = np.clip(cur_counts / max(len(X_new), 1), 1e-6, None)
        val = float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
        psi_results[c] = val
        flag = "STABLE" if val < PSI_STABLE else ("WARNING" if val < PSI_WARNING else "ACTION REQUIRED")
        tag = "[PROTECTED]" if c in BUSINESS_PROTECTED_FEATURES else ""
        print(f"{c:<32} PSI={val:.4f}  {flag}{tag}")

    subbanner("Type 2: Segment Size Drift vs saved baseline")
    SIZE_STABLE, SIZE_INVESTIGATE = 15.0, 30.0
    new_share = pd.Series(new_labels).value_counts(normalize=True).sort_index()
    for c in range(k):
        base_share = baseline["segment_size_shares"].get(str(c))
        if base_share is None or c not in new_share.index:
            print(f"   Cluster {c}: not present in baseline or new data \u2014 skipped")
            continue
        shift = abs(new_share[c] - base_share) / base_share * 100
        flag = "STABLE" if shift < SIZE_STABLE else ("INVESTIGATE" if shift < SIZE_INVESTIGATE else "RE-RUN SWEEP")
        print(f"Cluster {c}: baseline={base_share*100:.1f}%  new={new_share[c]*100:.1f}%  shift={shift:.2f}%  {flag}")

    subbanner("Type 2b: Sub-Segment Size Drift vs saved baseline")
    
    if "sub_segment_id" in scored.columns and baseline.get("sub_segment_size_shares"):
        new_sub_share = scored["sub_segment_id"].value_counts(normalize=True)
        for sub_label, base_share in baseline["sub_segment_size_shares"].items():
            new_pct = new_sub_share.get(sub_label)
            if new_pct is None:
                print(f"{sub_label}: not present in new data \u2014 skipped")
                continue
            shift = abs(new_pct - base_share) / base_share * 100
            flag = "STABLE" if shift < SIZE_STABLE else ("INVESTIGATE" if shift < SIZE_INVESTIGATE else "RE-RUN SWEEP")
            print(f"{sub_label}: baseline={base_share*100:.2f}%  new={new_pct*100:.2f}%  shift={shift:.2f}%  {flag}")
    else:
        print(f"No sub-segment predictions in this artefact, or no sub_segment_size_shares in this baseline")
        print(f"(produced before this check existed, or no cluster genuinely sub-divided) \u2014 skipped.")

    subbanner("Type 5: Calibration Drift vs saved baseline")
    proxy_feature = baseline.get("calibration_proxy_feature")
    if proxy_feature and proxy_feature in new_raw_df.columns and baseline.get("calibration_baseline_rates"):
        CAL_STABLE, CAL_INVESTIGATE = 10.0, 25.0
        for c in range(k):
            base_rate = baseline["calibration_baseline_rates"].get(str(c))
            mask = new_labels == c
            if base_rate is None or mask.sum() == 0:
                continue
            new_rate = float(new_raw_df.loc[mask, proxy_feature].mean())
            shift = abs(new_rate - base_rate) / max(abs(base_rate), 1e-6) * 100
            flag = "STABLE" if shift < CAL_STABLE else ("INVESTIGATE" if shift < CAL_INVESTIGATE else "RECALIBRATE")
            print(f"Cluster {c}: baseline={base_rate:.3f}  new={new_rate:.3f}  shift={shift:.2f}%  {flag}")
    else:
        print(f"No usable calibration baseline/proxy in the new data \u2014 skipped.")

    subbanner("Type 6: Centroid-Confidence Monitoring vs saved baseline")
    CENTROID_CONF_WARN = 15.0
    dists_new = np.linalg.norm(X_new[baseline["feat_cols_final"]].values[:, None, :] - centroids[None, :, :], axis=2)
    if baseline.get("centroid_confidence_baseline"):
        for c in range(k):
            base = baseline["centroid_confidence_baseline"].get(str(c))
            mask = new_labels == c
            if base is None or mask.sum() == 0:
                continue
            new_mean = float(dists_new[mask, c].mean())
            shift_pct = (new_mean - base["mean"]) / base["mean"] * 100
            flag = "STABLE" if abs(shift_pct) < CENTROID_CONF_WARN else "EARLY WARNING"
            print(f"Cluster {c}: baseline mean={base['mean']:.4f}  new mean={new_mean:.4f}  "
                  f"shift={shift_pct:+.2f}%  {flag}")

    subbanner("Type 7: Group Distribution Drift vs saved baseline")
    group_col = baseline.get("group_col")
    if group_col and group_col in new_raw_df.columns and baseline.get("group_baseline_counts"):
        base_counts = baseline["group_baseline_counts"]
        new_counts = new_raw_df[group_col].astype(str).value_counts().to_dict()
        levels = sorted(set(base_counts.keys()) | set(new_counts.keys()))
        ct = np.array([[base_counts.get(l, 0) for l in levels], [new_counts.get(l, 0) for l in levels]])
        chi2, p, _, _ = stats.chi2_contingency(ct + 1e-9)
        n = ct.sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else 0.0
        flag = "STABLE" if cramers_v < 0.15 else ("INVESTIGATE" if cramers_v < 0.25 else "RE-EXAMINE COHORT MIX")
        print(f" '{group_col}' distribution: baseline vs new Cramer's V={cramers_v:.4f}  p={p:.4g}  {flag}")
    else:
        print(f"No group baseline/column available \u2014 Type 7 skipped.")

    print(f"\n NOTE: Types 3/4 (centroid displacement + fresh-refit ARI) are intentionally NOT run here --")
    print(f"they require refitting the winning algorithm on new data (_fit_algorithm_generic from the")
    print(f"main pipeline), which this standalone script does not duplicate to keep it lightweight. Run")
    print(f"check_drift_against_baseline() from the main pipeline directly for the complete 7-type check.")

    return {"psi": psi_results, "max_psi": max(psi_results.values()) if psi_results else None}


def _preprocess_for_scoring(new_df, artifact):
   
    row = new_df.copy()
    for c, enc in artifact["cat_encoders"].items():
        if c not in row.columns:
            continue
        if enc["method"] == "onehot":
            for level in enc["levels"]:
                row[f"{c}_{level}"] = (row[c] == level).astype(float)
            row = row.drop(columns=[c])
        elif enc["method"] == "frequency":
            collapsed = row[c].where(row[c].isin(enc["freq_map"].keys()), "Other")
            row[f"{c}_freq"] = collapsed.map(enc["freq_map"]).fillna(0.0)
            row = row.drop(columns=[c])
    for c in artifact["feat_cols_post_encoding"]:
        if c not in row.columns:
            row[c] = np.nan
    row = row[artifact["feat_cols_post_encoding"]]
    row_imp = row.copy()
    for c, spec in artifact["impute_fill_values"].items():
        if c in row_imp.columns:
            row_imp[c] = row_imp[c].fillna(spec["value"])
    for c in artifact["log_transformed_cols"]:
        row_imp[c] = np.log1p(row_imp[c])
    for c in artifact["signed_log_transformed_cols"]:
        row_imp[c] = np.sign(row_imp[c]) * np.log1p(np.abs(row_imp[c]))
    return pd.DataFrame(artifact["scaler"].transform(row_imp), columns=artifact["feat_cols_post_encoding"],
                         index=row.index)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw = load_and_normalize(NEW_DATA_PATH)
    modelling_df, pre_activation_df, snapshot_date = aggregate_to_customer_level(df_raw)

    banner("STEP 3: Loading deployment artefact and batch-scoring")
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(f"deployment_artifacts.joblib not found at {ARTIFACT_PATH} -- run the main "
                                 f"pipeline through Step 12 first to produce it.")
    artifact = joblib.load(ARTIFACT_PATH)
    print(f"Loaded artefact version={artifact.get('version')}, algorithm="
          f"'{artifact['primary_model']['algorithm']}', k={artifact['primary_model']['k']}")

    results = batch_score(modelling_df, artifact)
    print(f"\n Scored {len(results):,} customers.")
    seg_counts = results["segment_id"].value_counts().sort_index()
    for seg, count in seg_counts.items():
        persona = artifact["persona_names"].get(seg, f"Segment {seg}")
        print(f"Segment {seg} ('{persona}'): {count:,} ({count / len(results) * 100:.2f}%)")

    if "sub_segment_id" in results.columns:
        subbanner("Sub-segment distribution")
        sub_counts = results["sub_segment_id"].value_counts().sort_index()
        for sub_seg, count in sub_counts.items():
            sub_persona = results.loc[results["sub_segment_id"] == sub_seg, "sub_persona"].iloc[0]
            print(f"{sub_seg} ('{sub_persona}'): {count:,} ({count / len(results) * 100:.2f}%)")

    conf_desc = results["assignment_confidence"].describe()
    print(f"\n Assignment confidence -- mean={conf_desc['mean']:.4f}  median={conf_desc['50%']:.4f}  "
          f"min={conf_desc['min']:.4f}  max={conf_desc['max']:.4f}")
    LOW_CONF_THRESHOLD = 0.1
    n_low_conf = (results["assignment_confidence"] < LOW_CONF_THRESHOLD).sum()
    print(f"Customers with confidence < {LOW_CONF_THRESHOLD}: {n_low_conf:,} "
          f"({n_low_conf / len(results) * 100:.2f}%) -- sit near a segment boundary, worth a closer look")
    print(f"if this fraction is materially higher than what training-time validation showed.")

    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        drift_result = check_drift_against_baseline(modelling_df, artifact, baseline, _preprocess_for_scoring)
    else:
        print(f"\n \u26a0\ufe0f  training_baseline.json not found at {BASELINE_PATH} -- skipping the drift check.")
        print(f"Run the main pipeline through Step 12 to produce it, then re-run this script for a full")
        print(f"drift-validated report.")
        drift_result = None

    banner("VALIDATION SUMMARY")
    print(f"Extract: {os.path.basename(NEW_DATA_PATH)}  |  snapshot_date: {snapshot_date}")
    print(f"Total customers in extract: {len(modelling_df) + len(pre_activation_df):,}")
    print(f"Zero-cash-funded cohort (excluded from segmentation): {len(pre_activation_df):,}")
    print(f"Scored: {len(results):,}")
    print(f"Segment distribution: " +
          ", ".join(f"{seg}={count:,} ({count/len(results)*100:.1f}%)" for seg, count in seg_counts.items()))
    if drift_result is not None and drift_result["max_psi"] is not None:
        print(f"Max feature PSI vs training baseline: {drift_result['max_psi']:.4f} "
              f"({'STABLE' if drift_result['max_psi'] < 0.10 else 'WARNING/ACTION REQUIRED -- see Type 1 above'})")

   
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    subbanner("Output: segment_assignments with bonus_amount_policy (full population)")
    general_cols = [c for c in results.columns if c not in ("recommended_bonus_amount", "amount_basis")]
    general_out = modelling_df[["Customer_ID"]].join(results[general_cols])

    results_with_id_for_policy = results.copy()
    results_with_id_for_policy["Customer_ID"] = modelling_df["Customer_ID"].values
    policy_input = modelling_df[["Customer_ID", "Total_Deposit_Amount", "Cash_Stake_Casino",
                                  NGR_RECENT_COL, "tenure_days"]].copy()
    policy_amounts = combine_bonus_rules(policy_input, results_with_id_for_policy)
    general_out = general_out.merge(policy_amounts, on="Customer_ID", how="left")

    
    spot_check_cols = modelling_df.set_index("Customer_ID")[[NGR_RECENT_COL, "avg_stake_per_bet", "theoretical_ggr"]] \
        .rename(columns={NGR_RECENT_COL: "customer_ngr", "avg_stake_per_bet": "customer_avg_stake"})
    general_out = general_out.merge(spot_check_cols, on="Customer_ID", how="left")

    
    general_out["bonus_amount_policy_uncapped"] = general_out["bonus_amount_policy"]
    safe_ggr_for_bonus = general_out["theoretical_ggr"].clip(lower=0)
    new_bonus = safe_ggr_for_bonus * 0.01
    general_out["bonus_ggr_capped"] = new_bonus < general_out["bonus_amount_policy_uncapped"]
    general_out["bonus_amount_policy"] = new_bonus
    deserves = general_out["deserves_bonus"].fillna(False)
    pool = general_out["bonus_amount_policy"].where(deserves).sum()
    print(f"Bonus amount now set directly as 1% of theoretical GGR (no worthiness-tier ceiling, no "
          f"stake brackets, no total-pool target). Real total pool: ${pool:,.2f}.")

    safe_ngr = general_out["customer_ngr"].where(general_out["customer_ngr"] > 0)
    safe_stake = general_out["customer_avg_stake"].where(general_out["customer_avg_stake"] > 0)
    general_out["bonus_to_ngr_ratio"] = np.round(general_out["bonus_amount_policy"] / safe_ngr, 3)
    general_out["bonus_to_stake_ratio"] = np.round(general_out["bonus_amount_policy"] / safe_stake, 3)

    
    stake_lookup = modelling_df.set_index("Customer_ID")[["favorite_game_name", "avg_stake_per_bet"]] \
        if "favorite_game_name" in modelling_df.columns else \
        modelling_df.set_index("Customer_ID")[["avg_stake_per_bet"]].assign(favorite_game_name=None)
    general_out = apply_aviator_flight_structure(general_out, stake_lookup)

  
    zero_cash_reason = ("Zero-cash-funded cohort: no cash stake recorded on casino or sports in the measured "
                        "period (total_stake_verified = 0), so the cash-stake-based features segments are built "
                        "on (return rate, casino affinity, bonus dependency) are undefined. Not assigned to any "
                        "segment or sub-segment, and therefore not part of the bonus allocation stream.")
    excluded_out = pre_activation_df[["Customer_ID"]].copy()
    excluded_out["deserves_bonus"] = False
    excluded_out["amount_basis"] = ("No bonus - not part of any segment (zero-cash-funded cohort, "
                                    "excluded from segmentation)")
    excluded_out["segment_exclusion_reason"] = zero_cash_reason
    general_out["segment_exclusion_reason"] = None
    
    excluded_out["is_excluded"] = True
    excluded_out["exclusion_reason_code"] = "ZERO_CASH_FUNDED"
    general_out["is_excluded"] = False
    general_out["exclusion_reason_code"] = None
    general_out = pd.concat([general_out, excluded_out], ignore_index=True, sort=False)
    
    general_out["segment_id"] = general_out["segment_id"].astype("Int64")
    # Keep the file sorted by Customer_ID
    general_out = general_out.sort_values("Customer_ID", kind="stable").reset_index(drop=True)
    print(f"{len(excluded_out):,} zero-cash-funded cohort customers appended -- in no segment or sub-segment, "
          f"so outside the bonus allocation stream.")

    general_path = os.path.join(OUTPUT_DIR, f"segment_assignments_{timestamp}.csv")
    general_out.to_csv(general_path, index=False)
    print(f"{len(general_out):,} customers (the full {os.path.basename(NEW_DATA_PATH)} population).")
    print(f"Written to: {general_path}")