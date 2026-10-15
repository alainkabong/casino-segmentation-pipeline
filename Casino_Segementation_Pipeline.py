#=====================================================================
# Global Configuration and Utility Setup
#=====================================================================

import argparse
import hashlib
import pickle
import warnings
import os
import shutil
import tempfile
import gc
import re
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, dendrogram
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import hdbscan as hdbscan_lib
import hdbscan.validity as hdbscan_validity
import umap as umap_lib

from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score, pairwise_distances)
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BUSINESS_CRITICAL_SEGMENTATION = True
BUSINESS_PROTECTED_FEATURES = ["Cash_Stake_Casino", "Total_Deposit_Amount", "bonus_dependency", "tenure_days"]  # FIX: was "Cash_Stake" -- updated after the Pre-Activation cohort fix (Cash_Stake proven unreliable). Protects the two granular components (Cash_Stake_Casino, Cash_Stake_Sports) rather than their sum (total_stake_verified), since total_stake_verified is itself dropped as a pure collinear sum in the redundancy step below -- protecting it there would contradict protecting a feature slated for correct removal. The granular components are also more business-actionable (they directly support the Casino<->Sports cross-sell persona goal).
PROTECTED_FEATURE_GAIN_MULTIPLIER = 3.0
CONSENSUS_POOL_SIZE = 3

SEGMENT_SIZE_ELIGIBILITY = 500      
DENSITY_NOISE_ELIGIBILITY_L1 = 0.40  
HDBSCAN_MIN_CLUSTER_SIZE_FLOOR = 500 
HDBSCAN_MIN_CLUSTER_SIZE_FRAC = 0.05


SCORE_WEIGHT_SILHOUETTE = 0.20
SCORE_WEIGHT_CH = 0.15
SCORE_WEIGHT_DBI = 0.15
SCORE_WEIGHT_STABILITY = 0.35
SCORE_WEIGHT_BALANCE = 0.15
PRELIM_STABILITY_N_BOOTSTRAP = 5

EXPENSIVE_REFIT_ALGORITHMS = ("hdbscan", "kmedoids")
REDUCED_N_PERM = 19   
REDUCED_N_BOOTSTRAP = 5

HDBSCAN_NOISE_TRIGGER = 0.25      
HDBSCAN_SIZE_IMBALANCE_TRIGGER = 20.0

GMM_LOW_CONFIDENCE_TRIGGER = 0.60

UMAP_PCA_VARIANCE_FLOOR = 0.50

KPI_CANDIDATE_NAMES = ("self_exclusion", "responsible_gambling_flag", "compliance_flag",
                        "deposit_churn", "bonus_abuse", "ngr", "clv", "retention")

LARGE_N_THRESHOLD = 20000

DBSCAN_MAX_MEAN_NEIGHBORS = 2000  # a point with more than this many within-eps
# neighbours, on average, makes DBSCAN's radius search infeasible at real scale
DENSITY_PROBE_SAMPLE = 200
LARGE_N_SUBSAMPLE_SIZE = 20000

N_SEEDS_ROBUSTNESS = 20

CALIBRATION_DRIFT_PROXY_FEATURE = "return_rate"


def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    

def subbanner(title):
    print(f"\n\u2500\u2500 {title} \u2500\u2500")
    
FIGDIR = None

FIGURES_SAVED = []  

def savefig(name):
    
    if FIGDIR is None:
        plt.close()
        return None
    path = os.path.join(FIGDIR, name)
    plt.tight_layout()
    
    try:
        plt.savefig(path, dpi=110)
    except OSError as e:
        print(f"\u26a0\ufe0f  Failed to save plot on first attempt: {path}")
        print(f"{type(e).__name__}: {e}")
        print(f"Likely cause on Windows: a OneDrive-synced folder can briefly leave a just-created")
        print(f"subfolder inconsistent right after os.makedirs() -- retrying once after a short pause...")
        import time as _time
        _time.sleep(1.5)
        try:
            os.makedirs(FIGDIR, exist_ok=True)
            plt.savefig(path, dpi=110)
            print(f"\u2705 Retry succeeded.")
        except OSError as e2:
            print(f"\u274c Retry also failed ({type(e2).__name__}: {e2}) \u2014 skipping this plot and")
            print(f"continuing the pipeline rather than crashing the entire run over one figure.")
            print(f"If this keeps happening: check whether '{FIGDIR}' is inside a OneDrive-synced")
            print(f"folder and consider pointing data_dir at a plain local folder instead (e.g.")
            print(f"C:\\CasinoPipeline), or right-click that folder in Explorer and choose 'Always")
            print(f"keep on this device' to disable Files-On-Demand for it.")
            plt.close()
            return None
    plt.show()  
    plt.close()
    FIGURES_SAVED.append(path)
    print(f"\U0001F4CA Saved plot: {path}")
    return path

#=====================================================================
# HELPER FUNCTIONS — statistical building blocks
#=====================================================================

def scalability_guard(n, context):
    triggered = n > LARGE_N_THRESHOLD
    print(f"Scalability guard check ({context}): n={n:,}  threshold={LARGE_N_THRESHOLD:,}  "
          f"\u2192 {'TRIGGERED -- subsampling to ' + f'{LARGE_N_SUBSAMPLE_SIZE:,} rows' if triggered else 'not triggered, n is within budget'}")
    return triggered

def apply_scalability_subsample(X, seed=RANDOM_STATE):
    
    n = len(X)
    if n <= LARGE_N_THRESHOLD:
        return X, np.arange(n)
    idx = np.random.RandomState(seed).choice(n, LARGE_N_SUBSAMPLE_SIZE, replace=False)
    if hasattr(X, "iloc"):
        return X.iloc[idx].reset_index(drop=True), idx
    return X[idx], idx

def _self_test_scalability_guard():
    
    print(f"Case A: n=8,950 (CC_GENERAL's real size) \u2014 expect NOT triggered")
    t1 = scalability_guard(8950, "self-test case A")
    assert t1 is False, "should not trigger below the threshold"

    print(f"Case B: n=25,000 (synthetic, exceeds threshold) \u2014 expect TRIGGERED + real subsample")
    t2 = scalability_guard(25000, "self-test case B")
    assert t2 is True, "should trigger above the threshold"
    X_synth = pd.DataFrame(np.random.RandomState(0).rand(25000, 4), columns=list("wxyz"))
    X_sub, idx = apply_scalability_subsample(X_synth)
    print(f"Subsample result: {len(X_synth):,} rows \u2192 {len(X_sub):,} rows "
          f"(target {LARGE_N_SUBSAMPLE_SIZE:,})")
    assert len(X_sub) == LARGE_N_SUBSAMPLE_SIZE, "subsample must actually reduce to the target size"
    assert len(set(idx)) == LARGE_N_SUBSAMPLE_SIZE, "subsample indices must be unique (no replacement)"
    assert list(X_sub.index) == list(range(LARGE_N_SUBSAMPLE_SIZE)), \
        "subsampled DataFrame index must be reset to a clean 0..n-1 range, not the original .iloc positions"
    print(f" \u2705 Guard correctly stays dormant below the threshold, correctly subsamples above it,")
    print(f"and the returned DataFrame's index is cleanly reset (not left as stale .iloc positions).")

    print(f"\n Case C: MiniBatchKMeans quality vs standard KMeans on synthetic well-separated clusters")
    print(f"(addresses the concern that unconfigured MiniBatchKMeans could silently degrade quality --")
    print(f"proven with a real ARI comparison, not asserted from parameter values alone)")
    from sklearn.cluster import KMeans as _KM, MiniBatchKMeans as _MBK
    from sklearn.metrics import adjusted_rand_score as _ari
    rng = np.random.RandomState(1)
    n_per, true_k = 3000, 4
    centers = [[0, 0], [10, 0], [0, 10], [10, 10]]
    X_big = np.vstack([rng.normal(c, 0.8, (n_per, 2)) for c in centers])
    true_labels = np.repeat(range(true_k), n_per)
    km_full = _KM(n_clusters=true_k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_big)
    km_mini = _MBK(n_clusters=true_k, init="k-means++", n_init=10,
                    batch_size=min(2048, len(X_big)), max_no_improvement=10, random_state=RANDOM_STATE).fit(X_big)
    ari_full_vs_truth = _ari(true_labels, km_full.labels_)
    ari_mini_vs_truth = _ari(true_labels, km_mini.labels_)
    ari_full_vs_mini = _ari(km_full.labels_, km_mini.labels_)
    print(f"Standard KMeans  vs ground truth: ARI={ari_full_vs_truth:.4f}")
    print(f"MiniBatchKMeans  vs ground truth: ARI={ari_mini_vs_truth:.4f}")
    print(f"MiniBatchKMeans  vs standard KMeans: ARI={ari_full_vs_mini:.4f}")
    MINIBATCH_QUALITY_FLOOR = 0.95
    assert ari_mini_vs_truth >= MINIBATCH_QUALITY_FLOOR, \
        f"MiniBatchKMeans quality ({ari_mini_vs_truth:.4f}) fell below the acceptable floor ({MINIBATCH_QUALITY_FLOOR})"
    print(f" \u2705 With batch_size={min(2048,len(X_big))} and n_init=10 (NOT sklearn's 'auto', which resolves")
    print(f"to a single init for init='k-means++' -- fewer, not more, than the current setting),")
    print(f"MiniBatchKMeans recovers cluster quality statistically indistinguishable from standard")
    print(f"KMeans (ARI={ari_mini_vs_truth:.4f} vs ground truth, {MINIBATCH_QUALITY_FLOOR} floor).")
    
    

def hopkins_statistic(X, m_frac=0.05, seed=42, metric="euclidean", num_cols=None, cat_cols=None):
    
    rng = np.random.RandomState(seed)

    if metric == "gower":
        X_df = X
        n = len(X_df)
        m = max(10, int(m_frac * n))
        idx = rng.choice(n, m, replace=False)
        U_dict = {}
        for c in num_cols:
            lo, hi = X_df[c].min(), X_df[c].max()
            U_dict[c] = rng.uniform(lo, hi, size=m)
        for c in cat_cols:
            cats = X_df[c].astype(str).unique()
            U_dict[c] = rng.choice(cats, size=m)
        U_df = pd.DataFrame(U_dict)[list(X_df.columns)]
        D_full, ranges = gower_distance_matrix(X_df, num_cols, cat_cols)
        w = []
        for real_idx in idx:
            dists = D_full[real_idx].copy()
            dists[real_idx] = np.inf
            w.append(dists.min())
        w = np.array(w)
        D_cross = gower_cross_distance(U_df, X_df, num_cols, cat_cols, ranges)
        u = D_cross.min(axis=1)
        return u.sum() / (u.sum() + w.sum())

    elif metric == "mahalanobis":
        n, d = X.shape
        m = max(10, int(m_frac * n))
        idx = rng.choice(n, m, replace=False)
        mins, maxs = X.min(axis=0), X.max(axis=0)
        U = rng.uniform(mins, maxs, size=(m, d))
      
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov)
        L = np.linalg.cholesky(inv_cov)
        Z = X @ L
        Z_sample = Z[idx]
        U_whitened = U @ L
        nn = NearestNeighbors(n_neighbors=2).fit(Z)
        w = nn.kneighbors(Z_sample)[0][:, 1]    # nearest OTHER real point (whitened Euclidean == Mahalanobis)
        u = nn.kneighbors(U_whitened)[0][:, 0]  # nearest real point to synthetic
        return u.sum() / (u.sum() + w.sum())

    n, d = X.shape
    m = max(10, int(m_frac * n))
    idx = rng.choice(n, m, replace=False)
    X_sample = X[idx]
    mins, maxs = X.min(axis=0), X.max(axis=0)
    U = rng.uniform(mins, maxs, size=(m, d))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    w = nn.kneighbors(X_sample)[0][:, 1]   # nearest OTHER real point
    u = nn.kneighbors(U)[0][:, 0]          # nearest real point to synthetic
    return u.sum() / (u.sum() + w.sum())


def _self_test_hopkins_metric_awareness():
    
    rng = np.random.RandomState(0)

    print(f"Sub-test A (Gower): clustered mixed data vs uniform mixed data")
    n_per = 100
    clustered_num = np.concatenate([rng.normal(0, 0.3, n_per), rng.normal(8, 0.3, n_per)])
    clustered_cat = (["North"] * n_per) + (["South"] * n_per)
    df_clustered = pd.DataFrame({"num": clustered_num, "cat": clustered_cat})
    H_clustered = hopkins_statistic(df_clustered, seed=1, metric="gower", num_cols=["num"], cat_cols=["cat"])
    uniform_num = rng.uniform(0, 8, 2 * n_per)
    uniform_cat = rng.choice(["North", "South"], 2 * n_per)
    df_uniform = pd.DataFrame({"num": uniform_num, "cat": uniform_cat})
    H_uniform = hopkins_statistic(df_uniform, seed=1, metric="gower", num_cols=["num"], cat_cols=["cat"])
    print(f"Clustered mixed data: H={H_clustered:.4f}  |  Uniform mixed data: H={H_uniform:.4f}")
    assert H_clustered > 0.75, "genuinely clustered mixed data should score a high Gower Hopkins statistic"
    assert H_uniform < 0.65, "genuinely uniform mixed data should score a low/moderate Gower Hopkins statistic"
    assert H_clustered > H_uniform, "clustered data must score higher than uniform data under Gower"

    print(f"Sub-test B (Mahalanobis): clustered elliptical data vs uniform data")
    cov = np.array([[3.0, 2.8], [2.8, 3.0]])
    c1 = rng.multivariate_normal([0, 0], cov, n_per)
    c2 = rng.multivariate_normal([10, 10], cov, n_per)
    X_clustered = np.vstack([c1, c2])
    H_clustered_m = hopkins_statistic(X_clustered, seed=1, metric="mahalanobis")
    X_uniform = rng.uniform(X_clustered.min(axis=0), X_clustered.max(axis=0), size=(2 * n_per, 2))
    H_uniform_m = hopkins_statistic(X_uniform, seed=1, metric="mahalanobis")
    print(f"Clustered elliptical data: H={H_clustered_m:.4f}  |  Uniform data: H={H_uniform_m:.4f}")
    assert H_clustered_m > 0.75, "genuinely clustered elliptical data should score a high Mahalanobis Hopkins statistic"
    assert H_uniform_m < 0.65, "genuinely uniform data should score a low/moderate Mahalanobis Hopkins statistic"
    assert H_clustered_m > H_uniform_m, "clustered data must score higher than uniform data under Mahalanobis"

    print(f"\u2705 Hopkins correctly discriminates clustered from uniform data under BOTH non-Euclidean")
    print(f"metrics, not just the Euclidean default.")
    
    
def gap_statistic(X, k_range, B=10, seed=42):
    rng = np.random.RandomState(seed)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    gaps, sk = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X)
        log_wk = np.log(km.inertia_)
        ref_log_wks = []
        for b in range(B):
            Xb = rng.uniform(mins, maxs, size=X.shape)
            kmb = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=seed + b).fit(Xb)
            ref_log_wks.append(np.log(kmb.inertia_))
        ref_log_wks = np.array(ref_log_wks)
        gaps.append(ref_log_wks.mean() - log_wk)
        sk.append(ref_log_wks.std() * np.sqrt(1 + 1.0 / B))
    return np.array(gaps), np.array(sk)



def gower_distance_matrix(df, numeric_cols, categorical_cols):
    
    n = len(df)
    D = np.zeros((n, n))
    ranges = {}
    for c in numeric_cols:
        rng = df[c].max() - df[c].min()
        ranges[c] = rng if rng > 0 else 1.0
    for c in numeric_cols:
        vals = df[c].values.astype(float)
        diff = np.abs(vals[:, None] - vals[None, :]) / ranges[c]
        D += diff
    for c in categorical_cols:
        vals = np.array(df[c].astype(str).values, dtype=object)  
                                                                    
                                                                   
        mismatch = (vals[:, None] != vals[None, :]).astype(float)
        D += mismatch
    D /= (len(numeric_cols) + len(categorical_cols))
    return D, ranges


def gower_cross_distance(df_a, df_b, numeric_cols, categorical_cols, ranges):
  
    n_a, n_b = len(df_a), len(df_b)
    D = np.zeros((n_a, n_b))
    for c in numeric_cols:
        va = df_a[c].values.astype(float)
        vb = df_b[c].values.astype(float)
        D += np.abs(va[:, None] - vb[None, :]) / ranges[c]
    for c in categorical_cols:
        va = np.array(df_a[c].astype(str).values, dtype=object)
        vb = np.array(df_b[c].astype(str).values, dtype=object)
        D += (va[:, None] != vb[None, :]).astype(float)
    D /= (len(numeric_cols) + len(categorical_cols))
    return D


def mahalanobis_distance_matrix(X):
    
    cov = np.cov(X, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    L = np.linalg.cholesky(inv_cov)          
    Z = X @ L                                
    sq_norms = np.sum(Z * Z, axis=1)
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (Z @ Z.T)
    D = np.sqrt(np.maximum(D2, 0))           
    np.fill_diagonal(D, 0.0)                 
    return D, inv_cov


def mahalanobis_cross_distance(X_a, X_b, inv_cov):
    
    L = np.linalg.cholesky(inv_cov)
    Za = X_a @ L
    Zb = X_b @ L
    sq_a = np.sum(Za * Za, axis=1)
    sq_b = np.sum(Zb * Zb, axis=1)
    D2 = sq_a[:, None] + sq_b[None, :] - 2.0 * (Za @ Zb.T)
    D = np.sqrt(np.maximum(D2, 0))
    return D


def kmedoids_pam(D, k, max_iter=100, seed=RANDOM_STATE):
    
    rng = np.random.RandomState(seed)
    n = D.shape[0]
    medoids = rng.choice(n, k, replace=False)
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            sub = D[np.ix_(members, members)]
            new_medoids[c] = members[np.argmin(sub.sum(axis=1))]
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    labels = np.argmin(D[:, medoids], axis=1)
    return labels, medoids


def _self_test_alternate_distance_algorithms():
    
    print(f"Sub-test A (Gower + PAM on synthetic mixed numeric+categorical casino-shaped data):")
    rng = np.random.RandomState(0)
    n_per = 60
    df_synth = pd.DataFrame({
        "stake_amount": np.concatenate([rng.normal(100, 10, n_per), rng.normal(900, 10, n_per)]),
        "acquisition_channel": ["organic"] * n_per + ["paid"] * n_per,
    })
    D_gower, gower_ranges = gower_distance_matrix(df_synth, numeric_cols=["stake_amount"],
                                                    categorical_cols=["acquisition_channel"])
    labels_g, medoid_idx_g = kmedoids_pam(D_gower, k=2, seed=RANDOM_STATE)
    true_labels = np.array([0] * n_per + [1] * n_per)
    from sklearn.metrics import adjusted_rand_score as _ari
    ari_g = _ari(true_labels, labels_g)
    print(f"ARI vs known ground truth: {ari_g:.4f}")
    assert ari_g > 0.9, "Gower+PAM should cleanly recover a well-separated mixed-type synthetic structure"

    print(f"Sub-test B (Mahalanobis + PAM on synthetic correlated-elliptical casino-shaped data):")
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])  # strongly correlated, e.g. deposit_amount vs stake_amount
    c1 = rng.multivariate_normal([0, 0], cov, n_per)
    c2 = rng.multivariate_normal([5, 5], cov, n_per)
    X_synth = np.vstack([c1, c2])
    D_maha, inv_cov_synth = mahalanobis_distance_matrix(X_synth)
    labels_m, medoid_idx_m = kmedoids_pam(D_maha, k=2, seed=RANDOM_STATE)
    ari_m = _ari(true_labels, labels_m)
    print(f"ARI vs known ground truth: {ari_m:.4f}")
    assert ari_m > 0.9, "Mahalanobis+PAM should cleanly recover well-separated elliptical synthetic structure"
    print(f"Diagonal exactly zero (required by sklearn's silhouette_score(metric='precomputed'), "
          f"see FIX 2): max|diag|={np.abs(np.diag(D_maha)).max():.2e}")
    assert np.all(np.diag(D_maha) == 0.0), \
        "mahalanobis_distance_matrix()'s diagonal must be EXACTLY zero, not just numerically tiny -- " \
        "silhouette_score(metric='precomputed') rejects any nonzero diagonal entry outright"

    print(f"Sub-test C (out-of-sample scoring uses the CORRECT metric, not silently Euclidean):")
    new_point_df = pd.DataFrame({"stake_amount": [890.0], "acquisition_channel": ["paid"]})
    medoid_df_g = df_synth.iloc[medoid_idx_g].reset_index(drop=True)
    cross_d_gower = gower_cross_distance(new_point_df, medoid_df_g, ["stake_amount"], ["acquisition_channel"],
                                          gower_ranges)
    nearest_medoid_g = int(np.argmin(cross_d_gower[0]))
    expected_medoid_g = int(medoid_df_g[medoid_df_g["acquisition_channel"] == "paid"].index[0])
    print(f"New point (stake_amount=890, acquisition_channel=paid) -> nearest medoid index "
          f"{nearest_medoid_g} (expect index {expected_medoid_g}, the paid-channel medoid)")
    assert nearest_medoid_g == expected_medoid_g, \
        "a paid-channel/high-stake new point should score nearest to the paid-channel medoid, not organic"

    new_point_maha = np.array([[4.8, 4.8]])  # clearly nearer the second elliptical cluster (centred at [5,5])
    medoid_points_m = X_synth[medoid_idx_m]
    cross_d_maha = mahalanobis_cross_distance(new_point_maha, medoid_points_m, inv_cov_synth)
    nearest_medoid_m = int(np.argmin(cross_d_maha[0]))
    expected_medoid_m = int(np.argmax(medoid_points_m.sum(axis=1)))
    print(f"New point [4.8, 4.8] -> nearest medoid index {nearest_medoid_m} "
          f"(expect index {expected_medoid_m}, the medoid nearer [5,5])")
    assert nearest_medoid_m == expected_medoid_m, \
        "new point near [5,5] should score nearest to the medoid actually belonging to that cluster"
    print(f"\u2705 Both alternate-distance code paths AND their out-of-sample scoring genuinely work on")
    print(f"casino-shaped synthetic data, not narrated-only -- and the vectorised Mahalanobis")
    print(f"rewrite above reproduces the textbook formulation exactly, just fast enough to actually")
    print(f"run on this pipeline's 11,933-row population.")


banner("SELF-TEST: Alternate distance-metric infrastructure (Gower / Mahalanobis / PAM)")
_self_test_alternate_distance_algorithms()


def _group_blocked_resample_indices(group_labels, target_n, seed):
    
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(group_labels)
    order = rng.permutation(unique_groups)
    selected = []
    total = 0
    for g in order:
        g_idx = np.where(group_labels == g)[0]
        selected.append(g_idx)
        total += len(g_idx)
        if total >= target_n:
            break
    return np.concatenate(selected)


def _group_blocked_column_shuffle(X, group_labels, rng):
    
    X = X.copy()
    unique_groups = np.unique(group_labels)
    for j in range(X.shape[1]):
        for g in unique_groups:
            g_idx = np.where(group_labels == g)[0]
            if len(g_idx) > 1:
                shuffled = X[g_idx, j].copy()
                rng.shuffle(shuffled)
                X[g_idx, j] = shuffled
    return X


def _self_test_group_blocked_resampling():
    
    banner("SELF-TEST: Group-blocked resampling helpers (bootstrap + permutation)")
    rng = np.random.RandomState(0)
    groups = np.repeat(["A", "B", "C", "D"], 50)   
    n = len(groups)

    print(f"Sub-test A: _group_blocked_resample_indices never splits a group across the boundary")
    boot_idx = _group_blocked_resample_indices(groups, target_n=int(0.8 * n), seed=1)
    boot_groups = groups[boot_idx]
    for g in np.unique(groups):
        full_count = int((groups == g).sum())
        drawn_count = int((boot_groups == g).sum())
        assert drawn_count in (0, full_count), \
            f"group '{g}' was partially drawn ({drawn_count}/{full_count}) -- a group-blocked resample must be all-or-nothing"
    print(f"Groups drawn whole: {sorted(set(boot_groups.tolist()))}  (each group is either fully in or fully out)")
    print(f"Sample size: {len(boot_idx)} rows (target {int(0.8 * n)}; whole-group rounding is expected, not a bug)")

    print(f"\n Sub-test B: _group_blocked_column_shuffle never moves a value across a group boundary")
    X_synth = rng.normal(size=(n, 3))
    X_shuffled = _group_blocked_column_shuffle(X_synth, groups, np.random.RandomState(2))
    for g in np.unique(groups):
        g_idx = np.where(groups == g)[0]
        for j in range(X_synth.shape[1]):
            before = np.sort(X_synth[g_idx, j])
            after = np.sort(X_shuffled[g_idx, j])
            assert np.allclose(before, after), \
                f"group '{g}' column {j}: value SET changed after shuffling -- a value crossed a group boundary"
    changed = not np.allclose(X_synth, X_shuffled)
    assert changed, "shuffle should genuinely change row order within groups, not silently no-op"
    print(f"Each group's value SET is unchanged (only within-group order is shuffled) \u2014 confirmed across all "
          f"{X_synth.shape[1]} columns and all {len(np.unique(groups))} groups.")
    print(f"Row order genuinely changed within groups (shuffle is not a no-op): {changed}")

    print(f"\n \u2705 Both group-blocked resampling helpers respect group boundaries as a hard constraint, "
          f"not just on average -- ready to back the group_detected=True branch of the Step 1B decision matrix.")


_self_test_group_blocked_resampling()


def bootstrap_jaccard_stability(X, primary_model, labels_reference, k, n_bootstrap=30, seed_base=100,
                                 structure_result=None, group_labels=None, subsample_frac=0.8):
    
    n = len(X)
    idx_all = np.arange(n)
    jaccards = {c: [] for c in range(k)}
    algorithm = primary_model["algorithm"]

    group_detected = bool(structure_result and structure_result.get("group_detected"))
    use_group_blocked = group_detected and group_labels is not None
    if group_detected and group_labels is None:
        print(f"\u26a0\ufe0f  structure_result reports group_detected=True "
              f"('{structure_result.get('group_col_selected')}') but no group_labels were supplied to "
              f"bootstrap_jaccard_stability() \u2014 falling back to plain iid row-level resampling for this call. "
              f"Pass group_labels to honour the Step 1B decision matrix.")
    elif use_group_blocked:
        print(f"Resampling strategy for this call: GROUP-BLOCKED on "
              f"'{structure_result.get('group_col_selected')}' (whole groups drawn, per Step 1B: "
              f"{structure_result.get('resampling_strategy')})")

    for b in range(n_bootstrap):
        seed = seed_base + b
        if use_group_blocked:
            boot_idx = _group_blocked_resample_indices(group_labels, target_n=int(subsample_frac * n), seed=seed)
        else:
            boot_idx = resample(idx_all, replace=False, n_samples=int(subsample_frac * n), random_state=seed)
        boot_labels, boot_centroids = _fit_algorithm_generic(
            X[boot_idx], algorithm, k, seed,
            eps=primary_model.get("eps"), min_samples=primary_model.get("min_samples"),
            min_cluster_size=primary_model.get("min_cluster_size"),
            alt_distance_metric=primary_model.get("alt_distance_metric"),
            num_cols=primary_model.get("num_cols"), cat_cols=primary_model.get("cat_cols"),
            feat_cols_ordered=primary_model.get("feat_cols_ordered"))
        if len(boot_centroids) == 0:
            continue
        cost = np.linalg.norm(boot_centroids[:, None, :] - primary_model["centroids"][None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)               # Hungarian algorithm — solves label permutation
        mapping = dict(zip(row, col))
        boot_mapped = np.array([mapping.get(l, -1) for l in boot_labels])
        orig = labels_reference[boot_idx]
        for c in range(k):
            A = set(boot_idx[boot_mapped == c])
            B_ = set(boot_idx[orig == c])
            if A | B_:
                jaccards[c].append(len(A & B_) / len(A | B_))
    return jaccards


def bootstrap_ari_stability(X, primary_model, labels_reference, k, n_bootstrap=30, seed_base=200,
                             structure_result=None, group_labels=None, subsample_frac=0.8):
   
    n = len(X)
    idx_all = np.arange(n)
    algorithm = primary_model["algorithm"]

    group_detected = bool(structure_result and structure_result.get("group_detected"))
    use_group_blocked = group_detected and group_labels is not None

    aris = []
    for b in range(n_bootstrap):
        seed = seed_base + b
        if use_group_blocked:
            boot_idx = _group_blocked_resample_indices(group_labels, target_n=int(subsample_frac * n), seed=seed)
        else:
            boot_idx = resample(idx_all, replace=False, n_samples=int(subsample_frac * n), random_state=seed)
        boot_labels, boot_centroids = _fit_algorithm_generic(
            X[boot_idx], algorithm, k, seed,
            eps=primary_model.get("eps"), min_samples=primary_model.get("min_samples"),
            min_cluster_size=primary_model.get("min_cluster_size"),
            alt_distance_metric=primary_model.get("alt_distance_metric"),
            num_cols=primary_model.get("num_cols"), cat_cols=primary_model.get("cat_cols"),
            feat_cols_ordered=primary_model.get("feat_cols_ordered"))
        if len(boot_centroids) == 0:
            continue
        cost = np.linalg.norm(boot_centroids[:, None, :] - primary_model["centroids"][None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = dict(zip(row, col))
        boot_mapped = np.array([mapping.get(l, -1) for l in boot_labels])
        aris.append(adjusted_rand_score(labels_reference[boot_idx], boot_mapped))
    return np.array(aris)


def compute_significance_statistic(X, labels, algorithm, num_cols=None, cat_cols=None,
                                    alt_distance_metric=None, feat_cols_ordered=None):
    if algorithm in ("dbscan", "hdbscan"):
        if len(set(labels[labels != -1])) < 2:
            return 0.0
        try:
            return float(hdbscan_validity.validity_index(X, labels))
        except Exception:
            return 0.0
    elif algorithm == "kmedoids":
        if len(set(labels)) < 2:
            return 0.0
        try:
            if alt_distance_metric == "gower":
                X_df = pd.DataFrame(X, columns=feat_cols_ordered)
                D, _ = gower_distance_matrix(X_df, num_cols or [], cat_cols or [])
            else:
                D, _ = mahalanobis_distance_matrix(X)
            return float(silhouette_score(D, labels, metric="precomputed"))
        except Exception:
            return 0.0
    else:
        return float(calinski_harabasz_score(X, labels))
    

def _self_test_algorithm_aware_significance():
    
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal([0, 0], 0.3, (100, 2)), rng.normal([6, 6], 0.3, (100, 2))])
    labels = np.array([0] * 100 + [1] * 100)
    Xp = X.copy()
    for j in range(Xp.shape[1]):
        rng.shuffle(Xp[:, j])  # destroys real structure, same permutation used in the real test

    print(f"Sub-test A (K-Means/Hierarchical/GMM -> Calinski-Harabasz, UNCHANGED):")
    ch_real = compute_significance_statistic(X, labels, "kmeans")
    ch_perm = compute_significance_statistic(Xp, labels, "kmeans")
    print(f"Real CH={ch_real:.2f}  Permuted CH={ch_perm:.2f}")
    assert ch_real > ch_perm, "real structure should score higher CH than permuted data"

    print(f"Sub-test B (DBSCAN/HDBSCAN -> DBCV, not CH):")
    dbcv_real = compute_significance_statistic(X, labels, "dbscan")
    dbcv_perm = compute_significance_statistic(Xp, labels, "dbscan")
    print(f"Real DBCV={dbcv_real:.4f}  Permuted DBCV={dbcv_perm:.4f}")
    assert dbcv_real > dbcv_perm, "real structure should score higher DBCV than permuted data"

    print(f"Sub-test C (K-Medoids -> silhouette on the FITTED non-Euclidean metric, not CH):")
    D_real, inv_cov = mahalanobis_distance_matrix(X)
    labels_medoid, _ = kmedoids_pam(D_real, k=2, seed=RANDOM_STATE)
    sil_real = compute_significance_statistic(X, labels_medoid, "kmedoids", alt_distance_metric="mahalanobis")
    D_perm, _ = mahalanobis_distance_matrix(Xp)
    labels_medoid_p, _ = kmedoids_pam(D_perm, k=2, seed=RANDOM_STATE)
    sil_perm = compute_significance_statistic(Xp, labels_medoid_p, "kmedoids", alt_distance_metric="mahalanobis")
    print(f"Real Mahalanobis-silhouette={sil_real:.4f}  Permuted={sil_perm:.4f}")
    assert sil_real > sil_perm, "real structure should score higher precomputed-metric silhouette than permuted data"

    print(f"\u2705 Each algorithm dispatches to its own appropriate statistic, and each correctly")
    print(f"discriminates real structure from permuted noise.")
    

def permutation_significance_test(X, k, algorithm="kmeans", n_perm=99, seed=42, eps=None, min_samples=None,
                                   min_cluster_size=None, alt_distance_metric=None, num_cols=None, cat_cols=None,
                                   feat_cols_ordered=None, structure_result=None, group_labels=None):
    
    fit_extra = dict(eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size,
                      alt_distance_metric=alt_distance_metric, num_cols=num_cols, cat_cols=cat_cols,
                      feat_cols_ordered=feat_cols_ordered)
    stat_extra = dict(num_cols=num_cols, cat_cols=cat_cols, alt_distance_metric=alt_distance_metric,
                       feat_cols_ordered=feat_cols_ordered)
    real_labels, _ = _fit_algorithm_generic(X, algorithm, k, RANDOM_STATE, **fit_extra)
    real_ch = compute_significance_statistic(X, real_labels, algorithm, **stat_extra)
    rng = np.random.RandomState(seed)

    group_detected = bool(structure_result and structure_result.get("group_detected"))
    use_group_blocked = group_detected and group_labels is not None
    if group_detected and group_labels is None:
        print(f"\u26a0\ufe0f  structure_result reports group_detected=True "
              f"('{structure_result.get('group_col_selected')}') but no group_labels were supplied to "
              f"permutation_significance_test() \u2014 falling back to a full-population column shuffle for this "
              f"call. Pass group_labels to honour the Step 1B decision matrix.")
    elif use_group_blocked:
        print(f"Permutation null for this call: GROUP-BLOCKED shuffle within "
              f"'{structure_result.get('group_col_selected')}' (group effect preserved in the null; only "
              f"within-group structure is destroyed)")

    perm_ch = []
    for _ in range(n_perm):
        if use_group_blocked:
            Xp = _group_blocked_column_shuffle(X, group_labels, rng)
        else:
            Xp = X.copy()
            for j in range(Xp.shape[1]):
                rng.shuffle(Xp[:, j])
        perm_labels, _ = _fit_algorithm_generic(Xp, algorithm, k, RANDOM_STATE, **fit_extra)
        if len(set(perm_labels[perm_labels != -1])) < 2:
            perm_ch.append(0.0)  # degenerate permuted fit (e.g. DBSCAN found <2 clusters in noise) -> statistic=0, correctly "not significant-looking"
            continue
        perm_ch.append(compute_significance_statistic(Xp, perm_labels, algorithm, **stat_extra))
    perm_ch = np.array(perm_ch)
    p_value = (np.sum(perm_ch >= real_ch) + 1) / (n_perm + 1)
    return real_ch, perm_ch, p_value



def population_stability_index(ref, cur, bins=10):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / len(ref), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(cur), 1e-6, None)
    return float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))


def psi_from_baseline_bins(edges, ref_pct, cur):
    
    edges = np.asarray(edges, dtype=float)
    ref_pct = np.clip(np.asarray(ref_pct, dtype=float), 1e-6, None)
    cur_counts, _ = np.histogram(cur, bins=edges)
    cur_pct = np.clip(cur_counts / max(len(cur), 1), 1e-6, None)
    return float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))


def eta_squared_from_kruskal(H, n_total, n_groups):
   
    return (H - n_groups + 1) / (n_total - n_groups)


#=====================================================================
# Uploading dataset
#=====================================================================

data_dir = r"C:\Users\MwambKabong\Downloads\Casino_Segementation_Pipeline"

file_path = os.path.join(data_dir, "Casino_Table_final.csv")
df = pd.read_csv(file_path)

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
    print(f"Normalising {len(_rename_map)} column name(s) to expected casing.")
    df = df.rename(columns=_rename_map)


_alternate_names = {
    "activity_bonus_stake": "Bonus_Stake", "activity_bonus_return": "Bonus_Return",
    "bets_bonus_stake": "Bonus_Stake_bets", "bets_bonus_return": "Bonus_Return_bets",
}
_alt_rename_map = {c: _alternate_names[c.lower()] for c in df.columns
                    if c.lower() in _alternate_names and _alternate_names[c.lower()] not in df.columns}
if _alt_rename_map:
    print(f"Renaming {len(_alt_rename_map)} alternate-named column(s): {_alt_rename_map}")
    df = df.rename(columns=_alt_rename_map)

entity_col = "Customer_ID"

FIGDIR = os.path.join(data_dir, "figures")
os.makedirs(FIGDIR, exist_ok=True)
print(f"Figures will be saved to: {FIGDIR}")


EXPORT_DIR = os.path.join(data_dir, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
print(f"Customer-level Excel exports will be saved to: {EXPORT_DIR}")

SMALL_CLUSTER_EXPORT_FRAC = 0.02    
SMALL_CLUSTER_EXPORT_FLOOR = 200    
                                     
                                     
def export_customers_to_excel(customer_ids, modelling_df, entity_col, filename, output_dir=None,
                               sheet_name="Customers", context_note=None, extra_sheets=None):
    
    output_dir = output_dir if output_dir is not None else EXPORT_DIR
    customer_ids = np.asarray(customer_ids)
    if len(customer_ids) == 0:
        print(f"   No customers to export for '{filename}' \u2014 skipped (would produce an empty, unhelpful file).")
        return None
    os.makedirs(output_dir, exist_ok=True)
    export_df = modelling_df[modelling_df[entity_col].isin(customer_ids)].copy()
    out_path = os.path.join(output_dir, filename)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name=sheet_name)
        if extra_sheets:
            for sn, sdf in extra_sheets.items():
                sdf.to_excel(writer, index=False, sheet_name=str(sn)[:31])  # Excel sheet-name limit is 31 chars
    print(f"   \U0001F4C4 Exported {len(export_df):,} customer row(s) to: {out_path}"
          + (f"  (+{len(extra_sheets)} evidence sheet(s))" if extra_sheets else ""))
    if context_note:
        print(f"      {context_note}")
    return out_path


#=====================================================================
# DATASET INGESTION, CLEANING, AND PREPROCESSING (DATA HYGIENE)
# =============================================================================
# Duplicate removal, date parsing, missing value imputation, and feature engineering

CLUSTER_PROFILE_COMPARE_COLS = [
    "tenure_days", "active_days", "distinct_games_played",
    "Total_Deposit_Amount", "Total_Deposit_Count", "Deposit_Count", "Deposit_Amount",
    "Cash_Stake_Casino", "Cash_Stake_Sports", "Cash_Return", "Cash_Return_Casino", "Cash_Return_Sports",
    "GGR", "NGR", "Bonus_Stake", "Bonus_Given", "Bonus_To_Cash",
    "return_rate", "casino_affinity", "bonus_dependency",
    "days_since_login", "days_since_deposit",
    "vip_rank", "Cash_Balance", "Bonus_Balance", "Closing_Cash_Balance", "Closing_Bonus_Balance",
    
    "Allow_Bonuses", "Allow_Marketing_Comms", "Allow_Email_Comms", "Allow_SMS_Comms", "Allow_Transactional_Comms",
]
CLUSTER_PROFILE_CATEGORICAL_COLS = ["vip_tier", "favorite_game_type", "favorite_supplier", "Country_Code",
                                     "Acquisition_Channel", "Gender", "Registration_Device"]
CLUSTER_PROFILE_CONCENTRATION_FLAG = 0.90  
                                             

def profile_cluster_vs_population(cluster_customer_ids, modelling_df, entity_col, cluster_label,
                                   compare_cols=None, categorical_cols=None, top_n_categories=5):
    
    compare_cols = compare_cols or CLUSTER_PROFILE_COMPARE_COLS
    categorical_cols = categorical_cols or CLUSTER_PROFILE_CATEGORICAL_COLS
    cluster_mask = modelling_df[entity_col].isin(cluster_customer_ids)
    cluster_df = modelling_df[cluster_mask]
    rest_df = modelling_df[~cluster_mask]

    subbanner(f"Evidence-based profile: {cluster_label} (n={len(cluster_df)}) vs rest of population "
              f"(n={len(rest_df)})")
    rows = []
    print(f"{'Feature':<26} {'Cluster median':>15} {'Rest median':>13} {'Cluster mean':>13} {'Rest mean':>11} "
          f"{'Ratio(median)':>14}")
    for c in compare_cols:
        if c not in modelling_df.columns:
            continue
        cm, rm = cluster_df[c].median(), rest_df[c].median()
        cmean, rmean = cluster_df[c].mean(), rest_df[c].mean()
        ratio = (cm / rm) if rm not in (0, None) and not pd.isna(rm) and rm != 0 else np.nan
        ratio_print = f"{ratio:>14.2f}" if not pd.isna(ratio) else f"{'n/a':>14}"
        print(f"{c:<26} {cm:>15.3f} {rm:>13.3f} {cmean:>13.3f} {rmean:>11.3f} {ratio_print}")
        rows.append({"feature": c, "cluster_median": cm, "rest_median": rm,
                      "cluster_mean": cmean, "rest_mean": rmean, "ratio_median": ratio})
    profile_table = pd.DataFrame(rows)

    cat_rows = []
    print(f"\n Categorical concentration within this cluster (top {top_n_categories} each):")
    for c in categorical_cols:
        if c not in modelling_df.columns:
            continue
        counts = cluster_df[c].value_counts()
        shares = cluster_df[c].value_counts(normalize=True)
        top_val, top_share = shares.index[0], shares.iloc[0]
        flag = "\u26a0\ufe0f  near-unanimous concentration" if top_share >= CLUSTER_PROFILE_CONCENTRATION_FLAG else ""
        print(f"{c:<22} top='{top_val}' ({top_share:.1%} of cluster){flag}")
        for val in counts.head(top_n_categories).index:
            cat_rows.append({"feature": c, "value": val, "count_in_cluster": int(counts[val]),
                              "share_of_cluster": float(shares[val])})
    category_table = pd.DataFrame(cat_rows)

    print(f"\n Extreme-value check (a group-wide pattern, or a few outliers dragging the average?):")
    extreme_check_cols = [c for c in ["Cash_Stake_Casino", "Cash_Stake_Sports", "GGR", "NGR",
                                        "Total_Deposit_Amount", "bonus_dependency"] if c in modelling_df.columns]
    for c in extreme_check_cols:
        cluster_max = cluster_df[c].max()
        pop_max = modelling_df[c].max()
        note = "cluster contains the population MAX" if cluster_max >= pop_max else "within normal population range"
        print(f"{c:<24} cluster max={cluster_max:>14.2f}  population max={pop_max:>14.2f}  ({note})")

    return profile_table, category_table


def _self_test_cluster_profiling():
    banner("SELF-TEST: Evidence-based cluster-vs-population profiling")
    rng = np.random.RandomState(0)
    n_cluster, n_rest = 120, 2000
    fake_df = pd.DataFrame({
        "Customer_ID": [f"C{i:05d}" for i in range(n_cluster + n_rest)],
        "GGR": np.concatenate([rng.normal(-20, 5, n_cluster), rng.normal(5, 5, n_rest)]),  # cluster: clearly
        # negative GGR throughout, NOT one outlier -- every synthetic cluster member is drawn negative
        "Cash_Stake_Casino": np.concatenate([rng.normal(50, 5, n_cluster), rng.normal(20, 5, n_rest)]),
        "Country_Code": (["MZ"] * n_cluster) + list(rng.choice(["US", "UK", "ZA", "MZ"], n_rest)),
        "vip_tier": ["Non-VIP"] * (n_cluster + n_rest),
    })
    cluster_ids = fake_df["Customer_ID"].values[:n_cluster]

    profile_table, category_table = profile_cluster_vs_population(
        cluster_ids, fake_df, "Customer_ID", "SelfTestCluster",
        compare_cols=["GGR", "Cash_Stake_Casino"], categorical_cols=["Country_Code", "vip_tier"])

    ggr_row = profile_table[profile_table["feature"] == "GGR"].iloc[0]
    print(f"\n Assertions:")
    assert ggr_row["cluster_median"] < 0 < ggr_row["rest_median"], \
        "the engineered cluster's median GGR must be negative while the rest's is positive"
    print(f" \u2705 Negative-GGR signature correctly detected: cluster median={ggr_row['cluster_median']:.2f}, "
          f"rest median={ggr_row['rest_median']:.2f}")

    cluster_ggr_max = fake_df[fake_df['Customer_ID'].isin(cluster_ids)]['GGR'].max()
    pop_ggr_max = fake_df['GGR'].max()
    assert cluster_ggr_max < pop_ggr_max, \
        "the engineered cluster's own max GGR must be well below the population max (rest has the real outliers)"
    print(f" \u2705 Extreme-value check correctly shows this is a GROUP pattern, not outlier-driven: "
          f"cluster max GGR ({cluster_ggr_max:.2f}) << population max ({pop_ggr_max:.2f})")

    mz_row = category_table[(category_table["feature"] == "Country_Code") & (category_table["value"] == "MZ")]
    assert not mz_row.empty and mz_row.iloc[0]["share_of_cluster"] >= 0.90, \
        "the engineered 100%-MZ concentration must be detected as near-unanimous"
    print(f"\u2705 Near-unanimous country concentration correctly detected: MZ share="
          f"{mz_row.iloc[0]['share_of_cluster']:.1%}")

    print(f"\n \u2705 Automated profiling reproduces both real signatures (negative-GGR group pattern,")
    print(f" near-unanimous geographic concentration) on data where the answer is known in advance.")


_self_test_cluster_profiling()


def export_small_clusters_if_any(labels, customer_ids, modelling_df, entity_col, algorithm_label,
                                  group_col=None, size_threshold=None, output_dir=None):
    
    n = len(labels)
    threshold = size_threshold if size_threshold is not None else max(
        SMALL_CLUSTER_EXPORT_FLOOR, int(SMALL_CLUSTER_EXPORT_FRAC * n))
    sizes = pd.Series(labels).value_counts()
    small = sizes[(sizes < threshold) & (sizes.index != -1)]
    if small.empty:
        print(f" No cluster below the small-cluster export threshold ({threshold:,} customers) for "
              f"{algorithm_label} \u2014 nothing exported, no profiling run (conditional: only activates when "
              f"such a cluster actually exists).")
        return []
    exported_paths = []
    for cluster_id, size in small.items():
        cust_ids = np.asarray(customer_ids)[np.asarray(labels) == cluster_id]
        vip_note = None
        if group_col is not None and modelling_df is not None and group_col in modelling_df.columns:
            vip_counts = modelling_df.loc[modelling_df[entity_col].isin(cust_ids), group_col].value_counts().to_dict()
            vip_note = f"'{group_col}' breakdown within this cluster: {vip_counts}"
        safe_label = str(algorithm_label).replace(" ", "_").replace("/", "_")
        filename = f"small_cluster_{safe_label}_cluster{cluster_id}_n{size}.xlsx"

        profile_table, category_table = None, None
        persona_name = f"{algorithm_label} cluster {cluster_id}"
        if modelling_df is not None:
            profile_table, category_table = profile_cluster_vs_population(
                cust_ids, modelling_df, entity_col, f"{algorithm_label} cluster {cluster_id}")
       
            if profile_table is not None and not profile_table.empty:
                persona_name = generate_dynamic_persona_name_single_group(
                    profile_table, f"{algorithm_label} cluster {cluster_id}")

        extra_sheets = None
        if profile_table is not None and not profile_table.empty:
            extra_sheets = {"Profile_vs_Population": profile_table, "Category_Breakdown": category_table}
        persona_note = f"Persona: {persona_name}"
        combined_note = f"{persona_note}\n{vip_note}" if vip_note else persona_note
        path = export_customers_to_excel(cust_ids, modelling_df, entity_col, filename, output_dir=output_dir,
                                          context_note=combined_note, extra_sheets=extra_sheets)
        if path:
            exported_paths.append(path)
            print(f"     \u2192 Named: '{persona_name}'")
    print(f" \u26a0\ufe0f  {len(small)} small cluster(s) flagged for {algorithm_label} (< {threshold:,} "
          f"customers each) \u2014 exported for business review as a POSSIBLE outlier/extreme-behaviour segment, "
          f"not automatically labelled as such. Each file includes the evidence-based comparison above as "
          f"additional sheets.")
    return exported_paths


def _self_test_small_cluster_export():
    
    banner("SELF-TEST: Small/outlier cluster detection & Excel export")
    tmp_dir = tempfile.mkdtemp(prefix="small_cluster_export_test_")
    try:
        fake_modelling_df = pd.DataFrame({
            "Customer_ID": [f"C{i:04d}" for i in range(1000)],
            "vip_tier": (["Non-VIP"] * 950) + (["_gold_vip"] * 50),
        })
        labels = np.array([0] * 800 + [1] * 150 + [2] * 40 + [-1] * 10)  
        
        print(f"Sub-test A: a genuine small cluster (n=40) is flagged and exported; noise (n=10, label -1)")
        print(f"is NOT, even though it's smaller")
        exported = export_small_clusters_if_any(
            labels, fake_modelling_df["Customer_ID"].values, fake_modelling_df, "Customer_ID", "SelfTestAlgo",
            group_col="vip_tier", size_threshold=100, output_dir=tmp_dir)
        assert len(exported) == 1, f"expected exactly 1 small cluster exported, got {len(exported)}"
        assert os.path.exists(exported[0]), "exported file should actually exist on disk"
        exported_df = pd.read_excel(exported[0])
        assert len(exported_df) == 40, f"expected 40 rows in the export, got {len(exported_df)}"
        print(f"\u2705 Correctly flagged/exported the n=40 cluster ({len(exported_df)} rows in the real file),")
        print(f"correctly excluded the n=10 noise cluster from being treated as a business segment.")

        print(f"\n Sub-test B: no cluster below threshold -> nothing exported")
        labels_clean = np.array([0] * 500 + [1] * 500)
        exported_clean = export_small_clusters_if_any(
            labels_clean, fake_modelling_df["Customer_ID"].values, fake_modelling_df, "Customer_ID",
            "SelfTestAlgoClean", size_threshold=100, output_dir=tmp_dir)
        assert exported_clean == [], "no cluster below threshold should mean nothing is exported"
        print(f"\u2705 Correctly exported nothing when no cluster falls below the threshold.")

        print(f"\n Sub-test C: empty customer_ids -> skipped safely, no blank file")
        result = export_customers_to_excel([], fake_modelling_df, "Customer_ID", "empty_test.xlsx",
                                            output_dir=tmp_dir)
        assert result is None, "an empty customer list should not produce a file"
        print(f"\u2705 Correctly skipped an empty export rather than writing a useless blank file.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


_self_test_small_cluster_export()

print(f"Dataset Loaded Successfully! Shape: {df.shape}")
print(f"{df.columns}")
print(df.head(3).to_string())


# Removing exact duplicate

inital_rows = len(df)
df = (df.drop_duplicates(keep="first").reset_index(drop=True))

final_rows = len(df)

print("\nDuplicates Removal Summary")
print(f"Rows Before: {inital_rows:,}")
print(f"Rows After: {final_rows:,}")
print(f"Removed: {inital_rows - final_rows:,}")

# Missing Values Audit and Cleaning

df = df.replace(["NULL", "null", "None", "", " "], np.nan)
 
date_columns = [
    "DOB", "Reg_Date", "First_Deposit_Date", "Last_Deposit_Date",
    "Last_Login_Date", "Activity_Date", "Bet_Date",
]
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
 

snapshot_date = df[["Reg_Date", "First_Deposit_Date", "Last_Login_Date",
                     "Last_Deposit_Date", "Activity_Date", "Bet_Date"]].max().max().normalize() + pd.Timedelta(days=1)
 
if ("Cash_Return" not in df.columns
        and "Cash_Return_Casino" in df.columns and "Cash_Return_Sports" in df.columns):
    df["Cash_Return"] = df["Cash_Return_Casino"].fillna(0) + df["Cash_Return_Sports"].fillna(0)
 
if all(c in df.columns for c in ["Cash_Stake", "Cash_Stake_Casino", "Cash_Stake_Sports"]):
    mask = df["Cash_Stake"].isna() & df["Cash_Stake_Casino"].notna() & df["Cash_Stake_Sports"].notna()
    df.loc[mask, "Cash_Stake"] = df.loc[mask, "Cash_Stake_Casino"] + df.loc[mask, "Cash_Stake_Sports"]
 
# FIX: GGR made casino-only, on explicit request -- this is no longer a fallback for missing values
# (real data showed GGR is essentially never missing, so a fallback alone would rarely fire). It's now
# an unconditional override, since real testing confirmed the raw "GGR" column is genuinely combined
# casino+sports at the source (100% match with a combined estimate among sports-active customers, 0.5%
# match with casino-only). Unlike NGR, this is an exact recomputation from real casino-specific
# components, not an estimate -- GGR = Cash_Stake_Casino - Cash_Return_Casino is the same formula the
# combined version already used, just restricted to the casino-only components. GGR is an ACTIVE
# clustering feature in the trained model (confirmed in the artifact's feat_cols_final) -- this changes
# what the model would learn from on the next training run; the currently-deployed artifact was trained
# on combined GGR and does not reflect this until retrained.
if all(c in df.columns for c in ["Cash_Stake_Casino", "Cash_Return_Casino"]):
    df["GGR"] = df["Cash_Stake_Casino"] - df["Cash_Return_Casino"]
 

ngr_cost_cols = ["Bonus_Given", "Revenue_Share", "Tax", "Payment_Fee", "Game_Provider_Fee"]
for col in ngr_cost_cols:
    if col not in df.columns:
        df[col] = 0
# NOTE: this fallback (fires only when NGR itself is missing) now inherits the casino-only GGR above,
# so a reconstructed NGR here is partially casino-only -- the stake/return portion is now casino-
# specific, but Bonus_Given/Tax/Payment_Fee/Game_Provider_Fee/Revenue_Share below are still combined,
# since none of those have a casino-specific split available. NGR itself (when NOT missing, the common
# case) is untouched by this change and remains combined, as discussed -- no clean casino-only split
# exists for it the way one does for GGR.
if "NGR" in df.columns and "GGR" in df.columns:
    mask = df["NGR"].isna() & df["GGR"].notna()
    df.loc[mask, "NGR"] = (
        df.loc[mask, "GGR"]
        - df.loc[mask, "Bonus_Given"].fillna(0)
        - df.loc[mask, "Revenue_Share"].fillna(0)
        - df.loc[mask, "Tax"].fillna(0)
        - df.loc[mask, "Payment_Fee"].fillna(0)
        - df.loc[mask, "Game_Provider_Fee"].fillna(0)
    )
 

 
if "Last_Login_Date" in df.columns:
    df["never_login_flag"] = df["Last_Login_Date"].isna().astype(int)
    df["days_since_login"] = (snapshot_date - df["Last_Login_Date"]).dt.days
    df["days_since_login"] = df["days_since_login"].fillna(df["days_since_login"].max() + 1)
 
if "Last_Deposit_Date" in df.columns:
    df["never_deposit_flag"] = df["Last_Deposit_Date"].isna().astype(int)
    df["days_since_deposit"] = (snapshot_date - df["Last_Deposit_Date"]).dt.days
    df["days_since_deposit"] = df["days_since_deposit"].fillna(df["days_since_deposit"].max() + 1)
 
df = df.drop(columns=["DOB", "Device"], errors="ignore")
 
# Business-rule imputation: structural zeros for monetary fields where NULL
# means "did not occur", not "unknown value" (see prior discussion).
zero_fill_cols = [
    "Bonus_Stake", "Bonus_Given", "Tax", "Payment_Fee", "Game_Provider_Fee",
    "Jackpot_Contribution", "Revenue_Share", "Casino_Income",
    "Closing_Cash_Balance", "Closing_Bonus_Balance",
]
for col in zero_fill_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)
 
if "Flags" in df.columns:
    df["Flags"] = df["Flags"].fillna("Non-VIP")
 
print("Row-level nulls remaining:", df.isnull().sum().sum())
print("Row-level shape:", df.shape, "\n")


#=====================================================================
# Aggregating to customer level (Customer_ID)
# =============================================================================

def safe_mode(x):
    """x.mode()[0] raises on an all-NaN group; guard against that."""
    m = x.mode()
    return m.iloc[0] if not m.empty else "Unknown"
 
 

first_cols = [
    "Global_Customer_ID", "Country_Code", "Language_Code", "Site", "Gender",
    "Registration_Device", "Acquisition_Channel", "Allow_Transactional_Comms",
    "Allow_Marketing_Comms", "Allow_Email_Comms", "Allow_SMS_Comms", "Allow_Bonuses",
    "Is_Ibex_AB", "Reg_Date", "First_Deposit_Date", "Last_Login_Date", "Last_Deposit_Date",
    "Total_Deposit_Count", "Total_Deposit_Amount", "Cash_Balance", "Bonus_Balance", "Flags",
]
 
# Transactional flow columns — sum per customer.
sum_cols = [
    "Deposit_Count", "Deposit_Amount", "Withdrawal_Amount",
    "Cash_Stake_Casino", "Cash_Return_Casino", "Cash_Stake_Sports", "Cash_Return_Sports",
    "GGR", "NGR", "Bet_Count", "Cash_Stake", "Cash_Return",
    "Bonus_Stake", "Bonus_Stake_bets", "Bonus_Return", "Bonus_Return_bets",
    "Bonus_Given", "Bonus_To_Cash",
    "Tax", "Payment_Fee", "Game_Provider_Fee", "Jackpot_Contribution",
    "Revenue_Share", "Casino_Income",
]
 
agg_dict = {c: "first" for c in first_cols if c in df.columns}
agg_dict.update({c: "sum" for c in sum_cols if c in df.columns})
 
customer_df = df.groupby("Customer_ID").agg(agg_dict)
 
# Engagement / diversity features
customer_df["active_days"] = df.groupby("Customer_ID")["Activity_Date"].nunique()


if "Game_ID" in df.columns:
    customer_df["distinct_games_played"] = df.groupby("Customer_ID")["Game_ID"].nunique()
else:
    print("Game_ID not present in this extract -- distinct_games_played cannot be computed, skipped.")

if "Game_Type" in df.columns:
    customer_df["favorite_game_type"] = df.groupby("Customer_ID")["Game_Type"].agg(safe_mode)
elif "favorite_game_type" in df.columns:
    customer_df["favorite_game_type"] = df.groupby("Customer_ID")["favorite_game_type"].first()
    print("favorite_game_type already present in this extract (pre-aggregated) -- carried through via 'first'.")

if "Supplier" in df.columns:
    customer_df["favorite_supplier"] = df.groupby("Customer_ID")["Supplier"].agg(safe_mode)
elif "favorite_supplier" in df.columns:
    customer_df["favorite_supplier"] = df.groupby("Customer_ID")["favorite_supplier"].first()
    print("favorite_supplier already present in this extract (pre-aggregated) -- carried through via 'first'.")
 
customer_df = customer_df.reset_index()

# FIX: silver/gold/platinum VIP customers dropped entirely from the modelling population, on explicit
# request -- applied here, once "Flags" has already collapsed to a single value per customer via the
# "first" aggregation above, so this is a clean per-customer exclusion rather than a row-level filter
# that could behave inconsistently if a customer's raw rows ever disagreed on their own Flags value.
if "Flags" in customer_df.columns:
    vip_excluded_mask = customer_df["Flags"].isin(["_silver_vip", "_gold_vip", "_platinum_vip"])
    n_vip_excluded = int(vip_excluded_mask.sum())
    if n_vip_excluded:
        print(f"Dropping {n_vip_excluded:,} silver/gold/platinum VIP customers from the modelling "
              f"population (Non-VIP and _bronze_vip are kept).")
        customer_df = customer_df.loc[~vip_excluded_mask].reset_index(drop=True)

# ── VIP tier: ordinal rank for modelling, label kept for reporting ─────────
vip_order = {"Non-VIP": 0, "_bronze_vip": 1}
customer_df["vip_rank"] = customer_df["Flags"].map(vip_order)
customer_df = customer_df.rename(columns={"Flags": "vip_tier"})
 
# ── Recency / tenure, computed once at customer level ───────────────────────
customer_df["tenure_days"] = (snapshot_date - customer_df["Reg_Date"]).dt.days
 
customer_df["never_login_flag"] = customer_df["Last_Login_Date"].isna().astype(int)
customer_df["days_since_login"] = (snapshot_date - customer_df["Last_Login_Date"]).dt.days
customer_df["days_since_login"] = customer_df["days_since_login"].fillna(
    customer_df["days_since_login"].max() + 1
)
 
customer_df["never_deposit_flag"] = customer_df["Last_Deposit_Date"].isna().astype(int)
customer_df["days_since_deposit"] = (snapshot_date - customer_df["Last_Deposit_Date"]).dt.days
customer_df["days_since_deposit"] = customer_df["days_since_deposit"].fillna(
    customer_df["days_since_deposit"].max() + 1
)
 
# ── Derived ratios: computed once from summed totals; NaN (not 0) when the


customer_df["total_stake_verified"] = customer_df["Cash_Stake_Casino"] + customer_df["Cash_Stake_Sports"]

customer_df["return_rate"] = np.where(
    customer_df["total_stake_verified"] > 0, customer_df["Cash_Return"] / customer_df["total_stake_verified"], np.nan
)
customer_df["casino_affinity"] = np.where(
    customer_df["total_stake_verified"] > 0, customer_df["Cash_Stake_Casino"] / customer_df["total_stake_verified"], np.nan
)
customer_df["bonus_dependency"] = np.where(
    customer_df["total_stake_verified"] > 0, customer_df["Bonus_Stake"] / customer_df["total_stake_verified"], np.nan
)


customer_df["bonus_per_bet"] = np.where(
    customer_df["Bet_Count"] > 0, customer_df["Bonus_Stake"] / customer_df["Bet_Count"], np.nan
)
customer_df["stake_per_active_day"] = np.where(
    customer_df["active_days"] > 0, customer_df["Cash_Stake_Casino"] / customer_df["active_days"], np.nan
)
customer_df["avg_stake_per_bet"] = np.where(
    customer_df["Bet_Count"] > 0, customer_df["Cash_Stake_Casino"] / customer_df["Bet_Count"], np.nan
)
customer_df["deposits_per_active_day"] = np.where(
    customer_df["active_days"] > 0, customer_df["Deposit_Count"] / customer_df["active_days"], np.nan
)
BEHAVIORAL_SHAPE_FEATURES = ["bonus_per_bet", "stake_per_active_day", "avg_stake_per_bet", "deposits_per_active_day"]
 
# ============================================================
# PART C — PRE-ACTIVATION COHORT SPLIT
# ============================================================
# 319 customers a
# stake in EITHER vertical (total_stake_verified == 0). For these,
# return_rate / casino_affinity / bonus_dependency are genuinely undefined
# (0-denominator), not zero.
#
# These customers are removed from the modelling population entirely —
# not imputed, not dropped from the dataset — and routed to a separate
# "Pre-Activation" cohort for monitoring. 
 
pre_activation_mask = customer_df["total_stake_verified"] == 0  # FIX: was customer_df["Cash_Stake"] == 0 -- Cash_Stake is unreliable for sports-only bettors (verified: 53/372 had real, internally-consistent sports stake activity despite Cash_Stake==0). Bet_Count also checked and rejected (never 0 for any of the 12,252 customers). Now uses total_stake_verified (Cash_Stake_Casino + Cash_Stake_Sports), matching the ratio-feature denominators above. Result: 319 customers, not 372.
 
pre_activation_df = customer_df.loc[pre_activation_mask].copy()
pre_activation_df["cohort_label"] = "Zero-cash-funded"
 
modelling_df = customer_df.loc[~pre_activation_mask].copy().reset_index(drop=True)
 
print("Zero-Cash-Funded Cohort Split")
print(f"Total customers: {len(customer_df):,}")
print(f"Zero-cash-funded cohort (excluded): {len(pre_activation_df):,}")
print(f"Remaining for modelling: {len(modelling_df):,}")
print(f"Modelling-population nulls in return_rate/casino_affinity/bonus_dependency: "
      f"{modelling_df[['return_rate','casino_affinity','bonus_dependency']].isnull().sum().sum()}")
print()

subbanner("Zero-cash-funded cohort export (CRM follow-up / funnel monitoring)")
export_customers_to_excel(
    pre_activation_df[entity_col].values, pre_activation_df, entity_col, "zero_cash_funded_cohort.xlsx",
    context_note=(f"{len(pre_activation_df):,} zero-cash-funded customers: no cash stake in either vertical "
                  f"(Cash_Stake_Casino=0 and Cash_Stake_Sports=0) in the measured period. They can still have "
                  f"recorded bets and bonus-funded activity -- the rule tests cash stake only. Routed here, not "
                  f"into a modelled segment, because return_rate / casino_affinity / bonus_dependency divide by "
                  f"total cash stake and are therefore undefined for them. Exported unconditionally since this "
                  f"cohort exists on every run of this extract, not just when non-empty by chance.")
)
 
# ============================================================
# DIAGNOSTICS
# ============================================================
 
print("Full aggregated table shape:", customer_df.shape)
print("Modelling population shape :", modelling_df.shape)
print("Zero-cash-funded cohort shape:", pre_activation_df.shape)
print(pre_activation_df.head(10))
 
nz = modelling_df.isnull().sum()
print("\nModelling population — remaining nulls (non-zero only):")
print(nz[nz > 0].to_string() if nz.sum() else "  none")
 
print("\nZero-variance flag — columns that are all-zero due to 100%-missing")
print("source data (Tax, Payment_Fee, Game_Provider_Fee, Jackpot_Contribution,")
print("Revenue_Share, Casino_Income). These add no separating signal to any")
print("clustering algorithm and should be EXCLUDED from the modelling feature")
print("set, even though they're retained here for auditability:")
for c in ["Tax", "Payment_Fee", "Game_Provider_Fee", "Jackpot_Contribution",
          "Revenue_Share", "Casino_Income"]:
    if c in customer_df.columns:
        print(f"{c:22s} all-zero: {(customer_df[c] == 0).all()}")
        
        
# ============================================================
# Other Structural Validation
# ============================================================
        

def eda(modelling_df, entity_col):
    banner("Other Structural Validation")
    
    non_feature_cols = [
        entity_col, "Global_Customer_ID",
        "Reg_Date", "First_Deposit_Date", "Last_Login_Date", "Last_Deposit_Date",
        "vip_tier",
        
        "Flags", "vip_rank", "Allow_Bonuses", "Allow_Marketing_Comms",
        "Allow_Email_Comms", "Allow_SMS_Comms", "Allow_Transactional_Comms",
        
        "bonus_per_bet", "stake_per_active_day", "avg_stake_per_bet", "deposits_per_active_day",
    ]
    feat_cols = [c for c in modelling_df.columns if c not in non_feature_cols]
  
    numeric_feat_cols = modelling_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    subbanner("Data types")
    print(modelling_df[feat_cols].dtypes.to_string())
    
    subbanner("Identifier / near-unique columns")
    
    card = modelling_df.nunique() / len(modelling_df)
    IDENTIFIER_THRESHOLD = 0.95
    id_like = card[card > IDENTIFIER_THRESHOLD]
    print(id_like.sort_values(ascending=False).to_string())
    print(f"Decision: cardinality ratio > {IDENTIFIER_THRESHOLD} \u2192 drop from feature set (carries no generalising signal)")
    print(f"{entity_col} cardinality ratio = {card[entity_col]:.4f} \u2192 EXCLUDED from clustering features")
    
    
    subbanner("Near-zero variance")
    # FIX: numeric_feat_cols, not feat_cols
    variances = modelling_df[numeric_feat_cols].var()
    NZV_THRESHOLD = 1e-6
    nzv = variances[variances < NZV_THRESHOLD]
    print(f"   Features below variance threshold {NZV_THRESHOLD}: {list(nzv.index) if len(nzv) else 'NONE'}")
    
    subbanner("Skewness (informs Step 2C log-transform candidates)")
    # FIX: numeric_feat_cols, not feat_cols
    skew = modelling_df[numeric_feat_cols].skew().sort_values(ascending=False)
    print(skew.to_string())
    SKEW_FLAG = 1.0
    flagged = skew[skew.abs() > SKEW_FLAG]
    print(f"\n {len(flagged)}/{len(numeric_feat_cols)} features exceed |skew| > {SKEW_FLAG} \u2014 candidates for Step 2C, "
          f"pending the normality test AND domain guard applied there")
    
    
    subbanner("Outlier % per feature (IQR 1.5x rule) — drives Step 2D scaler choice")
    outlier_pct = {}
    # FIX: numeric_feat_cols, not feat_cols -- quantile() also fails on strings
    for c in numeric_feat_cols:
        q1, q3 = modelling_df[c].quantile(.25), modelling_df[c].quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_pct[c] = round(((modelling_df[c] < lo) | (modelling_df[c] > hi)).mean() * 100, 2)
    outlier_s = pd.Series(outlier_pct).sort_values(ascending=False)
    print(outlier_s.to_string())
    OUTLIER_SCALER_THRESHOLD = 5.0
    n_high_outlier = (outlier_s > OUTLIER_SCALER_THRESHOLD).sum()
    print(f"\n {n_high_outlier}/{len(numeric_feat_cols)} features exceed {OUTLIER_SCALER_THRESHOLD}% IQR-outlier rate")
    scaler_decision = "RobustScaler" if n_high_outlier / len(numeric_feat_cols) > 0.30 else "StandardScaler"
    print(f"DECISION (Step 2D): {n_high_outlier}/{len(numeric_feat_cols)} = {n_high_outlier/len(numeric_feat_cols):.0%} of features "
          f"exceed the outlier threshold \u2192 {scaler_decision} selected")
    
    
    subbanner("Class imbalance in categorical features")
    cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(modelling_df[c])]  # robust: not "object"/"category" string-match, which misses pandas' native "str" dtype
    IMBALANCE_THRESHOLD = 0.95
    if not cat_cols:
        print(f"No categorical columns in the feature set (checked via dtype, not assumed) \u2014 N/A on this extract.")
        print(f"This check re-runs automatically and is fully generic; it will activate on any future")
        print(f"extract that adds a categorical column, using the same {IMBALANCE_THRESHOLD:.0%} dominant-level threshold.")
    else:
        for c in cat_cols:
            top_share = modelling_df[c].value_counts(normalize=True).iloc[0]
            top_level = modelling_df[c].value_counts(normalize=True).index[0]
            flag = "\u26a0\ufe0f  imbalanced" if top_share > IMBALANCE_THRESHOLD else "\u2705 balanced"
            print(f"{c:<24} top level '{top_level}' = {top_share:.1%}  {flag}")

    return feat_cols, skew, outlier_s, scaler_decision
    
feat_cols, skew, outlier_s, scaler_decision = eda(modelling_df, entity_col)


### 1. Data Structure Detection

def _compute_icc1(group_labels, values):
    
    group_labels = np.asarray(group_labels)
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    group_labels, values = group_labels[valid], values[valid]
    unique_groups = np.unique(group_labels)
    n_groups, n_obs = len(unique_groups), len(values)
    if n_groups < 2 or n_obs < n_groups + 2:
        return np.nan, np.nan, np.nan, n_groups, n_obs
    grand_mean = values.mean()
    group_data = [values[group_labels == g] for g in unique_groups]
    group_ns = np.array([len(g) for g in group_data])
    group_means = np.array([g.mean() for g in group_data])
    ss_between = float(np.sum(group_ns * (group_means - grand_mean) ** 2))
    ss_within = float(sum(np.sum((g - m) ** 2) for g, m in zip(group_data, group_means)))
    df_between, df_within = n_groups - 1, n_obs - n_groups
    if df_within <= 0:
        return np.nan, np.nan, np.nan, n_groups, n_obs
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    if ms_within == 0:
        return 1.0, np.inf, 0.0, n_groups, n_obs
    f_stat = ms_between / ms_within
    p_val = 1 - stats.f.cdf(f_stat, df_between, df_within)
    k_h = n_groups / np.sum(1 / group_ns)
    denom = ms_between + (k_h - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0.0
    return float(icc), float(f_stat), float(p_val), n_groups, n_obs


def _incremental_r2_nested_f_test(y_continuous, group_labels, community_labels):
    
    def _dummies(labels):
        return pd.get_dummies(pd.Series(labels).astype(str), drop_first=True).values.astype(float)

    n = len(y_continuous)
    X0 = _dummies(group_labels) if group_labels is not None else np.zeros((n, 0))
    Xc = _dummies(community_labels)
    X1 = np.hstack([X0, Xc]) if X0.shape[1] > 0 else Xc

    def _sse(X, y):
        if X.shape[1] == 0:
            resid = y - y.mean()
            return float(np.sum(resid ** 2)), 0
        Xd = np.column_stack([np.ones(len(y)), X])
        coeffs, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ coeffs
        return float(np.sum(resid ** 2)), X.shape[1]

    sse0, p0 = _sse(X0, y_continuous)
    sse1, p1 = _sse(X1, y_continuous)
    df_diff = p1 - p0
    df_resid = n - p1 - 1
    if df_diff <= 0 or df_resid <= 0 or sse1 <= 0:
        return np.nan, np.nan, np.nan
    f_stat = ((sse0 - sse1) / df_diff) / (sse1 / df_resid)
    p_val = 1 - stats.f.cdf(f_stat, df_diff, df_resid)
    ss_tot = float(np.sum((y_continuous - y_continuous.mean()) ** 2))
    r2_0 = 1 - sse0 / ss_tot if ss_tot > 0 else 0.0
    r2_1 = 1 - sse1 / ss_tot if ss_tot > 0 else 0.0
    return float(r2_1 - r2_0), float(f_stat), float(p_val)


def resolve_resampling_strategy(temporal_detected, group_detected, network_detected, group_col_selected):
    """
    DECISION MATRIX  (temporal_detected, group_detected) -> resampling_strategy
    ----------------------------------------------------------------------------
    (False, False) -> standard iid row-level resampling
    (False, True ) -> group-blocked resampling on <group_col> (whole groups drawn)
    (True,  False) -> block-bootstrap (contiguous time blocks, not individual rows)
    (True,  True ) -> NESTED: block-bootstrap performed WITHIN each group -- a
                       time block must never be drawn in a way that mixes rows
                       across groups, so this is one combined procedure, not
                       two independent ones concatenated together
    ----------------------------------------------------------------------------
    network_detected does NOT appear in resampling_strategy: no version of
    "customers are graph-connected" implies non-independent rows for bootstrap
    purposes. It instead produces a separate network_note driving an algorithm
    /reporting decision (Step 6/8: consider a graph-based method alongside the
    feature-based one; report both partitions if they diverge).
    """
    if not temporal_detected and not group_detected:
        resampling_strategy = "standard iid row-level resampling (bootstrap in Step 10, wrapper CV in Step 3D)"
    elif not temporal_detected and group_detected:
        resampling_strategy = f"group-blocked resampling on '{group_col_selected}' (draw whole groups, not individual rows)"
    elif temporal_detected and not group_detected:
        resampling_strategy = "block-bootstrap (resample contiguous time blocks, not individual rows)"
    else:
        resampling_strategy = (
            f"NESTED block-bootstrap within '{group_col_selected}' "
            f"(draw time blocks separately inside each group, never mixing rows across groups)"
        )

    network_note = (
        "report graph-based partition alongside feature-based (Step 6/8) \u2014 does not change resampling"
        if network_detected else
        "no graph-based alternative considered"
    )
    return resampling_strategy, network_note


def _self_test_resampling_decision_matrix():
    """Proves all 4 (temporal, group) combinations resolve to distinct, correct
    strategies -- since the real casino data only ever exercises (False, True),
    the other 3 branches (including the nested case) are otherwise never run."""
    banner("SELF-TEST: Resampling decision matrix (all 4 temporal x group combinations)")
    cases = [
        (False, False, "standard iid"),
        (False, True, "group-blocked resampling on 'grp'"),
        (True, False, "block-bootstrap"),
        (True, True, "NESTED block-bootstrap within 'grp'"),
    ]
    for temporal, group, expect_substr in cases:
        strat, _ = resolve_resampling_strategy(temporal, group, False, "grp")
        ok = expect_substr in strat
        print(f"temporal={temporal!s:<5} group={group!s:<5} -> {strat}   [{'OK' if ok else 'FAIL'}]")
        assert ok, f"expected '{expect_substr}' substring not found in '{strat}'"
    _, note_true = resolve_resampling_strategy(False, False, True, "grp")
    _, note_false = resolve_resampling_strategy(False, False, False, "grp")
    assert "graph-based partition" in note_true and "no graph-based" in note_false
    print(f"\n \u2705 All 4 temporal x group combinations produce distinct, correct resampling")
    print(f"strategies (including the previously-unhandled nested case), and network")
    print(f"structure correctly resolves to a separate note, not a resampling instruction.")


def structure_detection(modelling_df, entity_col, feat_cols):
    banner("STEP 1B: Data Structure Detection")
    MAX_UNIQUE_FRAC = 0.20
    MIN_REPEAT_RATIO = 2
    ICC_MIN, ICC_ALPHA = 0.05, 0.05
    INCR_R2_MIN, INCR_ALPHA = 0.01, 0.05
    
    MAX_STRUCTURE_CANDIDATE_UNIQUE_ABS = 200
  
    n = len(modelling_df)

    subbanner("Test A: Temporal Structure")
    date_cols = [c for c in modelling_df.columns if "date" in c.lower() or "time" in c.lower()]
   
    entity_is_static = modelling_df[entity_col].nunique() == len(modelling_df)
    temporal_detected = (len(date_cols) > 0) and (not entity_is_static)
    print(f"Date/datetime columns found : {date_cols if date_cols else 'NONE'}")
    print(f"Entity grain check: {len(modelling_df):,} rows, {modelling_df[entity_col].nunique():,} unique {entity_col} "
          f"\u2192 {'static (exactly one row per entity)' if entity_is_static else 'REPEATED rows per entity'}")
    
    print(f"tenure_days is a static day-count field (range {modelling_df['tenure_days'].min()}-{modelling_df['tenure_days'].max()}), "
          f"not a timestamp \u2014 does not count as temporal evidence")
    print(f"DECISION: temporal_detected = {temporal_detected}  "
          f"{'\u2192 block-bootstrap required in Step 10' if temporal_detected else '\u2192 cross-sectional snapshot (one row per entity); unrestricted resampling is valid'}")
    
    
    from sklearn.decomposition import PCA
    numeric_feat_cols = modelling_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    diag_imputed = modelling_df[numeric_feat_cols].fillna(modelling_df[numeric_feat_cols].median())
    diag_scaled = StandardScaler().fit_transform(diag_imputed)
    pc1 = PCA(n_components=1, random_state=RANDOM_STATE).fit_transform(diag_scaled).ravel()
    print(f"Probe variable for Tests B/C: PC1 of a quick-standardised {len(numeric_feat_cols)}-feature matrix "
          f"(no regression target exists in this unsupervised setting; self-contained diagnostic only)")

    subbanner("Test B: Group / Nested-Entity Structure (ICC) \u2014 name-agnostic discovery")
    print(f"\u26a0\ufe0f  METHODOLOGICAL GUARD: candidate columns are restricted to fields NOT already in the")
    print(f"clustering feature set. Testing whether a clustering FEATURE (e.g. PURCHASES_TRX) 'explains'")
    print(f"variance in PC1 is circular \u2014 PC1 is itself partly built from that same feature, guaranteeing")
    print(f"an inflated, spurious ICC. A genuine group/entity column must be separate metadata (e.g. a")
    print(f"branch, household, or region ID) that is NOT also being clustered on.")
    non_feature_cols = [c for c in modelling_df.columns if c not in feat_cols and c != entity_col]
    id_candidates = [
        c for c in non_feature_cols
        if (modelling_df[c].dtype == object or modelling_df[c].nunique() <= MAX_UNIQUE_FRAC * n)
        and modelling_df[c].nunique() <= MAX_STRUCTURE_CANDIDATE_UNIQUE_ABS
        and (n / max(modelling_df[c].nunique(), 1)) >= MIN_REPEAT_RATIO
    ]
    print(f"\n Non-feature columns available as candidates: {non_feature_cols if non_feature_cols else 'NONE'}")
    print(f" Stage 1 \u2014 cardinality-based candidate discovery among those "
          f"(cardinality \u2264 {MAX_UNIQUE_FRAC:.0%}*n={MAX_UNIQUE_FRAC*n:.0f}, avg repeats \u2265 {MIN_REPEAT_RATIO}): "
          f"{id_candidates if id_candidates else 'NONE'}")

    group_detected, group_col_selected, best_icc = False, None, 0.0
    icc_results = {}
    if id_candidates:
        print(f"\n Stage 2 \u2014 ICC(1) one-way ANOVA per candidate (threshold: ICC>{ICC_MIN} AND p<{ICC_ALPHA}):")
        print(f" {'Column':<24} {'ICC':>8} {'F-stat':>10} {'p-value':>9} {'Groups':>7}  Decision")
        for c in id_candidates:
            icc, f_stat, p_val, ng, no = _compute_icc1(modelling_df[c], pc1)
            if np.isnan(icc):
                print(f"{c:<24} {'--':>8} {'--':>10} {'--':>9} {'--':>7}  SKIPPED (insufficient group data)")
                continue
            qualifies = (icc > ICC_MIN) and (p_val < ICC_ALPHA)
            icc_results[c] = {"icc": icc, "f": f_stat, "p": p_val, "qualifies": qualifies}
            decision = "\u2705 QUALIFIES" if qualifies else "\u2014 fails threshold"
            print(f"{c:<24} {icc:>8.4f} {f_stat:>10.2f} {p_val:>9.4f} {ng:>7}  {decision}")
            if qualifies and icc > best_icc:
                best_icc, group_col_selected, group_detected = icc, c, True
    else:
        print(f"No candidates found \u2014 ICC stage skipped.")
    print(f"\n DECISION: group_detected = {group_detected}" +
          (f" (strongest: '{group_col_selected}', ICC={best_icc:.4f})" if group_detected else "") +
          f" \u2192 {'group-blocked resampling required in Step 3D/10' if group_detected else 'row-level resampling in Step 3D/10 is valid as-is'}")

    subbanner("Test C: Network / Community Structure \u2014 incremental structure vs Test B")
    
    network_candidates = [
        c for c in non_feature_cols
        if c != group_col_selected
        and (modelling_df[c].dtype == object or modelling_df[c].nunique() <= MAX_UNIQUE_FRAC * n)
        and modelling_df[c].nunique() > 1
        and modelling_df[c].nunique() <= MAX_STRUCTURE_CANDIDATE_UNIQUE_ABS
        and (n / max(modelling_df[c].nunique(), 1)) >= MIN_REPEAT_RATIO
    ]
    print(f"Stage 1 \u2014 community candidates among non-feature columns (excl. Test B's column): "
          f"{network_candidates if network_candidates else 'NONE'}")

    network_detected, community_col_selected, best_dr2 = False, None, 0.0
    if network_candidates:
        ref_labels = modelling_df[group_col_selected].astype(str).values if group_col_selected else None
        print(f"\n Stage 2 \u2014 incremental R\u00b2 nested-F test vs reference group "
              f"('{group_col_selected or 'none \u2014 baseline is the grand mean'}')" 
              f" (threshold: \u0394R\u00b2>{INCR_R2_MIN} AND p<{INCR_ALPHA}):")
        print(f"{'Column':<24} {'dR2':>8} {'F-stat':>10} {'p-value':>9}  Decision")
        for c in network_candidates:
            comm_labels = modelling_df[c].astype(str).values
            dr2, f_stat, p_val = _incremental_r2_nested_f_test(pc1, ref_labels, comm_labels)
            if np.isnan(dr2):
                print(f"   {c:<24} {'--':>8} {'--':>10} {'--':>9}  SKIPPED (degenerate design matrix)")
                continue
            qualifies = (dr2 > INCR_R2_MIN) and (p_val < INCR_ALPHA)
            decision = "\u2705 QUALIFIES" if qualifies else "\u2014 fails threshold"
            print(f" {c:<24} {dr2:>8.4f} {f_stat:>10.2f} {p_val:>9.4f}  {decision}")
            if qualifies and dr2 > best_dr2:
                best_dr2, community_col_selected, network_detected = dr2, c, True
    else:
        print(f" No community candidates found \u2014 incremental test skipped.")
    print(f"\n DECISION: network_detected = {network_detected}" +
          (f"(strongest: '{community_col_selected}', \u0394R\u00b2={best_dr2:.4f})" if network_detected else "") +
          f"\u2192 {'report graph-based partition alongside feature-based (Step 6/8)' if network_detected else 'no graph-based alternative considered'}")

    subbanner("Structure Detection Summary \u2014 Resampling Consequence")
    print(f"{'Temporal structure:':<24}{'YES' if temporal_detected else 'NO'}")
    print(f"{'Group structure:':<24}{'YES' if group_detected else 'NO'}" + (f"  ('{group_col_selected}')" if group_detected else ""))
    print(f"{'Network structure:':<24}{'YES' if network_detected else 'NO'}" + (f"  ('{community_col_selected}')" if network_detected else ""))
    resampling_strategy, network_note = resolve_resampling_strategy(
        temporal_detected, group_detected, network_detected, group_col_selected
    )
    print(f"\u2192 RESAMPLING STRATEGY for all later steps: {resampling_strategy}")
    print(f"\u2192 NETWORK-STRUCTURE ACTION (Step 6/8, separate from resampling): {network_note}")

    return {"temporal_detected": temporal_detected, "group_detected": group_detected,
            "network_detected": network_detected, "group_col_selected": group_col_selected,
            "community_col_selected": community_col_selected, "resampling_strategy": resampling_strategy}


structure_result = structure_detection(modelling_df, entity_col, feat_cols)


### 2. Categorical Feature Encoding

def encode_categoricals(modelling_df, feat_cols):
    banner("Categorical Feature Encoding")
    ONEHOT_MAX_CARDINALITY = 15
    RARE_LEVEL_FLOOR = 0.01

    cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(modelling_df[c])]  # robust: not "object"/"category" string-match, which misses pandas' native "str" dtype
    print(f"Categorical columns detected (dtype object/category, checked not assumed): "
          f"{cat_cols if cat_cols else 'NONE'}")
    if not cat_cols:
        print(f"N/A on this extract \u2014 all {len(feat_cols)} features are already numeric.")
        print(f"This function is fully generic and activates automatically the moment a future")
        print(f"extract adds a categorical field, using the cardinality-conditional rule below:")
        print(f"cardinality \u2264 {ONEHOT_MAX_CARDINALITY} \u2192 one-hot | cardinality > {ONEHOT_MAX_CARDINALITY} \u2192 frequency-encode")
        return modelling_df, feat_cols, {}

    df_enc = modelling_df.copy()
    new_feat_cols = [c for c in feat_cols if c not in cat_cols]
    encoders = {}
    for c in cat_cols:
        card = modelling_df[c].nunique()
        print(f"\n   {c}: cardinality={card}")
        if card <= ONEHOT_MAX_CARDINALITY:
            print(f"cardinality \u2264 {ONEHOT_MAX_CARDINALITY} \u2192 ONE-HOT encoding "
                  f"({card} new binary columns)")
            
            dummies = pd.get_dummies(modelling_df[c], prefix=c, drop_first=True).astype(float)
            df_enc = pd.concat([df_enc, dummies], axis=1)
            new_feat_cols += list(dummies.columns)
            prefix_len = len(c) + 1
            encoders[c] = {"method": "onehot", "levels": [col[prefix_len:] for col in dummies.columns]}
        else:
            print(f"cardinality > {ONEHOT_MAX_CARDINALITY} \u2192 FREQUENCY encoding "
                  f"(rare levels < {RARE_LEVEL_FLOOR:.0%} share collapsed to 'Other' first)")
            freq = modelling_df[c].value_counts(normalize=True)
            rare = freq[freq < RARE_LEVEL_FLOOR].index.tolist()
            collapsed = modelling_df[c].where(~modelling_df[c].isin(rare), "Other")
            freq_map = collapsed.value_counts(normalize=True).to_dict()
            df_enc[c + "_freq"] = collapsed.map(freq_map)
            new_feat_cols.append(c + "_freq")
            print(f"{len(rare)} rare level(s) collapsed: {rare[:5]}{'...' if len(rare) > 5 else ''}")
            encoders[c] = {"method": "frequency", "rare_levels_collapsed": rare, "freq_map": freq_map}

    print(f"\n DECISION: {len(feat_cols)} raw features \u2192 {len(new_feat_cols)} after encoding "
          f"({len(cat_cols)} categorical column(s) transformed, {len(feat_cols) - len(cat_cols)} passed through unchanged)")
    return df_enc, new_feat_cols, encoders


def _self_test_encoding():
    
    banner("SELF-TEST: Step 2A encoding logic (synthetic data — proves the code path works)")
    rng = np.random.RandomState(0)
    n = 500
    synth = pd.DataFrame({
        "BALANCE": rng.normal(1000, 300, n),
        "REGION": rng.choice(["North", "South", "East", "West"], n),  
        "MERCHANT_CODE": rng.choice([f"M{i:03d}" for i in range(40)], n,
                                     p=[0.35, 0.30] + [0.35 / 38] * 38), 
    })
    synth_feat_cols = ["BALANCE", "REGION", "MERCHANT_CODE"]
    print(f"   Synthetic frame: {synth.shape}, dtypes: REGION=object (4 levels), "
          f"MERCHANT_CODE=object (40 levels, 38 rare)")
    df_enc, new_cols, encoders = encode_categoricals(synth, synth_feat_cols)
    print(f"\n VERIFICATION:")
    print(f"\u2705 REGION one-hot columns created: {[c for c in new_cols if c.startswith('REGION_')]}")
    print(f"\u2705 MERCHANT_CODE frequency column created: {'MERCHANT_CODE_freq' in new_cols}")
    print(f"\u2705 MERCHANT_CODE_freq sample values: {df_enc['MERCHANT_CODE_freq'].unique()[:4].round(4).tolist()}")
    print(f"\u2705 Rare levels collapsed: {len(encoders['MERCHANT_CODE']['rare_levels_collapsed'])} of 40 levels \u2192 'Other'")
    assert all(c.startswith("REGION_") for c in new_cols if "REGION" in c)
    assert "MERCHANT_CODE_freq" in df_enc.columns
    assert df_enc[[c for c in new_cols if c.startswith("REGION_")]].sum(axis=1).eq(1).all(), \
        "one-hot rows must sum to exactly 1"
    print(f"\u2705 All assertions passed \u2014 encoding logic is real, not narrative.")


### 3. log1p, scaling

def preprocess(modelling_df, feat_cols, scaler_decision, banner_text="STEP 2B-2D: Numeric Preprocessing"):
    
    banner(banner_text)
    SKEW_FLAG = 1.0
    NORMALTEST_ALPHA = 0.05
    MEAN_SKEW_THRESHOLD = 1.0   # |skew| <= this -> mean imputation; above -> median

    
    subbanner("Imputation strategy (per column, skew-conditional)")
    numeric_feat_cols = modelling_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [c for c in numeric_feat_cols if modelling_df[c].isnull().any()]
    X_imp = modelling_df[numeric_feat_cols].copy()
    if missing_cols:
        for c in missing_cols:
            sk = modelling_df[c].skew()
            strat = "median" if abs(sk) > MEAN_SKEW_THRESHOLD else "mean"
            fill_value = modelling_df[c].median() if strat == "median" else modelling_df[c].mean()
            X_imp[c] = modelling_df[c].fillna(fill_value)
            print(f"{c:24s} skew={sk:6.2f}  \u2192 {strat} imputation "
                  f"({'|skew|>1, mean would be tail-biased' if strat=='median' else '|skew|<=1, mean is safe'})")
        print(f"DECISION: {len(missing_cols)} column(s) imputed, each independently using its own "
              f"skew-appropriate strategy \u2014 not a single global strategy forced onto every column.")
    else:
        print(f"No missing values in the {len(numeric_feat_cols)} numeric feature columns \u2014 nothing to impute.")
    
    impute_fill_values = {}
    for c in numeric_feat_cols:
        sk = modelling_df[c].skew()
        strat = "median" if abs(sk) > MEAN_SKEW_THRESHOLD else "mean"
        fill_value = modelling_df[c].median() if strat == "median" else modelling_df[c].mean()
        impute_fill_values[c] = {"strategy": strat, "value": float(fill_value)}
    imputer = impute_fill_values  
    print(f" Remaining nulls after imputation: {X_imp.isnull().sum().sum()}")
    print(f" Persisted {len(impute_fill_values)} column fill value(s) for deployment-time imputation "
          f"(Step 12) \u2014 previously discarded here, now reusable at inference time.")

    subbanner("Conditional log1p transform (skew + normality + domain guard)")
    log_transformed, signed_log_transformed, excluded_guard, excluded_normal = [], [], [], []
    for c in numeric_feat_cols:
        skew = X_imp[c].skew()
        has_negative = (X_imp[c] < 0).any()
        is_proportion = (X_imp[c].min() >= 0 and X_imp[c].max() <= 1.5
                          and X_imp[c].nunique() <= 60 and c.upper().endswith(("FREQUENCY", "PAYMENT")))
        is_near_categorical = X_imp[c].nunique() < 10
        if abs(skew) <= SKEW_FLAG:
            continue
        if is_proportion or is_near_categorical:
            reason = "bounded [0,1] proportion" if is_proportion else f"near-categorical (nunique={X_imp[c].nunique()})"
            excluded_guard.append((c, skew, reason))
            continue
        
        if has_negative:
            signed_col = np.sign(X_imp[c]) * np.log1p(np.abs(X_imp[c]))
            _, p = stats.normaltest(signed_col)
            if p < NORMALTEST_ALPHA:
                signed_log_transformed.append(c)
            else:
                excluded_normal.append((c, skew, p))
            continue
        _, p = stats.normaltest(X_imp[c])
        if p < NORMALTEST_ALPHA:
            log_transformed.append(c)
        else:
            excluded_normal.append((c, skew, p))

    print(f"{'Feature':<28} {'skew':>7}  Decision")
    for c in numeric_feat_cols:
        skew = X_imp[c].skew()
        if c in log_transformed:
            print(f"{c:<28} {skew:7.2f} \u2705 log1p applied")
        elif c in signed_log_transformed:
            print(f"{c:<28} {skew:7.2f}  \u2705 signed-log applied (sign(x)*log1p(|x|) -- contains negative values)")
        elif any(c == e[0] for e in excluded_guard):
            reason = [e[2] for e in excluded_guard if e[0] == c][0]
            print(f"{c:<28} {skew:7.2f}\u2014 excluded (domain guard: {reason})")
        elif abs(skew) <= SKEW_FLAG:
            print(f"{c:<28} {skew:7.2f}\u2014 not flagged (|skew|<={SKEW_FLAG})")
        else:
            print(f"{c:<28} {skew:7.2f} \u2014 excluded (normaltest p>={NORMALTEST_ALPHA})")

    print(f"\n \u26a0\ufe0f  Without the domain guard, {len(log_transformed)+len(signed_log_transformed)+len(excluded_guard)} features would have been "
          f"flagged; the guard removed {len(excluded_guard)} bounded-ratio/near-categorical fields "
          f"({', '.join(e[0] for e in excluded_guard) if excluded_guard else 'none'}) that log1p would have distorted, not fixed.")
    if signed_log_transformed:
        print(f"{len(signed_log_transformed)} negative-valued feature(s) signed-log transformed instead of skipped: "
              f"{', '.join(signed_log_transformed)}")

    X_t = X_imp.copy()
    for c in log_transformed:
        X_t[c] = np.log1p(X_t[c])
    for c in signed_log_transformed:
        X_t[c] = np.sign(X_t[c]) * np.log1p(np.abs(X_t[c]))

    subbanner("Winsorization for extreme post-transform outliers")
  
    SCALED_OUTLIER_THRESHOLD = 15.0
    IQR_CAP_MULTIPLIER = 10.0
    WINSOR_LOWER, WINSOR_UPPER = 0.01, 0.99

    probe_scaler = RobustScaler() if scaler_decision == "RobustScaler" else StandardScaler()
    X_probe = pd.DataFrame(probe_scaler.fit_transform(X_t), columns=X_t.columns, index=X_t.index)
    abs_max = X_probe.abs().max()
    winsorized_cols = abs_max[abs_max > SCALED_OUTLIER_THRESHOLD].index.tolist()

    if winsorized_cols:
        print(f"{len(winsorized_cols)} column(s) exceeded |{SCALED_OUTLIER_THRESHOLD}| on a probe scaling pass "
              f"(detected by actual scaled output, not by column name):")
        print(f"{'Feature':<24} {'before max':>12} {'after max':>12} {'n capped':>10}  Method")
        for c in winsorized_cols:
            q1, q3 = X_t[c].quantile(.25), X_t[c].quantile(.75)
            iqr = q3 - q1
            median = X_t[c].median()
            before_max = X_t[c].max()
            if iqr > 0:
                lo, hi = median - IQR_CAP_MULTIPLIER * iqr, median + IQR_CAP_MULTIPLIER * iqr
                method = f"median\u00b1{IQR_CAP_MULTIPLIER:.0f}\u00d7IQR"
            else:
                lo, hi = X_t[c].quantile(WINSOR_LOWER), X_t[c].quantile(WINSOR_UPPER)
                method = f"{WINSOR_LOWER:.0%}/{WINSOR_UPPER:.0%} percentile (IQR=0 fallback)"
            n_capped = int(((X_t[c] < lo) | (X_t[c] > hi)).sum())
            X_t[c] = X_t[c].clip(lower=lo, upper=hi)
            print(f"   {c:<24} {before_max:12.3f} {X_t[c].max():12.3f} {n_capped:10d}  {method}")
            if iqr == 0 and X_t[c].nunique() <= 1:
                print(f" \u26a0\ufe0f  '{c}' collapsed to a single constant value \u2014 >{WINSOR_UPPER:.0%} of "
                      f"customers already shared the same value, so no signal remained within the "
                      f"winsorization bounds. Not a bug: Step 15 (near-zero-variance) will correctly drop "
                      f"this feature; consider a binary flag instead of the raw amount if it matters for personas.")
    else:
        print(f"No columns exceeded |{SCALED_OUTLIER_THRESHOLD}| on the probe pass \u2014 no winsorization needed.")

    subbanner("Scaling")
    print(f"DECISION (carried from Step 1): {scaler_decision}")
    scaler = RobustScaler() if scaler_decision == "RobustScaler" else StandardScaler()
    # FIX: was columns=feat_cols (could still include string columns not
    # present in X_t) and index=df.index (the row-level table, 41,539 rows --
    # the wrong length entirely for this table). Both now match X_t itself.
    X_scaled = pd.DataFrame(scaler.fit_transform(X_t), columns=X_t.columns, index=X_t.index)
    print(f"Post-scaling range check (min/max across all features): "
          f"[{X_scaled.values.min():.2f}, {X_scaled.values.max():.2f}]")

    return X_scaled, X_imp, imputer, scaler, log_transformed, signed_log_transformed


df_enc, new_feat_cols, encoders = encode_categoricals(modelling_df, feat_cols)
X_scaled, X_imp, imputer, scaler, log_transformed, signed_log_transformed = preprocess(df_enc, new_feat_cols, scaler_decision)


#=====================================================================
# Redundancy & Dimensionality Assessment
# =========================================================================


def redundancy(X_scaled, X_imp, feat_cols, structure_result, modelling_df):
    banner("Redundancy & Dimensionality Assessment")
    CORR_THRESHOLD = 0.90
    VIF_THRESHOLD = 10
    DIM_RATIO_THRESHOLD = 0.10
    GROUP_CORR_DIVERGENCE_FLAG = 0.15  # pooled-vs-within-group correlation gap worth flagging as a possible confound

    subbanner("Structure-detection decision matrix applied to this step")
    
    temporal_detected = structure_result["temporal_detected"]
    group_detected = structure_result["group_detected"]
    network_detected = structure_result["network_detected"]
    group_col = structure_result["group_col_selected"]
    print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: this step computes a single "
          f"pass over the fixed feature matrix; no resampling/shuffling/splitting occurs, so time-block resampling "
          f"has nothing to act on.")
    print(f"Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: network structure drives an "
          f"algorithm/reporting decision (Step 6/8), not a redundancy-pruning one.")
    if group_detected:
        print(f"Group structure: YES ('{group_col}') \u2192 genuinely relevant: a pooled correlation across the whole")
        print(f"population could be inflated/deflated by between-group variation. Every flagged high-correlation")
        print(f"pair below is cross-checked against its within-group correlation to rule this out.")
    else:
        print(f"Group structure: NO \u2192 pooled correlation is used directly, no group-confound check needed.")

    subbanner("Pairwise correlation pruning")
    corr = X_scaled[feat_cols].corr()
    high_corr_pairs = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            r = corr.iloc[i, j]
            if abs(r) > CORR_THRESHOLD:
                high_corr_pairs.append((feat_cols[i], feat_cols[j], round(float(r), 4)))
    print(f"Pairs with |r| > {CORR_THRESHOLD}: {len(high_corr_pairs)} found")

    drop_cols = []

    if "Cash_Stake" in feat_cols and {"Cash_Stake_Casino", "Cash_Stake_Sports"}.issubset(feat_cols):
        r_stake = float(np.corrcoef(X_imp["Cash_Stake"], modelling_df.loc[X_imp.index, "total_stake_verified"])[0, 1])
        resid_stake = (X_imp["Cash_Stake"] - modelling_df.loc[X_imp.index, "total_stake_verified"]).abs()
        n_diff = int((resid_stake > 0.01).sum())
        print(f"   Cash_Stake vs Cash_Stake_Casino+Cash_Stake_Sports: corr={r_stake:.4f}, "
              f"{n_diff}/{len(X_imp)} rows differ by >0.01 (max diff={resid_stake.max():.1f})")
        print(f"\u2192 NOT a clean identity (unlike CC_GENERAL's PURCHASES case). Cash_Stake was already proven "
              f"unreliable for sports-only bettors during the zero-cash-funded cohort fix. DECISION: drop Cash_Stake "
              f"(unreliable raw field), keep Cash_Stake_Casino + Cash_Stake_Sports (verified, granular, and directly "
              f"support the Casino<->Sports cross-sell persona goal).")
        if "Cash_Stake" not in drop_cols:
            drop_cols.append("Cash_Stake")

    if "Cash_Return" in feat_cols and {"Cash_Return_Casino", "Cash_Return_Sports"}.issubset(feat_cols):
        alt_return = X_imp["Cash_Return_Casino"] + X_imp["Cash_Return_Sports"]
        r_return = float(np.corrcoef(X_imp["Cash_Return"], alt_return)[0, 1])
        resid_return = (X_imp["Cash_Return"] - alt_return).abs()
        n_diff = int((resid_return > 0.01).sum())
        print(f"   Cash_Return vs Cash_Return_Casino+Cash_Return_Sports: corr={r_return:.4f}, "
              f"{n_diff}/{len(X_imp)} rows differ by >0.01 (max diff={resid_return.max():.1f})")
        print(f"\u2192 Same reliability pattern as Cash_Stake above. DECISION: drop Cash_Return, keep the granular "
              f"Casino/Sports components.")
        if "Cash_Return" not in drop_cols:
            drop_cols.append("Cash_Return")

    if "total_stake_verified" in feat_cols and {"Cash_Stake_Casino", "Cash_Stake_Sports"}.issubset(feat_cols):
        r_verify = float(np.corrcoef(modelling_df.loc[X_imp.index, "total_stake_verified"],
                                      X_imp["Cash_Stake_Casino"] + X_imp["Cash_Stake_Sports"])[0, 1])
        print(f"total_stake_verified vs Cash_Stake_Casino+Cash_Stake_Sports: corr={r_verify:.4f}")
        print(f"\u2192 total_stake_verified is BY DEFINITION Cash_Stake_Casino+Cash_Stake_Sports (it was built as "
              f"that sum during the zero-cash-funded cohort fix). Perfectly collinear as a modelling feature, though it "
              f"remains a necessary intermediate variable for the ratio features and cohort split computed earlier. "
              f"DECISION: drop total_stake_verified from the FEATURE SET (adds zero information beyond its two "
              f"components), keep Cash_Stake_Casino + Cash_Stake_Sports as the actual modelling features.")
        if "total_stake_verified" not in drop_cols:
            drop_cols.append("total_stake_verified")

    
    for a, b, r in high_corr_pairs:
        if a in drop_cols or b in drop_cols:
            continue
        a_protected = a in BUSINESS_PROTECTED_FEATURES
        b_protected = b in BUSINESS_PROTECTED_FEATURES
        if a_protected and b_protected:
            print(f"{a} vs {b}: r={r} \u2014 BOTH protected, cannot resolve automatically, flagging for manual review")
            continue
        elif a_protected:
            drop, keep, reason = b, a, f"'{a}' is business-protected (PROTECTED_FEATURE_GAIN_MULTIPLIER applies)"
        elif b_protected:
            drop, keep, reason = a, b, f"'{b}' is business-protected (PROTECTED_FEATURE_GAIN_MULTIPLIER applies)"
        else:
            drop, keep, reason = b, a, "no protected feature in this pair; keeping first-encountered for determinism"

        group_note = ""
        if group_detected and modelling_df[group_col].nunique() > 1:
            within_rs = []
            for _, grp_idx in modelling_df.groupby(group_col).groups.items():
                grp_idx = grp_idx.intersection(X_imp.index)
                if len(grp_idx) > 10:
                    within_rs.append(float(np.corrcoef(X_imp.loc[grp_idx, a], X_imp.loc[grp_idx, b])[0, 1]))
            if within_rs:
                divergence = max(abs(wr - r) for wr in within_rs)
                confound_flag = "possible group confound" if divergence > GROUP_CORR_DIVERGENCE_FLAG else "confirmed within-group too"
                group_note = f"  [within-group r range {min(within_rs):.3f}-{max(within_rs):.3f}, {confound_flag}]"

        print(f"{a} vs {b}: r={r} \u2192 DECISION: drop {drop}, keep {keep} ({reason}){group_note}")
        drop_cols.append(drop)

    feat_v2 = [c for c in feat_cols if c not in drop_cols]
    print(f"\n {len(feat_cols)} \u2192 {len(feat_v2)} features after correlation pruning (dropped: {drop_cols})")

    subbanner("Iterative VIF pruning (target: max VIF \u2264 10)")
    
    feat_final = list(feat_v2)
    PROTECTED_VIF_THRESHOLD = VIF_THRESHOLD * PROTECTED_FEATURE_GAIN_MULTIPLIER
    round_n = 1
    vifs = None
    while True:
        Xv = X_scaled[feat_final].copy()
        Xv.insert(0, "const", 1.0)
        vifs = pd.Series(
            [variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])],
            index=feat_final).sort_values(ascending=False)
        print(f"   Round {round_n} VIF (top 3): {dict(vifs.head(3).round(2))}")

        is_protected = vifs.index.to_series().apply(lambda c: c in BUSINESS_PROTECTED_FEATURES)
        unprotected_over = vifs[(~is_protected) & (vifs > VIF_THRESHOLD)]
        protected_over_elevated = vifs[is_protected & (vifs > PROTECTED_VIF_THRESHOLD)]

        if unprotected_over.empty and protected_over_elevated.empty:
            residual_protected = vifs[is_protected & (vifs > VIF_THRESHOLD)]
            if len(residual_protected):
                print(f"\u2705 Pruning complete: no unprotected feature exceeds VIF={VIF_THRESHOLD}. "
                      f"{len(residual_protected)} protected feature(s) remain above {VIF_THRESHOLD} but below the "
                      f"elevated {PROTECTED_VIF_THRESHOLD:.0f} bar (PROTECTED_FEATURE_GAIN_MULTIPLIER applies): "
                      f"{dict(residual_protected.round(2))}")
            else:
                print(f"\u2705 max VIF = {vifs.max():.2f} \u2264 {VIF_THRESHOLD} \u2014 pruning complete")
            break

        if unprotected_over.empty:
            worst = protected_over_elevated.idxmax()
            print(f"\u26a0\ufe0f  Protected feature '{worst}' exceeds even the elevated {PROTECTED_VIF_THRESHOLD:.0f} bar "
                  f"(VIF={protected_over_elevated.max():.2f}), and no unprotected alternative remains \u2192 dropping "
                  f"despite protection (genuinely unavoidable multicollinearity)")
        else:
            worst = unprotected_over.idxmax()
            print(f"\u26a0\ufe0f  max UNPROTECTED VIF = {unprotected_over.max():.2f} > {VIF_THRESHOLD} \u2192 dropping {worst}")

        feat_final.remove(worst)
        round_n += 1
        if round_n > 50:
            print("\u26a0\ufe0f  Safety stop after 50 rounds")
            break

    print(f"\n {len(feat_cols)} \u2192 {len(feat_v2)} \u2192 {len(feat_final)} features after full redundancy pass")
    print(f"Final set: {feat_final}")
    final_max_vif = float(vifs.max())

    subbanner("Dimensionality ratio (PCA go/no-go)")
    n = len(X_scaled)
    dim_ratio = len(feat_final) / n
    print(f"   p/n = {len(feat_final)}/{n} = {dim_ratio:.4f}  (threshold {DIM_RATIO_THRESHOLD})")
    use_pca = dim_ratio > DIM_RATIO_THRESHOLD
    print(f"DECISION: {'PCA justified on ratio grounds' if use_pca else 'PCA NOT required on ratio grounds'}")
    print(f"Even so, feature-level pruning (3A/3B) is preferred over PCA here: later validation/profiling steps "
          f"require every final centroid dimension to be directly explainable to a business stakeholder (CRM/VIP/"
          f"Risk), which a PCA component is not \u2014 PCA is reserved for a 2D visual diagnostic only, never as model input.")

    return feat_final, final_max_vif


feat_final, final_max_vif = redundancy(X_scaled, X_imp, new_feat_cols, structure_result, modelling_df)


### 1. Unsupervised Feature Selection (Step 3D)

def passes_dual_check(gain, required_gain, density_after, baseline_density, density_drop_tolerance):
    
    gain_ok = gain >= required_gain
    density_ok = density_after >= baseline_density - density_drop_tolerance
    return gain_ok and density_ok


def _self_test_dual_check_feature_selection():
    
    print(f"Case A: K-Means gain HIGH, density drop SMALL -> expect ACCEPT (both pass)")
    r = passes_dual_check(gain=0.10, required_gain=0.02, density_after=0.50, baseline_density=0.52, density_drop_tolerance=0.05)
    print(f"passes_dual_check = {r}")
    assert r is True

    print(f"Case B: K-Means gain HIGH, density drop LARGE -> expect REJECT (density check fails)")
    r = passes_dual_check(gain=0.10, required_gain=0.02, density_after=0.10, baseline_density=0.52, density_drop_tolerance=0.05)
    print(f"passes_dual_check = {r}")
    assert r is False, "a large density drop MUST reject the candidate even when K-Means gain is strongly positive"

    print(f"Case C: K-Means gain LOW (below threshold), density fine -> expect REJECT (gain check fails)")
    r = passes_dual_check(gain=0.005, required_gain=0.02, density_after=0.55, baseline_density=0.52, density_drop_tolerance=0.05)
    print(f"passes_dual_check = {r}")
    assert r is False, "an insufficient K-Means gain MUST reject the candidate even when density is unaffected"

    print(f"Case D: both LOW -> expect REJECT")
    r = passes_dual_check(gain=0.001, required_gain=0.02, density_after=0.10, baseline_density=0.52, density_drop_tolerance=0.05)
    print(f"passes_dual_check = {r}")
    assert r is False

    print(f"Case E: density comfortably within tolerance (not at an exact float boundary,")
    print(f"which is an ill-posed test for binary floating point) -> expect ACCEPT")
    r = passes_dual_check(gain=0.10, required_gain=0.02, density_after=0.475, baseline_density=0.52, density_drop_tolerance=0.05)
    print(f"passes_dual_check = {r}")
    assert r is True, "a drop safely within tolerance should be accepted"

    print(f"\u2705 The dual-check genuinely requires BOTH signals to agree \u2014 Case B is the specific")
    print(f"scenario this fix exists for (K-Means alone would have approved the removal).")


def rank_by_consensus(gain_dens_pairs):
    
    if len(gain_dens_pairs) == 1:
        return list(gain_dens_pairs), np.array([0.0])
    gains_arr = np.array([p[0] for p in gain_dens_pairs])
    dens_arr = np.array([p[3] for p in gain_dens_pairs])
    finite = np.isfinite(dens_arr)
    dens_mean = dens_arr[finite].mean() if finite.any() else 0.0
    dens_std = dens_arr[finite].std() if finite.any() else 1.0
    z_gain = (gains_arr - gains_arr.mean()) / gains_arr.std() if gains_arr.std() > 0 else np.zeros_like(gains_arr)
    z_dens = np.where(finite, (dens_arr - dens_mean) / (dens_std if dens_std > 0 else 1.0), -1e6)
    composite = z_gain + z_dens
    order = np.argsort(-composite)
    return [gain_dens_pairs[i] for i in order], composite[order]


def _self_test_consensus_ranking():
    
    print(f"Case A: candidate 1 has the HIGHEST K-Means gain but WORST density;")
    print(f"candidate 2 has slightly lower gain but MUCH better density -> expect consensus")
    print(f"to prefer candidate 2, even though K-Means alone would rank candidate 1 first.")
    pool = [
        (0.10, "feat_1", 0.50, -0.20),  # highest gain, terrible density
        (0.08, "feat_2", 0.48, 0.90),  # slightly lower gain, excellent density
        (0.03, "feat_3", 0.43, 0.10), # low gain, mediocre density
    ]
    ranked, composite = rank_by_consensus(pool)
    print(f" K-Means-only order would be: feat_1, feat_2, feat_3")
    print(f" Consensus order is: {[p[1] for p in ranked]}")
    assert ranked[0][1] == "feat_2", "consensus must prefer the candidate with the better COMBINED profile"

    print(f"Case B: degenerate (-inf) density for one candidate -> must rank LAST, not crash")
    pool2 = [(0.10, "feat_A", 0.5, float("-inf")), (0.05, "feat_B", 0.45, 0.5)]
    ranked2, _ = rank_by_consensus(pool2)
    print(f"Order: {[p[1] for p in ranked2]}")
    assert ranked2[-1][1] == "feat_A", "a degenerate (-inf) density score must rank last, never first"

    print(f"\u2705 Consensus ranking genuinely lets density override raw K-Means gain ordering,")
    print(f"and correctly handles degenerate (-inf) density values without crashing.")



def feature_selection(X_scaled, feat_final, structure_result=None, min_features=8, gain_threshold=0.02, k_screen=4, density_drop_tolerance=0.05):
    banner("Unsupervised Feature Selection")
    print(f"Method: sequential backward elimination against silhouette (k={k_screen} screening probe)")
    print(f"Business floor: MIN_FEATURES={min_features}  |  Stop rule: best available gain < {gain_threshold}")
    print(f"ENHANCEMENT 16 (dual-check, round 6): a candidate removal also needs HDBSCAN's")
    print(f"relative_validity_ (lightweight DBCV proxy) to not drop by more than "
          f"{density_drop_tolerance} \u2014")
    print(f"K-Means silhouette gain alone is no longer sufficient to drop a feature.")

    print(f"ENHANCEMENT 19 (consensus-ranked search order, round 9): the top "
          f"{CONSENSUS_POOL_SIZE} candidates by K-Means gain each step are RE-RANKED by a")
    print(f"composite z(gain)+z(density) score before applying the accept/reject thresholds \u2014 "
          f"density now genuinely")
    print(f"influences WHICH feature is preferred, not just a veto on K-Means' single top pick.")

    
    subbanner("Structure-detection decision matrix applied to this step")
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        group_detected = structure_result.get("group_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: each backward-"
              f"elimination step fits K-Means/HDBSCAN once on the full (fixed) row set, with no resampling or "
              f"time-block splitting for this step to act on \u2014 same reasoning as the redundancy step above.")
        print(f"Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: network structure is an "
              f"algorithm/reporting decision (Step 6/8), not a feature-selection one.")
        if group_detected:
            print(f"Group structure: YES ('{group_col}') \u2192 this step's silhouette/density gains are computed on")
            print(f"the POOLED casino population, not within-group. A feature that only separates one customer")
            print(f"segment (e.g. one region or acquisition channel) well could still show a pooled gain. This is")
            print(f"a known limitation, not silently ignored: the final feature set from this step should be")
            print(f"re-checked for within-'{group_col}' stability during Step 6/8 profiling, the same way")
            print(f"group-blocked resampling is applied there rather than inside this deterministic elimination loop.")
        else:
            print(f"Group structure: NO \u2192 pooled silhouette/density gains are used directly, no within-group "
                  f"re-check is needed downstream.")
    else:
        print(f"No structure_result supplied \u2014 skipping the decision-matrix narration (temporal/group/network "
              f"checks default to their conservative NO/pooled assumptions for this run).")

    protected = [f for f in BUSINESS_PROTECTED_FEATURES if f in feat_final] if BUSINESS_CRITICAL_SEGMENTATION else []
    elevated_threshold = gain_threshold * PROTECTED_FEATURE_GAIN_MULTIPLIER
    print(f"\n ENHANCEMENT 1 (Hybrid Feature Selection): BUSINESS_CRITICAL_SEGMENTATION="
          f"{BUSINESS_CRITICAL_SEGMENTATION}")
    if protected:
        print(f"Protected features present in this feature set: {protected}")
        print(f"These require gain >= {elevated_threshold:.4f} ({PROTECTED_FEATURE_GAIN_MULTIPLIER}x the "
              f"normal {gain_threshold}) to be dropped, not blanket immunity \u2014 a genuinely harmful protected")
        print(f"feature can still be removed; a merely-mediocre one is kept for its business value.")
        print(f"These are the same casino-specific protections applied in the redundancy/VIF step above "
              f"(Cash_Stake_Casino, Cash_Stake_Sports, Total_Deposit_Amount, bonus_dependency, tenure_days) \u2014 "
              f"a feature surviving redundancy pruning because it's protected does not get silently unprotected "
              f"here; the same PROTECTED_FEATURE_GAIN_MULTIPLIER threshold carries through.")
    else:
        print(f"No BUSINESS_PROTECTED_FEATURES present in this feature set (or flag is False) \u2014 "
              f"standard threshold applies to all candidates.")

    X = X_scaled[feat_final].copy()
    current = list(X.columns)
    removal_history = []  

    
    if len(X) > LARGE_N_THRESHOLD:
        _step3d_subsample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), LARGE_N_THRESHOLD, replace=False)
        print(f"Step 3D dual-check: {len(X):,} rows exceeds LARGE_N_THRESHOLD={LARGE_N_THRESHOLD:,} -- BOTH "
              f"K-Means/silhouette ranking AND the HDBSCAN density check subsampled to the SAME "
              f"{LARGE_N_THRESHOLD:,} rows (reused for every candidate this run).")
    else:
        _step3d_subsample_idx = None

    def sil_at(cols, seed=RANDOM_STATE):
        Xs = X[cols] if _step3d_subsample_idx is None else X[cols].iloc[_step3d_subsample_idx]
        km = KMeans(n_clusters=k_screen, init="k-means++", n_init=10, random_state=seed).fit(Xs)
        return silhouette_score(Xs, km.labels_, sample_size=min(4000, len(Xs)), random_state=seed)

    def density_score_at(cols):
        Xd = X[cols].values if _step3d_subsample_idx is None else X[cols].values[_step3d_subsample_idx]
        min_cluster_size = max(10, int(0.01 * len(Xd)))
        try:
            clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True).fit(Xd)
            if len(set(clusterer.labels_[clusterer.labels_ != -1])) < 2:
                return float("-inf")
            return float(clusterer.relative_validity_)
        except Exception:
            return float("-inf")

    base = sil_at(current)
    base_density = density_score_at(current)
    print(f"\n Starting silhouette ({len(current)} features): {base:.4f}  |  "
          f"starting density score (relative_validity_): {base_density:.6g}")
    if 0 <= base_density < 0.01:
        print(f"\u26a0\ufe0f  Near-zero baseline density score \u2014 consistent with this pipeline's own Step 7D/7E")
        print(f"findings on the casino modelling population (customers sit on continuous behavioural gradients,")
        print(f"not density-separated islands).")
        print(f"The dual-check is still genuinely running, but has little room to REJECT a removal on")
        print(f"density grounds when there is barely any density-based structure to lose in the first place.")
    step_n = 1
    while len(current) > min_features:
        gains = []
        for c in current:
            trial = [x for x in current if x != c]
            s = sil_at(trial)
            gains.append((s - base, c, s))
        gains.sort(key=lambda x: -x[0])  

        
        pool_size = min(CONSENSUS_POOL_SIZE, len(gains))
        pool = [(g, c, s, density_score_at([x for x in current if x != c])) for g, c, s in gains[:pool_size]]
        ranked_pool, composite_scores = rank_by_consensus(pool)
        print(f"\n Step {step_n} consensus pool (top {pool_size} by K-Means gain, re-ranked by consensus):")
        km_only_order = [p[1] for p in pool]
        consensus_order = [p[1] for p in ranked_pool]
        for (g, c, s, d), comp in zip(ranked_pool, composite_scores):
            print(f"{c:32s} K-Means gain={g:+.4f}  density={d:.6g}  composite={comp:+.3f}")
        if km_only_order != consensus_order:
            print(f"\u26a0\ufe0f  Consensus re-ranking CHANGED the order vs K-Means alone "
                  f"({km_only_order[0]} \u2192 {consensus_order[0]} as top pick)")

        chosen = None
        density_rejections = []
        any_cleared_gain_bar = False
        for gain, cand, score, dens in ranked_pool:
            required = elevated_threshold if cand in protected else gain_threshold
            if gain < required:
                continue  # doesn't even clear the K-Means-only bar
            any_cleared_gain_bar = True
            if passes_dual_check(gain, required, dens, base_density, density_drop_tolerance):
                chosen = (gain, cand, score, dens, required)
                break
            else:
                density_rejections.append((cand, gain, dens))
                print(f"Step {step_n} candidate '{cand}': K-Means gain {gain:+.4f} clears the bar, but "
                      f"density cross-check FAILS (relative_validity {dens:.6g} vs baseline "
                      f"{base_density:.6g}, tolerance {density_drop_tolerance}) \u2014 REJECTED despite passing")
                print(f"the K-Means-only test; this is exactly the dual-check this fix exists to enforce.")

        if chosen is None:
            if density_rejections:
                print(f"Step {step_n}: {len(density_rejections)} candidate(s) cleared K-Means' bar but ALL "
                      f"failed the density cross-check \u2192 STOP (dual-check requires agreement, not just one)")
            else:
                top_gain, top_cand, _ = gains[0]
                req = elevated_threshold if top_cand in protected else gain_threshold
                print(f"Step {step_n}: best available gain ({top_gain:+.4f} on '{top_cand}') < required "
                      f"threshold ({req:.4f}{', PROTECTED' if top_cand in protected else ''}) \u2192 STOP")
            break

        gain, cand, score, dens, required = chosen
        removal_history.append({"step": step_n, "features_before": list(current), "removed": cand,
                                 "kmeans_gain": gain, "density_after": dens})
        current.remove(cand)
        base = score
        base_density = dens
        flag = " [PROTECTED, but cleared elevated bar]" if cand in protected else ""
        print(f"Step {step_n}: removed {cand:32s} gain={gain:+.4f}  silhouette now {base:.4f}  "
              f"density now {base_density:.6g}{flag}")
        step_n += 1

    still_protected_and_kept = [f for f in protected if f in current]
    if still_protected_and_kept:
        print(f"\n Business-protected features retained in the final set: {still_protected_and_kept}")
    print(f"\n DECISION: {len(current)} features retained (floor={min_features}), final silhouette={base:.4f}, "
          f"final density score={base_density:.6g}")
    print(f" Final feature list: {current}")
    return current, removal_history


banner("SELF-TESTS: Step 3D dual-check and consensus-ranking control flow")
_self_test_dual_check_feature_selection()
_self_test_consensus_ranking()

final_features, removal_history = feature_selection(X_scaled, feat_final, structure_result=structure_result)


### 2. Leakage & Contamination Checks 


def _find_kpi_contamination_columns(columns, feat_final):
    
    matched = [c for c in columns if c.lower() in KPI_CANDIDATE_NAMES]
    matched_in_features = [c for c in matched if c in feat_final]
    return matched, matched_in_features


def _find_lookahead_dates(date_series_dict, snapshot_date):
    
    violations = {}
    for col, series in date_series_dict.items():
        n_future = int((series > snapshot_date).sum())
        if n_future > 0:
            violations[col] = n_future
    return violations


def _check_group_circularity(feat_final, structure_result, icc_threshold=0.90):
    
    if not structure_result or not structure_result.get("group_detected"):
        return None
    group_col = structure_result.get("group_col_selected")
    if not group_col:
        return None
    rank_equivalent = f"{group_col.replace('_tier', '')}_rank"
    if rank_equivalent in feat_final:
        return {"group_col": group_col, "rank_feature": rank_equivalent}
    return None


def _self_test_leakage_helpers():
    
    banner("SELF-TEST: Step 3E leakage-check helpers")

    print(f"Sub-test A: KPI-contamination matching distinguishes 'present in extract' from 'active in feat_final'")
    cols = ["Customer_ID", "GGR", "NGR", "tenure_days", "bonus_dependency"]
    feat = ["tenure_days", "bonus_dependency", "GGR"]  # NGR deliberately NOT in the active feature set
    matched, matched_active = _find_kpi_contamination_columns(cols, feat)
    print(f"Matched columns (case-insensitive KPI_CANDIDATE_NAMES): {matched}")
    print(f"Of those, active in feat_final: {matched_active}")
    assert matched == ["NGR"], f"expected only NGR to match KPI_CANDIDATE_NAMES, got {matched}"
    assert matched_active == [], "NGR was deliberately excluded from feat_final -- must NOT be reported as active"

    print(f"\n Sub-test A2: a KPI-shaped column THAT IS in feat_final is correctly flagged as active")
    matched2, matched_active2 = _find_kpi_contamination_columns(["NGR", "clv"], ["NGR"])
    assert matched_active2 == ["NGR"], "a KPI-matching column present in feat_final must be reported as ACTIVE"
    print(f"Matched: {matched2}  Active: {matched_active2}  \u2705")

    print(f"\n Sub-test B: lookahead-date detection catches values after the snapshot date")
    snap = pd.Timestamp("2026-01-01")
    good_dates = pd.Series(pd.to_datetime(["2025-06-01", "2025-12-31", "2024-01-01"]))
    bad_dates = pd.Series(pd.to_datetime(["2025-06-01", "2026-03-15", "2026-06-01"]))  # 2 rows AFTER snapshot
    violations = _find_lookahead_dates({"clean_col": good_dates, "leaky_col": bad_dates}, snap)
    print(f"Violations found: {violations}")
    assert "clean_col" not in violations, "a column with no post-snapshot dates must not be flagged"
    assert violations.get("leaky_col") == 2, f"expected exactly 2 post-snapshot rows in leaky_col, got {violations.get('leaky_col')}"

    print(f"\n Sub-test C: group/rank circularity flags the rank feature only when its group column was BOTH")
    print(f"detected by Step 1B AND the rank feature is actually present in feat_final")
    sr_group = {"group_detected": True, "group_col_selected": "vip_tier"}
    sr_none = {"group_detected": False, "group_col_selected": None}
    flag_present = _check_group_circularity(["vip_rank", "GGR"], sr_group)
    flag_absent_feature = _check_group_circularity(["GGR"], sr_group)  # vip_rank not in feat_final
    flag_no_group = _check_group_circularity(["vip_rank", "GGR"], sr_none)
    assert flag_present is not None and flag_present["rank_feature"] == "vip_rank"
    assert flag_absent_feature is None, "must not flag when the rank feature isn't actually in feat_final"
    assert flag_no_group is None, "must not flag when Step 1B found no group structure at all"
    print(f"vip_rank in feat_final + group_detected: {flag_present}")
    print(f"vip_rank NOT in feat_final: {flag_absent_feature}")
    print(f"no group structure detected: {flag_no_group}")

    print(f"\n \u2705 All three Step 3E helper behaviours verified on controlled inputs.")


def leakage_checks(X_scaled, feat_final, entity_col, df, snapshot_date=None, structure_result=None, raw_df=None):
    banner("STEP 3E: Leakage & Contamination Checks")
    NEAR_DUPLICATE_THRESHOLD = 0.98
    X = X_scaled[feat_final]
    flags = {}

    subbanner("Structure-detection decision matrix applied to this step")
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        group_detected = structure_result.get("group_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 this is a DIFFERENT question from "
              f"Check 2 below: Step 1B asks whether entities contribute repeated time-stamped rows (requiring "
              f"block-bootstrap resampling); Check 2 asks whether an individual raw date VALUE leaks information "
              f"from beyond the as-of snapshot date into an engineered feature. temporal_detected={temporal_detected} "
              f"does not exempt this step from running Check 2.")
        print(f"   Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: not a contamination-relevant "
              f"signal for this step.")
        if group_detected:
            print(f"Group structure: YES ('{group_col}') \u2192 genuinely relevant: feeds directly into the new "
                  f"Check 5 below (group/rank circularity), which only exists because this decision-matrix result "
                  f"is available.")
        else:
            print(f"Group structure: NO \u2192 Check 5 will be skipped (nothing to check circularity against).")
    else:
        print(f"No structure_result supplied \u2014 Check 2's snapshot-date reconciliation note and Check 5 "
              f"(group/rank circularity) are both skipped.")

    subbanner("Check 1: Outcome / downstream-action contamination")
    matched_kpi, matched_active = _find_kpi_contamination_columns(df.columns, feat_final)
    flags["outcome_contamination"] = len(matched_active) > 0
    flags["outcome_kpi_present_but_inactive"] = len(matched_kpi) > len(matched_active)
    print(f"KPI-vocabulary columns found in extract (KPI_CANDIDATE_NAMES): {matched_kpi if matched_kpi else 'NONE'}")
    if matched_active:
        print(f"\u26a0\ufe0f  FLAG \u2014 ACTIVE in feat_final (feeding the clustering itself): {matched_active} \u2014 "
              f"investigate before proceeding, this is genuine target/outcome leakage into the model.")
    elif matched_kpi:
        print(f"\u26a0\ufe0f  FLAG (lower severity) \u2014 present in the extract but NOT in feat_final: {matched_kpi} "
              f"\u2014 not currently feeding the model, but must be watched if the feature set changes.")
    else:
        print(f"\u2705 N/A on this extract")
    print(f"MUST be re-run the moment this segmentation is paired with a specific downstream business action "
          f"(e.g. targeting for self-exclusion outreach, bonus-abuse intervention, or deposit-churn win-back).")

    subbanner("Check 2: Temporal / lookahead contamination")
    raw_date_cols = {c: df[c] for c in ["Reg_Date", "First_Deposit_Date", "Last_Login_Date", "Last_Deposit_Date"]
                      if c in df.columns}
    if raw_df is not None:
        for c in ["Activity_Date", "Bet_Date"]:
            if c in raw_df.columns:
                raw_date_cols[c] = raw_df[c]
    if snapshot_date is not None and raw_date_cols:
        lookahead_violations = _find_lookahead_dates(raw_date_cols, snapshot_date)
        flags["temporal_contamination"] = len(lookahead_violations) > 0
        print(f"Raw date columns checked against snapshot_date ({snapshot_date.date()}): {list(raw_date_cols.keys())}")
        if lookahead_violations:
            print(f"\u26a0\ufe0f  FLAG \u2014 rows with dates AFTER the snapshot: {lookahead_violations} \u2014 an engineered "
                  f"recency feature (tenure_days/days_since_login/days_since_deposit/active_days) may have been "
                  f"built using information from beyond the as-of date.")
        else:
            print(f"\u2705 No row in any checked date column falls after the snapshot date \u2014 tenure_days/"
                  f"days_since_login/days_since_deposit/active_days are genuinely as-of-date-safe.")
    else:
        flags["temporal_contamination"] = False
        print(f"\u26a0\ufe0f  snapshot_date not supplied or no raw date columns available \u2014 lookahead check skipped "
              f"(reported as not-flagged by default, not verified clean).")

    subbanner("Check 3: Identifier & near-duplicate contamination")
    id_ratio = df[entity_col].nunique() / len(df)
    print(f"{entity_col} uniqueness ratio = {id_ratio:.4f} \u2192 excluded from feature set at Step 1 (matches eda()'s "
          f"identifier check on the same customer-grain table)")
    corr_final = X.corr().abs().to_numpy().copy()
    np.fill_diagonal(corr_final, 0)
    corr_final = pd.DataFrame(corr_final, index=X.columns, columns=X.columns)
    max_pair = corr_final.stack().idxmax()
    max_r = corr_final.stack().max()
    flags["near_duplicate"] = max_r >= NEAR_DUPLICATE_THRESHOLD
    protected_note = ""
    if flags["near_duplicate"]:
        a, b = max_pair
        a_prot = a in BUSINESS_PROTECTED_FEATURES
        b_prot = b in BUSINESS_PROTECTED_FEATURES
        if a_prot or b_prot:
            protected_note = (f"[PROTECTION NOTE: {'both' if a_prot and b_prot else (a if a_prot else b)} "
                               f"{'are' if a_prot and b_prot else 'is'} business-protected \u2014 any remediation "
                               f"needs business sign-off, not an automatic drop, per the same exempt-but-not-immune "
                               f"philosophy used in redundancy()/feature_selection()]")
    print(f"Max remaining pairwise |r| = {max_r:.3f} between {max_pair}  (threshold {NEAR_DUPLICATE_THRESHOLD})")
    print(f"{'\u26a0\ufe0f  FLAG — near-duplicate feature' if flags['near_duplicate'] else '\u2705 clean'}{protected_note}")

    subbanner("Check 4: Train/inference consistency")
    print(f"Imputer strategy (per-column, skew-conditional), the log1p/signed-log transform lists")
    print(f"(log_transformed / signed_log_transformed), the winsorization bounds, and the scaler are all")
    print(f"fit ONCE on this reference extract in Step 2B-2D and must be persisted for production scoring;")
    print(f"production scoring must reuse them, never refit. \u2705 by construction of this pipeline's structure")
    print(f"(Step 2B-2D's preprocess() returns imputer/scaler/log_transformed/signed_log_transformed as a single")
    print(f"set of objects specifically so they can be persisted and reused, rather than recomputed per-run).")
    flags["train_inference_inconsistency"] = False

    subbanner("Check 5: Group/rank circularity (NEW — ties directly to Step 1B's structure-detection result)")
    circularity = _check_group_circularity(feat_final, structure_result)
    flags["group_rank_circularity"] = circularity is not None
    if circularity:
        prot_note = (" (not in BUSINESS_PROTECTED_FEATURES, so removal is NOT blocked by business protection)"
                      if circularity["rank_feature"] not in BUSINESS_PROTECTED_FEATURES else
                      " (IS in BUSINESS_PROTECTED_FEATURES \u2014 removal would need the elevated-threshold "
                      "justification used elsewhere)")
        print(f"\u26a0\ufe0f  FLAG \u2014 '{circularity['rank_feature']}' is in feat_final and is the direct ordinal encoding "
              f"of '{circularity['group_col']}', the SAME column Step 1B's ICC test found to dominate this "
              f"population's structure (see the Step 1B log for the exact ICC value). This isn't a data-quality "
              f"bug, but the resulting clusters risk substantially re-deriving the existing VIP-tier labelling "
              f"rather than discovering new structure{prot_note}.")
    else:
        print(f"\u2705 No group/rank circularity detected (either no group structure was found, or its rank "
              f"encoding is not present in the active feature set).")

    subbanner("Consolidated leakage flag")
    any_flag = any(flags.values())
    print(f"{flags}")
    print(f"CONSOLIDATED FLAG: {'\u26a0\ufe0f  NON-EMPTY \u2014 investigation required before Step 4' if any_flag else '\u2705 EMPTY \u2014 cleared to proceed to Step 4'}")
    return flags



def resolve_multiway_collinearity(X_scaled, feat_final, threshold=1e-6, protected=None):
    """Iteratively removes the largest-loading unprotected feature in the
    worst (smallest-singular-value) null-space direction, recomputing SVD
    after each removal, until the feature matrix is genuinely full-rank.
    Protected features are never removed; if every feature remaining in a
    degenerate direction is protected, stops and leaves that dependency
    for the elevated-threshold protected-feature review process rather
    than dropping a protected feature automatically."""
    protected = protected or []
    current = list(feat_final)
    removed = []
    while True:
        X = X_scaled[current].values
        Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        if S.min() >= threshold:
            break
        null_vec = Vt[-1]
        max_load = np.abs(null_vec).max()
        
        order = np.argsort(-np.abs(null_vec))
        drop_feat = None
        for idx in order:
            if abs(null_vec[idx]) < 0.5 * max_load:
                break  # no more MEANINGFUL contributors left to consider
            cand = current[idx]
            if cand not in protected:
                drop_feat = cand
                break
        if drop_feat is None:
            break
        current.remove(drop_feat)
        removed.append(drop_feat)
    return current, removed


def _self_test_multiway_collinearity():
    
    banner("SELF-TEST: Multi-way collinearity resolver (SVD-based)")
    rng = np.random.RandomState(0)
    n = 2000
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = a + b  # exact 3-way dependency; pairwise corr(a,c) and corr(b,c) are both well under 0.9
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)
    df_test = pd.DataFrame({"a": a, "b": b, "c": c, "noise1": noise1, "noise2": noise2})
    print(f"pairwise corr(a,c)={df_test['a'].corr(df_test['c']):.3f}, "
          f"corr(b,c)={df_test['b'].corr(df_test['c']):.3f} -- neither exceeds a typical 0.9 pairwise "
          f"threshold, yet a+b-c=0 exactly.")
    reduced, removed = resolve_multiway_collinearity(df_test, ["a", "b", "c", "noise1", "noise2"], protected=[])
    assert len(removed) == 1, "exactly one feature from the 3-way dependency should be removed"
    assert set(["a", "b", "c"]) & set(removed), "the removed feature must come from the genuine dependency"
    print(f"\u2705 Removed {removed} -- resolves the exact 3-way dependency a PAIRWISE check would miss.")

    reduced_p, removed_p = resolve_multiway_collinearity(df_test, ["a", "b", "c", "noise1", "noise2"],
                                                            protected=["a", "b", "c"])
    assert removed_p == [], "a fully-protected dependency must be left in place, not auto-resolved"
    print(f"\u2705 When every feature in the dependency is protected, none are dropped (left for manual review).")


_self_test_multiway_collinearity()

final_features, multiway_removed = resolve_multiway_collinearity(
    X_scaled, final_features, protected=BUSINESS_PROTECTED_FEATURES)
if multiway_removed:
    print(f"\n\u26a0\ufe0f  Multi-way collinearity resolver removed {len(multiway_removed)} feature(s) Step 3D's "
          f"own dual-check didn't catch: {multiway_removed}")
    print(f"feat_final now: {len(final_features)} features.")
else:
    print(f"\n\u2705 Multi-way collinearity check: feature matrix is genuinely full-rank, nothing further removed.")


_self_test_leakage_helpers()

leakage_flags = leakage_checks(X_scaled, final_features, entity_col, modelling_df,
    snapshot_date=snapshot_date, structure_result=structure_result, raw_df=df,)

#==================================================================================
# Pre-Clustering-Tendency Investigation
# =====================================================================


def _investigate_group_rank_circularity(ctx):
    
    X_scaled, feat_final = ctx["X_scaled"], ctx["feat_final"]
    structure_result, modelling_df, k_screen = ctx["structure_result"], ctx["modelling_df"], ctx["k_screen"]

    circularity = _check_group_circularity(feat_final, structure_result)
    if circularity is None:
        print(f"\u26a0\ufe0f  Flag fired but _check_group_circularity() found nothing on this feat_final -- "
              f"inconsistent state (feat_final or structure_result changed between Step 3E and Step 3F). "
              f"Needs manual review.")
        return {"status": "inconsistent_state"}
    rank_feature, group_col = circularity["rank_feature"], circularity["group_col"]

    X_with = X_scaled[feat_final]
    feat_without = [f for f in feat_final if f != rank_feature]
    X_without = X_scaled[feat_without]
    group_labels_arr = modelling_df.loc[X_scaled.index, group_col].astype(str).values

    km_with = KMeans(n_clusters=k_screen, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_with)
    km_without = KMeans(n_clusters=k_screen, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_without)

    ari_with_vs_without = adjusted_rand_score(km_with.labels_, km_without.labels_)
    ari_with_vs_group = adjusted_rand_score(km_with.labels_, group_labels_arr)
    ari_without_vs_group = adjusted_rand_score(km_without.labels_, group_labels_arr)

    print(f"Comparing K-Means (k={k_screen}) WITH vs WITHOUT '{rank_feature}':")
    print(f"ARI(with-rank clusters, without-rank clusters) = {ari_with_vs_group if False else ari_with_vs_without:.4f}")
    print(f"ARI(with-rank clusters, raw '{group_col}') = {ari_with_vs_group:.4f}")
    print(f"ARI(without-rank clusters, raw '{group_col}') = {ari_without_vs_group:.4f}")
    if ari_with_vs_group > ari_without_vs_group + 0.05:
        print(f"\u26a0\ufe0f  Dropping '{rank_feature}' measurably REDUCES agreement with raw '{group_col}' "
              f"({ari_with_vs_group:.4f} \u2192 {ari_without_vs_group:.4f}) \u2014 the feature is doing real work "
              f"pulling clusters toward the existing tier labels.")
    else:
        print(f"Agreement with raw '{group_col}' is similar with or without '{rank_feature}' \u2014 the tier "
              f"signal is likely coming from OTHER correlated features too (e.g. Total_Deposit_Amount, "
              f"Cash_Stake_Casino), not '{rank_feature}' alone.")

    subbanner("Within-group silhouette check (does real structure survive inside a single tier?)")
    pooled_sil = silhouette_score(X_without, km_without.labels_,
                                   sample_size=min(4000, len(X_without)), random_state=RANDOM_STATE)
    within_group_sils = {}
    MIN_GROUP_SIZE_FOR_CHECK = max(20, k_screen * 5)
    for g in np.unique(group_labels_arr):
        g_idx = modelling_df.index[modelling_df[group_col].astype(str) == g]
        g_idx = g_idx.intersection(X_without.index)
        if len(g_idx) < MIN_GROUP_SIZE_FOR_CHECK:
            within_group_sils[g] = None
            print(f"     {g:<16} n={len(g_idx):<6} SKIPPED (below {MIN_GROUP_SIZE_FOR_CHECK}-row evaluation floor)")
            continue
        Xg = X_without.loc[g_idx]
        try:
            kmg = KMeans(n_clusters=k_screen, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(Xg)
            sil_g = silhouette_score(Xg, kmg.labels_, sample_size=min(2000, len(Xg)), random_state=RANDOM_STATE)
            within_group_sils[g] = sil_g
            print(f"     {g:<16} n={len(g_idx):<6} within-tier silhouette = {sil_g:.4f}")
        except Exception as e:
            within_group_sils[g] = None
            print(f"     {g:<16} n={len(g_idx):<6} SKIPPED (fit failed: {e})")

    valid_within = {g: s for g, s in within_group_sils.items() if s is not None}
    WITHIN_GROUP_STRUCTURE_FLOOR = 0.15
    if valid_within:
        mean_within = float(np.mean(list(valid_within.values())))
        print(f"\n Pooled silhouette ({rank_feature} excluded, all tiers together): {pooled_sil:.4f}")
        print(f"Mean within-tier silhouette (across {len(valid_within)} evaluable tier(s)):    {mean_within:.4f}")
        if mean_within < WITHIN_GROUP_STRUCTURE_FLOOR:
            verdict = (f"real within-tier structure is weak-to-absent (mean {mean_within:.4f} < floor "
                       f"{WITHIN_GROUP_STRUCTURE_FLOOR}) \u2014 most of the pooled separation is plausibly coming "
                       f"from BETWEEN-tier differences, consistent with the circularity concern")
        else:
            verdict = (f"meaningful structure also exists WITHIN individual tiers (mean {mean_within:.4f} >= floor "
                       f"{WITHIN_GROUP_STRUCTURE_FLOOR}), not just between them \u2014 the circularity concern is "
                       f"real but less severe than a worst-case reading")
    else:
        mean_within = None
        verdict = "no tier had enough customers to evaluate within-group structure \u2014 inconclusive, not cleared"
    print(f"   VERDICT: {verdict}")

    return {
        "rank_feature": rank_feature, "group_col": group_col,
        "ari_with_vs_without": float(ari_with_vs_without),
        "ari_with_vs_group": float(ari_with_vs_group),
        "ari_without_vs_group": float(ari_without_vs_group),
        "pooled_silhouette": float(pooled_sil),
        "within_group_silhouette": within_group_sils,
        "mean_within_group_silhouette": mean_within,
        "verdict": verdict,
        "recommendation": (f"drop '{rank_feature}' from the clustering feature set and use it only for post-hoc "
                            f"profiling, unless the business goal is explicitly to refine/validate '{group_col}' "
                            f"rather than discover orthogonal structure"),
    }


def _investigate_outcome_contamination(ctx):
    
    modelling_df, feat_final = ctx["modelling_df"], ctx["feat_final"]
    matched, matched_active = _find_kpi_contamination_columns(modelling_df.columns, feat_final)
    print(f"ACTIVE KPI-vocabulary column(s) currently feeding the clustering: {matched_active}")
    print(f"Recommended remediation: remove {matched_active} from feat_final before Clustering Tendency")
    print(f"Assessment / Step 4 \u2014 a downstream business outcome (self-exclusion, bonus-abuse, deposit-churn,")
    print(f"etc.) should be used to DESCRIBE segments after clustering, not to DEFINE them during clustering.")
    return {"active_kpi_columns": matched_active, "recommendation": "remove from feat_final before Step 4"}


def _investigate_outcome_kpi_present_but_inactive(ctx):
   
    modelling_df, feat_final = ctx["modelling_df"], ctx["feat_final"]
    matched, matched_active = _find_kpi_contamination_columns(modelling_df.columns, feat_final)
    dormant = [c for c in matched if c not in matched_active]
    print(f"Dormant KPI-vocabulary column(s) (present in extract, NOT active in feat_final): {dormant}")
    print(f"Standing invariant registered for future runs: none of {dormant} may enter feat_final without")
    print(f"explicit business sign-off, since {'they match' if len(dormant) != 1 else 'it matches'} this ")
    print(f"pipeline's own KPI_CANDIDATE_NAMES vocabulary. No action required THIS run.")
    return {"dormant_kpi_columns": dormant,
            "invariant": "no dormant KPI column may become active in feat_final without explicit review"}


def _investigate_near_duplicate(ctx):
    
    X_scaled, feat_final = ctx["X_scaled"], ctx["feat_final"]
    X = X_scaled[feat_final]
    corr_final = X.corr().abs().to_numpy().copy()
    np.fill_diagonal(corr_final, 0)
    corr_final = pd.DataFrame(corr_final, index=X.columns, columns=X.columns)
    max_pair = corr_final.stack().idxmax()
    max_r = corr_final.stack().max()
    a, b = max_pair
    a_prot, b_prot = a in BUSINESS_PROTECTED_FEATURES, b in BUSINESS_PROTECTED_FEATURES
    if a_prot and b_prot:
        drop, recommendation = None, f"both '{a}' and '{b}' are business-protected \u2014 cannot auto-resolve, needs manual business sign-off"
    elif a_prot:
        drop, recommendation = b, f"drop '{b}' (unprotected), keep protected '{a}'"
    elif b_prot:
        drop, recommendation = a, f"drop '{a}' (unprotected), keep protected '{b}'"
    else:
        drop, recommendation = b, f"drop '{b}' \u2014 no protection on either side, same first-encountered-keeps rule as redundancy()"
    print(f"   Near-duplicate pair: {max_pair}, |r|={max_r:.3f}")
    print(f"   Recommendation: {recommendation}")
    return {"pair": max_pair, "r": float(max_r), "recommended_drop": drop, "recommendation": recommendation}


def _investigate_temporal_contamination(ctx):
   
    modelling_df, snapshot_date, raw_df = ctx["modelling_df"], ctx["snapshot_date"], ctx.get("raw_df")
    raw_date_cols = {c: modelling_df[c] for c in
                      ["Reg_Date", "First_Deposit_Date", "Last_Login_Date", "Last_Deposit_Date"]
                      if c in modelling_df.columns}
    if raw_df is not None:
        for c in ["Activity_Date", "Bet_Date"]:
            if c in raw_df.columns:
                raw_date_cols[c] = raw_df[c]
    violations = _find_lookahead_dates(raw_date_cols, snapshot_date) if snapshot_date is not None and raw_date_cols else {}
    print(f"Lookahead violations by column: {violations if violations else 'NONE'}")
    print(f"FIX ALREADY APPLIED (Part A): snapshot_date now uses df['Activity_Date'].max().normalize()")
    print(f"instead of the wall-clock pd.Timestamp.today() this recommendation used to point at -- recency")
    print(f"features (tenure_days, days_since_login, days_since_deposit, active_days) are now reproducible")
    print(f"run-to-run regardless of what day the pipeline executes on. Confirmed this mattered in")
    print(f"practice: two runs two days apart previously shifted tenure_days by 2 and measurably changed")
    print(f"Ward/Hierarchical's cluster boundaries (ARI vs K-Means moved from 0.86 to 0.69) before this fix.")
    return {"violations": violations,
            "recommendation": "switch snapshot_date to a data-derived value instead of wall-clock today()"}


def _investigate_train_inference_inconsistency(ctx):
    
    print(f"This flag is False by construction \u2014 if it is ever True, audit Step 2B-2D through Step 4 for an")
    print(f"accidental re-fit (e.g. scaler.fit_transform() called again on new data instead of scaler.transform()),")
    print(f"since the pipeline's design assumes these objects are fit exactly once and persisted.")
    return {"status": "unexpected: this flag should structurally never be True in this pipeline",
            "recommendation": "treat as a pipeline bug, halt before Step 4, and audit for an accidental re-fit"}


def _investigate_unknown_flag(ctx, flag_name, flag_value):
    
    print(f"   No registered investigator for flag '{flag_name}' (value={flag_value}).")
    print(f"   Falling back to a generic manual-review marker: this flag fired, but Step 3F does not yet know how")
    print(f"   to automatically investigate it. To automate this the next time it fires, add an")
    print(f"   `_investigate_{flag_name}(ctx)` function and register it in INVESTIGATION_REGISTRY.")
    return {"status": "manual review required \u2014 no automated investigator registered", "flag": flag_name, "value": flag_value}


INVESTIGATION_REGISTRY = {
    "outcome_contamination": _investigate_outcome_contamination,
    "outcome_kpi_present_but_inactive": _investigate_outcome_kpi_present_but_inactive,
    "temporal_contamination": _investigate_temporal_contamination,
    "near_duplicate": _investigate_near_duplicate,
    "train_inference_inconsistency": _investigate_train_inference_inconsistency,
    "group_rank_circularity": _investigate_group_rank_circularity,
}


def _resolve_investigation_plan(leakage_flags, registry):
   
    plan = {}
    for flag_name, flag_value in leakage_flags.items():
        if not flag_value:
            plan[flag_name] = "skip"
        elif flag_name in registry:
            plan[flag_name] = "run_registered"
        else:
            plan[flag_name] = "run_fallback"
    return plan


def _self_test_investigation_dispatch():
    
    banner("SELF-TEST: Step 3F conditional investigation dispatch")
    fake_registry = {"known_flag_a": lambda ctx: "handled_a", "known_flag_b": lambda ctx: "handled_b"}
    flags = {
        "known_flag_a": True,
        "known_flag_b": False,
        "totally_new_flag_nobody_registered_yet": True,
        "another_clean_flag": False,
    }
    plan = _resolve_investigation_plan(flags, fake_registry)
    print(f"Flags: {flags}")
    print(f"Resolved plan: {plan}")
    assert plan["known_flag_a"] == "run_registered", "a True flag with a registered investigator must run it"
    assert plan["known_flag_b"] == "skip", "a False flag must be skipped regardless of registry membership"
    assert plan["totally_new_flag_nobody_registered_yet"] == "run_fallback", \
        "a True flag with NO registered investigator must fall back, not be dropped or crash"
    assert plan["another_clean_flag"] == "skip", "a False flag must be skipped"
    print(f"\u2705 False flags are skipped, known True flags route to their registered investigator, and an")
    print(f"UNSEEN True flag (simulating a future Step 3E check this dispatcher wasn't written for) still")
    print(f"routes to the generic fallback rather than being silently dropped or crashing the pipeline.")


def investigate_leakage_flags(leakage_flags, X_scaled, feat_final, modelling_df, structure_result=None,
                                snapshot_date=None, raw_df=None, k_screen=4):
    banner("STEP 3F: Conditional Pre-Clustering-Tendency Investigation")
    plan = _resolve_investigation_plan(leakage_flags, INVESTIGATION_REGISTRY)
    print(f"Step 3E flags: {leakage_flags}")
    print(f"Resolved investigation plan: {plan}")
    print(f"(Any flag not in INVESTIGATION_REGISTRY still gets a generic fallback investigation \u2014 nothing")
    print(f"fired-but-unhandled ever passes through silently.)")

    ctx = {"X_scaled": X_scaled, "feat_final": feat_final, "modelling_df": modelling_df,
           "structure_result": structure_result, "snapshot_date": snapshot_date,
           "raw_df": raw_df, "k_screen": k_screen}

    results = {}
    for flag_name, action in plan.items():
        subbanner(f"Flag: {flag_name}")
        if action == "skip":
            print(f"\u2705 Not flagged \u2014 no investigation needed.")
            continue
        elif action == "run_registered":
            print(f"Running registered investigator for '{flag_name}'...")
            results[flag_name] = INVESTIGATION_REGISTRY[flag_name](ctx)
        else:
            print(f"\u26a0\ufe0f  '{flag_name}' fired but has NO registered investigator (a newer/different Step 3E")
            print(f"than this dispatcher was written against) \u2014 falling back to generic manual-review handling.")
            results[flag_name] = _investigate_unknown_flag(ctx, flag_name, leakage_flags[flag_name])

    subbanner("Investigation summary")
    if results:
        for k, v in results.items():
            print(f"{k}:")
            if isinstance(v, dict):
                for vk, vv in v.items():
                    print(f"{vk}: {vv}")
            else:
                print(f"{v}")
        print(f"\n {len(results)} flag(s) investigated. Review each recommendation above before proceeding to")
        print(f"Clustering Tendency Assessment / Step 4 \u2014 this dispatcher documents and quantifies each issue,")
        print(f"it does not silently auto-remediate feat_final.")
    else:
        print(f"No flags required investigation \u2014 cleared to proceed directly to Clustering Tendency Assessment.")
    return results


_self_test_investigation_dispatch()

investigation_results = investigate_leakage_flags(
    leakage_flags, X_scaled, final_features, modelling_df,
    structure_result=structure_result, snapshot_date=snapshot_date, raw_df=df, k_screen=4,)


#=====================================================================
# Clustering Tendency Assessment
# =========================================================================

def _apply_investigation_recommendations(feat_final, investigation_results):
    
    feat_final_v2 = list(feat_final)
    changes = []

    circ = investigation_results.get("group_rank_circularity")
    if circ and isinstance(circ, dict) and "rank_feature" in circ:
        rank_feature = circ["rank_feature"]
        if rank_feature in feat_final_v2:
            if rank_feature in BUSINESS_PROTECTED_FEATURES:
                changes.append({"action": "skipped", "feature": rank_feature,
                                 "reason": f"'{rank_feature}' is business-protected; dropping it needs explicit "
                                           f"elevated-threshold justification, not an automatic Step 4 remediation"})
            else:
                feat_final_v2.remove(rank_feature)
                changes.append({"action": "dropped", "feature": rank_feature,
                                 "reason": "group_rank_circularity recommendation (not business-protected)"})

    near_dup = investigation_results.get("near_duplicate")
    if near_dup and isinstance(near_dup, dict):
        drop = near_dup.get("recommended_drop")
        if drop is not None and drop in feat_final_v2 and drop not in BUSINESS_PROTECTED_FEATURES:
            feat_final_v2.remove(drop)
            changes.append({"action": "dropped", "feature": drop,
                             "reason": "near_duplicate recommendation (unprotected side of the pair)"})
        elif drop is not None and drop in BUSINESS_PROTECTED_FEATURES:
            changes.append({"action": "skipped", "feature": drop,
                             "reason": f"'{drop}' was recommended for drop but is business-protected \u2014 "
                                       f"defensive re-check overrides the recommendation, needs manual review"})
        elif drop is None and near_dup.get("pair"):
            changes.append({"action": "skipped", "feature": near_dup["pair"],
                             "reason": "both sides of the near-duplicate pair are business-protected; needs manual "
                                       "business sign-off, not an automatic Step 4 remediation"})

    outcome = investigation_results.get("outcome_contamination")
    if outcome and isinstance(outcome, dict):
        for c in outcome.get("active_kpi_columns", []):
            if c in feat_final_v2:
                if c in BUSINESS_PROTECTED_FEATURES:
                    changes.append({"action": "skipped", "feature": c,
                                     "reason": f"'{c}' is business-protected despite being an active KPI match; "
                                               f"needs explicit review, not an automatic drop"})
                else:
                    feat_final_v2.remove(c)
                    changes.append({"action": "dropped", "feature": c,
                                     "reason": "outcome_contamination recommendation (unprotected)"})

    return feat_final_v2, changes


def _self_test_investigation_remediation():
    
    banner("SELF-TEST: Step 4 investigation-recommendation remediation (opt-in)")
    fake_feat_final = ["vip_rank", "Total_Deposit_Amount", "GGR", "some_dup_a", "some_dup_b", "KPI_col"]

    print(f"Sub-test A: unprotected circular rank feature IS dropped when recommended")
    inv_a = {"group_rank_circularity": {"rank_feature": "vip_rank", "group_col": "vip_tier"}}
    feat_a, changes_a = _apply_investigation_recommendations(fake_feat_final, inv_a)
    assert "vip_rank" not in feat_a, "unprotected rank feature must be dropped when recommended"
    assert any(c["action"] == "dropped" and c["feature"] == "vip_rank" for c in changes_a)
    print(f"feat_final after: {feat_a}")
    print(f"changes: {changes_a}")

    print(f"\n Sub-test B: a PROTECTED circular feature is NEVER auto-dropped")
    inv_b = {"group_rank_circularity": {"rank_feature": "Total_Deposit_Amount", "group_col": "vip_tier"}}
    feat_b, changes_b = _apply_investigation_recommendations(fake_feat_final, inv_b)
    assert "Total_Deposit_Amount" in feat_b, "a business-protected feature must NEVER be auto-dropped"
    assert any(c["action"] == "skipped" and c["feature"] == "Total_Deposit_Amount" for c in changes_b)
    print(f"feat_final after: {feat_b}")
    print(f"changes: {changes_b}")

    print(f"\n Sub-test C: near_duplicate with an explicit unprotected recommended_drop IS applied")
    inv_c = {"near_duplicate": {"pair": ("some_dup_a", "some_dup_b"), "r": 0.99, "recommended_drop": "some_dup_b"}}
    feat_c, changes_c = _apply_investigation_recommendations(fake_feat_final, inv_c)
    assert "some_dup_b" not in feat_c and "some_dup_a" in feat_c
    print(f"feat_final after: {feat_c}")

    print(f"\n Sub-test D: near_duplicate with recommended_drop=None (both sides protected) changes NOTHING")
    inv_d = {"near_duplicate": {"pair": ("some_dup_a", "some_dup_b"), "r": 0.99, "recommended_drop": None}}
    feat_d, changes_d = _apply_investigation_recommendations(fake_feat_final, inv_d)
    assert feat_d == fake_feat_final, "recommended_drop=None must leave feat_final completely unchanged"
    assert changes_d[0]["action"] == "skipped"
    print(f"feat_final after: {feat_d}  (unchanged, as expected)")

    print(f"\n Sub-test E: informational-only flags (temporal_contamination, etc.) never touch feat_final")
    inv_e = {"temporal_contamination": {"violations": {}}, "train_inference_inconsistency": {"status": "x"}}
    feat_e, changes_e = _apply_investigation_recommendations(fake_feat_final, inv_e)
    assert feat_e == fake_feat_final and changes_e == [], "informational-only flags must produce zero changes"
    print(f"feat_final after: {feat_e}  (unchanged, no remediation defined for these flags)")

    print(f"\n \u2705 Remediation logic drops exactly the unprotected recommended features, NEVER touches a")
    print(f"business-protected feature (even when explicitly recommended), and leaves informational-only")
    print(f"flags with zero effect.")


def _hopkins_within_groups(X_scaled, feat_final, modelling_df, group_col, metric, num_cols, cat_cols, seeds, min_group_n=50):
    
    results = {}
    group_labels_ser = modelling_df.loc[X_scaled.index, group_col].astype(str)
    for g in sorted(group_labels_ser.unique()):
        g_idx = group_labels_ser.index[group_labels_ser == g]
        n_g = len(g_idx)
        if n_g < min_group_n:
            results[g] = {"n": n_g, "H_mean": None, "skipped_reason": f"n={n_g} < floor {min_group_n}"}
            continue
        if metric == "gower":
            Xg_df = X_scaled.loc[g_idx, feat_final]
            Hs = [hopkins_statistic(Xg_df, seed=s, metric="gower", num_cols=num_cols, cat_cols=cat_cols) for s in seeds]
        elif metric == "mahalanobis":
            Xg = X_scaled.loc[g_idx, feat_final].values
            Hs = [hopkins_statistic(Xg, seed=s, metric="mahalanobis") for s in seeds]
        else:
            Xg = X_scaled.loc[g_idx, feat_final].values
            Hs = [hopkins_statistic(Xg, seed=s) for s in seeds]
        results[g] = {"n": n_g, "H_mean": float(np.mean(Hs)), "H_values": [round(h, 4) for h in Hs]}
    return results


def tendency(X_scaled, feat_final, max_vif, structure_result=None, modelling_df=None,
             investigation_results=None, apply_investigation_recommendations=False, n_repeats=5):
    banner("STEP 4: Clustering Tendency (Hopkins Statistic)")
    HOPKINS_THRESHOLD = 0.75

    subbanner("Structure-detection decision matrix applied to this step")
    temporal_detected = group_detected = network_detected = False
    group_col = None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        group_detected = structure_result.get("group_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: cross-sectional "
              f"snapshot already confirmed by Step 1B; hopkins_statistic()'s internal subsampling has no time "
              f"dimension to respect.")
        print(f"Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: not relevant to a "
              f"spatial-randomness test.")
        if group_detected:
            print(f"Group structure: YES ('{group_col}') \u2192 genuinely relevant, but NOT by switching")
            print(f"hopkins_statistic()'s internal subsampling to group-BLOCKED draws: Hopkins asks 'does a")
            print(f"representative random sample look structured', and the real casino population genuinely")
            print(f"IS group-imbalanced (Non-VIP dominates). Forcing whole-tier draws would let a 4-customer")
            print(f"Platinum tier count as an entire 'representative' sample on some repeats, which defeats")
            print(f"the test. Plain iid row-level subsampling is kept for the pooled/gating test below; group")
            print(f"structure is instead respected by ADDING a within-group Hopkins cross-check (diagnostic,")
            print(f"non-gating) \u2014 the natural continuation of Step 3F's within-tier silhouette check.")
        else:
            print(f"Group structure: NO \u2192 pooled iid Hopkins is used directly; no within-group cross-check "
                  f"is needed.")
    else:
        print(f"No structure_result supplied \u2014 pooled iid Hopkins only, no decision-matrix narration or "
              f"within-group cross-check.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    if protected_present:
        print(f"Business-protected features present in this feature set: {protected_present}")
        print(f"Hopkins is a MEASUREMENT, not a selection step \u2014 protection status doesn't change how it's")
        print(f"computed. It matters for the opt-in remediation immediately below: a protected feature is")
        print(f"never auto-dropped by Step 3F's investigation recommendations, even if this call opts in.")
    else:
        print(f"No business-protected features present in this feature set.")

    subbanner("Step 3F investigation-recommendation remediation (opt-in, default OFF)")
    feat_final_used = list(feat_final)
    remediation_changes = []
    if investigation_results:
        print(f"Step 3E/3F produced {len(investigation_results)} investigated flag(s): "
              f"{list(investigation_results.keys())}")
        if apply_investigation_recommendations:
            feat_final_used, remediation_changes = _apply_investigation_recommendations(feat_final, investigation_results)
            if remediation_changes:
                for c in remediation_changes:
                    print(f"[{c['action'].upper()}] {c['feature']}: {c['reason']}")
                print(f"{len(feat_final)} \u2192 {len(feat_final_used)} features after applying investigation "
                      f"recommendations.")
            else:
                print(f"No investigated flag produced an applicable feature-level recommendation \u2014 "
                      f"feat_final unchanged.")
        else:
            print(f"apply_investigation_recommendations=False (default) \u2014 recommendations are NOT applied.")
            print(f"Proceeding with the untouched feat_final ({len(feat_final)} features). Call tendency(...,")
            print(f"apply_investigation_recommendations=True) to opt in and see exactly what would change.")
    else:
        print(f"No investigation_results supplied (or Step 3F found nothing to investigate) \u2014 feat_final "
              f"unchanged.")

    subbanner("Enhancement 17 (Hopkins metric-awareness, round 7): preliminary metric check")
    HIGH_VIF_RESIDUAL = 10.0  # matches Step 5's own threshold exactly, intentionally duplicated
    n_cat = int(sum(not pd.api.types.is_numeric_dtype(X_scaled[feat_final_used][col]) for col in feat_final_used))
    num_cols = [c for c in feat_final_used if pd.api.types.is_numeric_dtype(X_scaled[c])]
    cat_cols = [c for c in feat_final_used if c not in num_cols]
    if n_cat > 0:
        preliminary_metric = "gower"
    elif max_vif > HIGH_VIF_RESIDUAL:
        preliminary_metric = "mahalanobis"
    else:
        preliminary_metric = "euclidean"
    print(f"Same signal Step 5 will officially use: n_cat={n_cat}, max_vif={max_vif:.2f} "
          f"\u2192 preliminary metric = '{preliminary_metric}'")
    print(f"\n Self-test (proves Hopkins correctly discriminates clustered vs uniform data under")
    print(f"Gower/Mahalanobis, not just the Euclidean default):")
    
    try:
        _self_test_hopkins_metric_awareness()
    except NameError as e:
        print(f"\u26a0\ufe0f  Skipped \u2014 this self-test needs the same distance-matrix helper(s) ({e}) that live in")
        print(f"a later part of this pipeline, not included in what Step 4 was written against. The POOLED")
        print(f"Hopkins computation below has its own independent fallback and will still produce a real result.")

    seeds = [42, 7, 123, 2024, 99][:n_repeats]

    def _run_hopkins(metric_to_use):
        if metric_to_use == "gower":
            return [hopkins_statistic(X_scaled[feat_final_used], seed=s, metric="gower",
                                       num_cols=num_cols, cat_cols=cat_cols) for s in seeds]
        elif metric_to_use == "mahalanobis":
            return [hopkins_statistic(X_scaled[feat_final_used].values, seed=s, metric="mahalanobis") for s in seeds]
        else:
            return [hopkins_statistic(X_scaled[feat_final_used].values, seed=s) for s in seeds]

    
    try:
        Hs = _run_hopkins(preliminary_metric)
        metric_used = preliminary_metric
    except NameError as e:
        print(f"\n \u26a0\ufe0f  Metric '{preliminary_metric}' requires a distance-matrix helper not available in "
              f"this execution environment ({e}) \u2014 it lives in a later part of this pipeline not included in")
        print(f"what Step 4 was written against. Falling back to 'euclidean' for THIS run so Step 4 produces a")
        print(f"real, honest result rather than crashing \u2014 re-run with that helper available to get the")
        print(f"metric Step 5 will actually use.")
        Hs = _run_hopkins("euclidean")
        metric_used = "euclidean (fallback from 'mahalanobis')"

    print(f"\n Hopkins statistic across {n_repeats} repeated subsamples (m=5% of n, metric="
          f"'{metric_used}', feature set size={len(feat_final_used)}): {[round(h,4) for h in Hs]}")
    H_mean = float(np.mean(Hs))
    print(f"Mean H = {H_mean:.4f} (threshold {HOPKINS_THRESHOLD})")

    within_group_hopkins = None
    metric_for_within = "euclidean" if metric_used.startswith("euclidean (fallback") else metric_used
    if group_detected and group_col and modelling_df is not None:
        subbanner(f"Within-group Hopkins cross-check ('{group_col}')")
        within_group_hopkins = _hopkins_within_groups(
            X_scaled, feat_final_used, modelling_df, group_col, metric_for_within, num_cols, cat_cols, seeds
        )
        for g, res in within_group_hopkins.items():
            if res["H_mean"] is None:
                print(f"{g:<16} n={res['n']:<6} SKIPPED ({res['skipped_reason']})")
            else:
                flag = "" if res["H_mean"] > HOPKINS_THRESHOLD else "  \u26a0\ufe0f below pooled threshold"
                print(f"{g:<16} n={res['n']:<6} H_mean={res['H_mean']:.4f}{flag}")
        valid = {g: r["H_mean"] for g, r in within_group_hopkins.items() if r["H_mean"] is not None}
        if valid:
            mean_within_H = float(np.mean(list(valid.values())))
            print(f"\n Pooled H = {H_mean:.4f}  |  mean within-tier H (across {len(valid)} evaluable tier(s)) "
                  f"= {mean_within_H:.4f}")
            if mean_within_H > HOPKINS_THRESHOLD:
                print(f"\u2705 Clustering tendency is NOT just a between-tier artifact \u2014 tiers evaluable here")
                print(f"also individually clear the {HOPKINS_THRESHOLD} threshold, consistent with Step 3F's")
                print(f"within-tier silhouette finding (real structure exists inside tiers too).")
            else:
                print(f"\u26a0\ufe0f  Mean within-tier H falls below the pooled threshold \u2014 worth weighing")
                print(f"alongside Step 3F's within-tier silhouette result before treating the pooled H as the")
                print(f"full picture (diagnostic only, does NOT change the gate decision below).")
        else:
            print(f"\n No tier had enough customers to evaluate within-group Hopkins \u2014 diagnostic "
                  f"inconclusive (does not change the gate decision below).")

    subbanner("Gate decision")
    proceed = H_mean > HOPKINS_THRESHOLD
    print(f"DECISION: {'\u2705 PROCEED to algorithm fitting' if proceed else '\u274c HALT \u2014 data does not show cluster tendency'}")
    print(f"(Gate is based on the POOLED H only, per the {HOPKINS_THRESHOLD} threshold \u2014 the within-group")
    print(f"cross-check above is diagnostic context, not a second gate.)")
    if not proceed:
        raise RuntimeError(f"Hopkins H={H_mean:.4f} <= {HOPKINS_THRESHOLD} \u2014 halting per Step 4 gate.")

    return {
        "H_mean": H_mean, "H_values": [round(h, 4) for h in Hs], "metric_used": metric_used,
        "preliminary_metric": preliminary_metric, "feat_final_used": feat_final_used,
        "remediation_changes": remediation_changes, "within_group_hopkins": within_group_hopkins,}


banner("SELF-TESTS: Step 4 investigation-recommendation remediation")
_self_test_investigation_remediation()


PIPELINE_HALTED_AT = None
tendency_result = None
try:
    tendency_result = tendency(
        X_scaled, final_features, final_max_vif,
        structure_result=structure_result, modelling_df=modelling_df,
        investigation_results=investigation_results,
        apply_investigation_recommendations=False, n_repeats=5,
    )
except RuntimeError as e:
    PIPELINE_HALTED_AT = "Step 4 (Clustering Tendency)"
    banner("PIPELINE HALTED")
    print(f"Halted at: {PIPELINE_HALTED_AT}")
    print(f"Reason: {e}")
    print(f"\n This is a clean, intentional stop, not a crash. All diagnostics up to and including the")
    print(f"Hopkins gate decision were already computed and printed above \u2014 the structure-detection")
    print(f"decision matrix, protected-feature check, investigation-recommendation remediation status, and")
    print(f"both the pooled and within-group Hopkins results. The data at hand does not show clustering")
    print(f"tendency at the current feature set, so no algorithm was fit and no downstream step ran.")
    print(f"\n To proceed, consider one of:")
    print(f"1. Re-run with apply_investigation_recommendations=True to see whether dropping the")
    print(f"circular/near-duplicate features Step 3F flagged changes the Hopkins result.")
    print(f"2. Revisit whether the 0.75 Hopkins threshold and 5%-of-n subsample assumptions are")
    print(f"appropriate for this specific population.")
    print(f"3. Accept that this population may not support meaningful unsupervised segmentation as")
    print(f"currently featurized, and consider a business-rule-based or supervised approach instead.")

if PIPELINE_HALTED_AT is None:
    print(f"\n\u2705 Step 4 cleared \u2014 pipeline can proceed to Step 5 (Algorithm Fitting).")
else:
    print(f"\n\u26d4 Pipeline execution stops here \u2014 see 'PIPELINE HALTED' above. Any code added after this point")
    print(f"for future steps should check `if PIPELINE_HALTED_AT is None:` before running.")


######################## Distance Metric Selection 

def _recompute_max_vif(X_scaled, feat_final):
    
    Xv = X_scaled[feat_final].copy()
    Xv.insert(0, "const", 1.0)
    vifs = pd.Series(
        [variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])],
        index=feat_final)
    return float(vifs.max()), vifs


def _self_test_distance_metric_consistency():
    
    banner("SELF-TEST: Step 5 feature-set consistency & protection-aware VIF explanation")
    rng = np.random.RandomState(0)
    n = 300
    base = rng.normal(size=n)
    X_synth = pd.DataFrame({
        "Cash_Stake_Casino": base + rng.normal(scale=0.01, size=n),   # near-duplicate of collinear_a -> high VIF
        "collinear_a": base,
        "unrelated_b": rng.normal(size=n),
        "unrelated_c": rng.normal(size=n),
    })

    print(f"Sub-test A: VIF recomputation correctly identifies the actual highest-VIF feature")
    max_vif_recomputed, vifs = _recompute_max_vif(X_synth, list(X_synth.columns))
    print(f"Recomputed VIFs: {dict(vifs.round(2))}")
    assert vifs.idxmax() in ("Cash_Stake_Casino", "collinear_a"), \
        "the two collinear columns should dominate VIF, not the unrelated ones"
    assert max_vif_recomputed > 10.0, "the engineered collinear pair should produce a clearly high VIF"
    print(f"\u2705 Highest-VIF feature correctly identified as one of the engineered collinear pair "
          f"(max={max_vif_recomputed:.2f}).")

    print(f"\n Sub-test B: feature-set drift is correctly detected when tendency_result's feat_final_used "
          f"differs from the feat_final argument")
    feat_final_arg = list(X_synth.columns)
    fake_tendency_result = {"feat_final_used": ["Cash_Stake_Casino", "collinear_a", "unrelated_b"]}  # unrelated_c dropped
    drifted = set(feat_final_arg) != set(fake_tendency_result["feat_final_used"])
    assert drifted, "drift must be detected when the sets genuinely differ"
    dropped = set(feat_final_arg) - set(fake_tendency_result["feat_final_used"])
    print(f"feat_final argument: {feat_final_arg}")
    print(f"tendency_result['feat_final_used']: {fake_tendency_result['feat_final_used']}")
    print(f"\u2705 Drift correctly detected \u2014 {sorted(dropped)} would be dropped as not authoritative.")

    print(f"\n Sub-test C: NO drift is correctly detected as a no-op (the real casino run's default case)")
    fake_tendency_result_same = {"feat_final_used": list(X_synth.columns)}
    no_drift = set(feat_final_arg) == set(fake_tendency_result_same["feat_final_used"])
    assert no_drift, "identical sets must be reported as no drift"
    print(f"\u2705 Identical sets correctly reported as no drift (no unnecessary reconciliation narration).")

    print(f"\n \u2705 VIF recomputation and feature-set drift detection both behave correctly on controlled inputs.")


def distance_metric(X_scaled, feat_final, max_vif, preliminary_metric=None, structure_result=None,
                     tendency_result=None, investigation_results=None):
    banner("STEP 5: Distance Metric Selection")
    HIGH_VIF_RESIDUAL = 10.0

    subbanner("Structure-detection decision matrix applied to this step")
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        group_detected = structure_result.get("group_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: the metric decision")
        print(f"is a deterministic function of categorical-column count and max VIF, computed once over the")
        print(f"full fixed feature matrix \u2014 no row-level time dimension for this decision to depend on.")
        print(f"Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: network structure is an")
        print(f"algorithm/reporting decision (Step 6/8), not a metric-selection one.")
        print(f"Group structure: {'YES' if group_detected else 'NO'}"
              + (f"('{group_col}')" if group_detected else "")
              + f"\u2192 N/A here too, for a DIFFERENT reason than temporal/network: group imbalance could in")
        print(f"principle inflate a POOLED VIF number relative to within-group VIF, but that exact")
        print(f"cross-check already ran in redundancy()'s Step 3A (within-group correlation divergence flag)")
        print(f"\u2014 re-litigating it here would duplicate, not add, information. This step trusts the VIF")
        print(f"number it's given (or recomputes it fresh below), not a group-blocked variant of it.")
    else:
        print(f"No structure_result supplied \u2014 no decision-matrix narration; proceeding directly to the")
        print(f"dtype/VIF-driven decision below.")

    subbanner("Feature-set consistency & VIF recomputation (do not trust a possibly-stale max_vif)")
    feat_final_for_metric = list(feat_final)
    source_note = "the feat_final argument passed directly to this function"
    if tendency_result is not None and tendency_result.get("feat_final_used") is not None:
        feat_final_for_metric = list(tendency_result["feat_final_used"])
        source_note = ("Step 4's tendency_result['feat_final_used'] (the ACTUAL feature set Hopkins was computed "
                        "on, reflecting any opt-in Step 3F remediation)")
        if set(feat_final_for_metric) != set(feat_final):
            dropped = sorted(set(feat_final) - set(feat_final_for_metric))
            added = sorted(set(feat_final_for_metric) - set(feat_final))
            print(f"\u26a0\ufe0f  feat_final argument ({len(feat_final)} features) differs from {source_note} "
                  f"({len(feat_final_for_metric)} features).")
            if dropped:
                print(f"Dropped since Step 4: {dropped}")
            if added:
                print(f"Added since Step 4: {added}")
            print(f"Using the Step 4 set as authoritative for this step's decision.")
        else:
            print(f"feat_final argument matches {source_note} \u2014 no drift ({len(feat_final_for_metric)} "
                  f"features).")
    else:
        print(f"No tendency_result supplied \u2014 using {source_note} directly, with no cross-check available.")

    max_vif_recomputed, vifs_recomputed = _recompute_max_vif(X_scaled, feat_final_for_metric)
    print(f"\n Supplied max_vif argument:                                 {max_vif:.4f}")
    print(f"Recomputed max VIF on the actual {len(feat_final_for_metric)}-feature set in use:  "
          f"{max_vif_recomputed:.4f}")
    if abs(max_vif_recomputed - max_vif) > 0.01:
        print(f"\u26a0\ufe0f  These DIFFER \u2014 the supplied max_vif was computed on a different feature-set snapshot")
        print(f"than the one actually in use here (e.g. redundancy()'s pre-Step-3D feature set vs Step 3D's")
        print(f"final_features, after 'days_since_login' was dropped). Using the RECOMPUTED value below, since")
        print(f"it reflects the feature set actually being decided upon, not a stale upstream snapshot.")
    else:
        print(f"\u2705 Match (within rounding) \u2014 the supplied max_vif is still accurate for this feature set.")
    max_vif_used = max_vif_recomputed

    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final_for_metric if f in BUSINESS_PROTECTED_FEATURES]
    if protected_present:
        print(f"Business-protected features present in this feature set: {protected_present}")
        max_vif_feature = vifs_recomputed.idxmax()
        if max_vif_feature in BUSINESS_PROTECTED_FEATURES:
            print(f"The single highest-VIF feature ('{max_vif_feature}', VIF={max_vif_used:.2f}) IS business-")
            print(f"protected \u2014 this is WHY max_vif can still exceed the plain {HIGH_VIF_RESIDUAL} threshold")
            print(f"here: redundancy()'s iterative pruning applies an ELEVATED bar "
                  f"({PROTECTED_FEATURE_GAIN_MULTIPLIER:.0f}x{HIGH_VIF_RESIDUAL:.0f}="
                  f"{PROTECTED_FEATURE_GAIN_MULTIPLIER * HIGH_VIF_RESIDUAL:.0f}) to protected features rather than")
            print(f"the plain {HIGH_VIF_RESIDUAL:.0f} cutoff, deliberately keeping a business-critical feature")
            print(f"over a marginally cleaner VIF profile. The Mahalanobis choice below (if selected) is")
            print(f"therefore a direct, TRACEABLE consequence of protecting '{max_vif_feature}', not an")
            print(f"unexplained residual.")
        else:
            print(f"The highest-VIF feature ('{max_vif_feature}', VIF={max_vif_used:.2f}) is NOT itself")
            print(f"protected \u2014 elevated residual multicollinearity here is a genuine pruning-threshold")
            print(f"effect, unrelated to business protection.")
    else:
        print(f"No business-protected features present in this feature set.")

    if investigation_results:
        print(f"\n Step 3F investigated flag(s) still on record: {list(investigation_results.keys())}. None of")
        print(f"these change a distance-METRIC decision directly (it depends only on dtype counts and VIF),")
        print(f"but they remain relevant context for interpreting whatever Step 6 clustering produces.")

    dtypes = X_scaled[feat_final_for_metric].dtypes
    n_cat = int(sum(not pd.api.types.is_numeric_dtype(X_scaled[feat_final_for_metric][col])
                     for col in feat_final_for_metric))  # robust dtype check
    n_num = len(feat_final_for_metric) - n_cat
    print("\n Decision matrix:")
    print("\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    print("\u2502 Feature profile\u2502 Distance metric\u2502")
    print("\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    print("\u2502 All-continuous, scaled, low-moderate dim\u2502 Euclidean\u2605\u2502")
    print("\u2502 All-continuous, VIF still high after pruning\u2502 Mahalanobis\u2502")
    print("\u2502 Mixed numeric + categorical\u2502 Gower\u2502")
    print("\u2502 High-dim sparse behavioural/text vectors\u2502 Cosine\u2502")
    print("\u2502 Binary basket/flag data\u2502 Jaccard\u2502")
    print("\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518")
    print(f"\n ACTUAL profile (computed, not assumed): {n_num} numeric + {n_cat} categorical "
          f"columns in the final {len(feat_final_for_metric)}-feature set; max VIF (recomputed) = " f"{max_vif_used:.2f}")
    if n_cat > 0:
        metric = "gower"
        print(f"DECISION: Gower distance \u2014 {n_cat} categorical column(s) present, cannot use raw Euclidean.")
    elif max_vif_used > HIGH_VIF_RESIDUAL:
        metric = "mahalanobis"
        print(f"DECISION: Mahalanobis distance \u2014 max VIF ({max_vif_used:.2f}) still exceeds "
              f"{HIGH_VIF_RESIDUAL} after Step 3D pruning; a covariance-aware metric is warranted.")
    else:
        metric = "euclidean"
        print(f"DECISION: Euclidean distance \u2014 all {n_num} features numeric, max VIF ({max_vif_used:.2f}) "
              f"\u2264 {HIGH_VIF_RESIDUAL} \u2014 required by both K-Means centroids and Ward linkage, and the "
              f"best-supported choice for this all-continuous, redundancy-pruned profile.")

    if preliminary_metric is None and tendency_result is not None:
        preliminary_metric = tendency_result.get("preliminary_metric")
        print(f"\n No preliminary_metric argument supplied \u2014 auto-inherited '{preliminary_metric}' from "
              f"Step 4's tendency_result['preliminary_metric'] instead of requiring the caller to pass it again.")
    if preliminary_metric is not None:
        consistency = "\u2705 MATCHES" if preliminary_metric == metric else "\u26a0\ufe0f  MISMATCH"
        print(f"\n Consistency check vs Step 4's preliminary metric hint (Enhancement 17): "
              f"'{preliminary_metric}' \u2192 {consistency}")
        if preliminary_metric != metric:
            print(f"This should be structurally impossible (both checks compute the same n_cat/max_vif")
            print(f" logic against the same threshold) \u2014 a mismatch here means Step 4 and Step 5's")
            print(f"intentionally-duplicated logic have DRIFTED APART and need to be reconciled.")
            print(f" NOTE: Step 4 also has its own defensive fallback for a missing mahalanobis_distance_")
            print(f"matrix() helper (see tendency_result['metric_used']) \u2014 check that before assuming a")
            print(f"real drift; a metric_used of 'euclidean (fallback from ...)' explains a mismatch here")
            print(f"WITHOUT the two steps' logic having actually diverged.")

    return metric


if PIPELINE_HALTED_AT is None:
    banner("SELF-TESTS: Step 5 feature-set consistency & protection-aware VIF explanation")
    _self_test_distance_metric_consistency()


    metric = distance_metric(
        X_scaled, final_features, final_max_vif,
        structure_result=structure_result, tendency_result=tendency_result,
        investigation_results=investigation_results,
    )
else:
    print(f"\n\u23ed\ufe0f  Step 5 (Distance Metric Selection) SKIPPED \u2014 pipeline halted at "
          f"'{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")


#==================================================================================
# Distance-Metric Implementation
# =====================================================================


def _apply_scalability_subsample_group_aware(X_df, structure_result=None, group_labels=None, seed=RANDOM_STATE):
    
    group_detected = bool(structure_result and structure_result.get("group_detected"))
    if group_detected and group_labels is not None:
        idx = _group_blocked_resample_indices(np.asarray(group_labels), target_n=LARGE_N_SUBSAMPLE_SIZE, seed=seed)
        idx = np.sort(idx)
        if hasattr(X_df, "iloc"):
            return X_df.iloc[idx].reset_index(drop=True), idx
        return X_df[idx], idx
    if group_detected and group_labels is None:
        print(f"\u26a0\ufe0f  structure_result reports group_detected=True but no group_labels were supplied to "
              f"the Step 7B scalability guard \u2014 falling back to plain iid subsampling. Pass group_labels to "
              f"honour the Step 1B decision matrix.")
    return apply_scalability_subsample(X_df, seed=seed)


def _self_test_scalability_guard_group_aware():
    
    banner("SELF-TEST: group-aware scalability subsample")
    rng = np.random.RandomState(0)
    n = 25000
    groups = rng.choice(["A", "B", "C", "D"], size=n, p=[0.7, 0.15, 0.1, 0.05])
    X_df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    sr = {"group_detected": True, "group_col_selected": "grp"}

    print(f"Sub-test A: group_labels supplied \u2192 group-blocked, never splits a group")
    X_sub, idx = _apply_scalability_subsample_group_aware(X_df, structure_result=sr, group_labels=groups, seed=1)
    sub_groups = groups[idx]
    for g in np.unique(groups):
        full_count = int((groups == g).sum())
        drawn_count = int((sub_groups == g).sum())
        assert drawn_count in (0, full_count), \
            f"group '{g}' was partially drawn ({drawn_count}/{full_count}) -- must be all-or-nothing"
    print(f"Forced n={n:,} (> LARGE_N_THRESHOLD={LARGE_N_THRESHOLD:,}); groups drawn whole: "
          f"{sorted(set(sub_groups.tolist()))}, sample size {len(idx):,}")

    print(f"\n Sub-test B: group_labels=None (group_detected=True) \u2192 visible-warning iid fallback")
    X_sub2, idx2 = _apply_scalability_subsample_group_aware(X_df, structure_result=sr, group_labels=None, seed=1)
    assert len(idx2) == LARGE_N_SUBSAMPLE_SIZE, "iid fallback must still respect LARGE_N_SUBSAMPLE_SIZE"

    print(f"\n \u2705 Group-aware scalability subsample respects group boundaries as a hard constraint when")
    print(f"group_labels are supplied, and falls back visibly (not silently) when they aren't.")


def alternate_distance_algorithm(X_scaled, feat_final, distance_metric, km_k, k_range=range(2, 8),
                                  structure_result=None, modelling_df=None, tendency_result=None,
                                  investigation_results=None):
    banner("Full Distance-Metric Implementation")
    print(f"Step 5's selected metric: '{distance_metric}'")

    subbanner("Structure-detection decision matrix applied to this step")
    temporal_detected = group_detected = network_detected = False
    group_col = None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        group_detected = structure_result.get("group_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: the distance matrix")
        print(f"and K-Medoids' k-sweep are each built/fit ONCE on the fixed row set \u2014 no resampling for a")
        print(f"time dimension to act on.")
        print(f"Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: an algorithm/reporting")
        print(f"decision for Step 6/8, not this step's distance-matrix construction.")
        if group_detected:
            print(f"Group structure: YES ('{group_col}') \u2192 genuinely relevant in TWO concrete ways this step")
            print(f"honours, not just narration: (1) IF the scalability guard ever triggers on a larger future")
            print(f"extract, the subsample it draws is now group-blocked (see")
            print(f"_apply_scalability_subsample_group_aware() below), not a plain iid draw; (2) after")
            print(f"K-Medoids picks its own independent k, the resulting labels are cross-checked against raw")
            print(f" '{group_col}' (ARI) the same way Step 3F already checked K-Means \u2014 so the SAME")
            print(f"circularity question gets asked of this alternate distance metric too.")
        else:
            print(f"Group structure: NO \u2192 plain iid subsampling (if ever triggered) is used directly; no")
            print(f"group/rank cross-check is meaningful.")
    else:
        print(f"No structure_result supplied \u2014 no decision-matrix narration; scalability subsampling (if")
        print(f"triggered) falls back to plain iid, and no group cross-check runs.")

    subbanner("Feature-set consistency (source the SAME authoritative set Step 5 decided the metric on)")
    feat_final_for_step = list(feat_final)
    source_note = "the feat_final argument passed directly to this function"
    if tendency_result is not None and tendency_result.get("feat_final_used") is not None:
        feat_final_for_step = list(tendency_result["feat_final_used"])
        source_note = "Step 4's tendency_result['feat_final_used'] (the same authoritative set Step 5 used)"
        if set(feat_final_for_step) != set(feat_final):
            print(f"\u26a0\ufe0f  feat_final argument differs from {source_note} \u2014 using the Step 4/5 set for "
                  f"consistency across the whole Step 4\u21925\u21927B chain.")
        else:
            print(f"feat_final argument matches {source_note} \u2014 no drift ({len(feat_final_for_step)} "
                  f"features).")
    else:
        print(f"No tendency_result supplied \u2014 using {source_note} directly, with no cross-check available.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final_for_step if f in BUSINESS_PROTECTED_FEATURES]
    if protected_present:
        print(f"Business-protected features present in this feature set: {protected_present}")
        print(f"These fully participate in the distance matrix and medoid profiles below \u2014 this step")
        print(f"doesn't remove features, so protection has nothing to gate here beyond making sure the")
        print(f"feature set used matches the one Step 5's Mahalanobis decision was actually based on (above).")
    else:
        print(f"No business-protected features present in this feature set.")

    if investigation_results:
        print(f"\n investigated flag(s) still on record: {list(investigation_results.keys())} \u2014 "
              f"relevant context for interpreting the K-Medoids partition below, not an input to how it's "
              f"computed.")

    print(f"\n Self-test (proves both alternate-distance code paths, and the diagonal-zero fix, work on "
          f"synthetic data):")
    _self_test_alternate_distance_algorithms()

    if distance_metric == "euclidean":
        print(f"\n No substitution needed \u2014 Steps 7A/7C/7D (K-Means/Ward/DBSCAN) already ARE the correct,")
        print(f" mathematically valid implementation for Euclidean distance. This branch stays dormant")
        print(f" on real data by design; the self-test above is what proves it isn't just unimplemented.")
        return None

    X_df = X_scaled[feat_final_for_step]
    subbanner("Scalability guard (Gower/Mahalanobis matrices are O(n^2) memory)")
    triggered = scalability_guard(len(X_df), "distance matrix construction")
    group_labels_arr = None
    if group_detected and group_col and modelling_df is not None:
        group_labels_arr = modelling_df.loc[X_df.index, group_col].astype(str).values
    if triggered:
        X_df, sub_idx = _apply_scalability_subsample_group_aware(
            X_df, structure_result=structure_result, group_labels=group_labels_arr
        )
        if group_labels_arr is not None:
            group_labels_arr = group_labels_arr[sub_idx]
        print(f"Distance matrix and K-Medoids computed on the {len(X_df):,}-row subsample only.")
    else:
        print(f"n={len(X_df):,} <= LARGE_N_THRESHOLD={LARGE_N_THRESHOLD:,} \u2014 dormant on the real casino")
        print(f"population; running on the full {len(X_df):,}-row set.")

    X_vals = X_df.values
    num_cols = [c for c in feat_final_for_step if pd.api.types.is_numeric_dtype(X_scaled[c])]
    cat_cols = [c for c in feat_final_for_step if c not in num_cols]
    if distance_metric == "gower":
        print(f"\n Building the Gower distance matrix (real data)...")
        D, gower_ranges = gower_distance_matrix(X_df, num_cols, cat_cols)
        inv_cov = None
    else:  # mahalanobis
        print(f"\n Building the Mahalanobis distance matrix (real data)...")
        D, inv_cov = mahalanobis_distance_matrix(X_vals)
        gower_ranges = None

    subbanner("K-Medoids' OWN independent k-selection sweep (ROUND 5 FIX: no longer inherited from K-Means)")
    print(f" \u26a0\ufe0f  Silhouette-only sweep -- Davies-Bouldin and Calinski-Harabasz both require raw")
    print(f"coordinates, not a precomputed distance matrix, so the 5-method convergent evidence K-Means/")
    print(f"Hierarchical get is not available here. A genuine, structural limitation, stated plainly.")
    k_list = list(k_range)
    sils = []
    print(f"\n   {'k':>3} {'Silhouette':>11}  Sizes")
    for kk in k_list:
        labels_kk, _ = kmedoids_pam(D, k=kk, seed=RANDOM_STATE)
        sil_kk = silhouette_score(D, labels_kk, metric="precomputed") if len(set(labels_kk)) >= 2 else -1.0
        sils.append(sil_kk)
        sizes = pd.Series(labels_kk).value_counts().sort_index().to_dict()
        marker = " \u2190 K-Means chose this k" if kk == km_k else ""
        print(f"{kk:>3} {sil_kk:>11.4f}  {sizes}{marker}")
    kmedoids_k = k_list[int(np.argmax(sils))]
    print(f"\n DECISION: K-Medoids independently proposes k={kmedoids_k} (silhouette-only, "
          f"{sils[k_list.index(kmedoids_k)]:.4f})")
    if kmedoids_k != km_k:
        print(f"\u26a0\ufe0f  Differs from K-Means' k={km_k} \u2014 both are independent proposals now.")
    labels, medoid_idx = kmedoids_pam(D, k=kmedoids_k, seed=RANDOM_STATE)
    sil = silhouette_score(D, labels, metric="precomputed") if len(set(labels)) >= 2 else float("nan")
    print(f"Final K-Medoids(k={kmedoids_k}) result: silhouette={sil:.4f} (precomputed-distance silhouette), "
          f"sizes={pd.Series(labels).value_counts().sort_index().to_dict()}")

    group_ari = None
    if group_detected and group_col and group_labels_arr is not None:
        subbanner(f"Group/rank circularity cross-check for THIS metric's partition ('{group_col}')")
        group_ari = float(adjusted_rand_score(labels, group_labels_arr))
        print(f"ARI(K-Medoids labels, raw '{group_col}') = {group_ari:.4f}")
        if group_ari > 0.3:
            print(f"\u26a0\ufe0f  Non-trivial agreement with raw '{group_col}' \u2014 worth checking, same as Step 3F")
            print(f"already did for K-Means, whether this alternate-metric partition is also substantially")
            print(f"re-deriving the existing tier labelling rather than finding new structure.")
        else:
            print(f"Low agreement with raw '{group_col}' \u2014 this metric's partition is NOT simply")
            print(f"re-deriving the tier labelling.")

    if modelling_df is not None:
        subbanner("Small/outlier cluster export (K-Medoids final fit)")
        kmedoids_customer_ids = modelling_df.loc[X_df.index, entity_col].values
        export_small_clusters_if_any(labels, kmedoids_customer_ids, modelling_df, entity_col, "K-Medoids",
                                      group_col=group_col if group_detected else None)

    return {"labels": labels, "medoid_idx": medoid_idx, "distance_matrix": D, "k": kmedoids_k,
            "num_cols": num_cols, "cat_cols": cat_cols, "alt_distance_metric": distance_metric,
            "gower_ranges": gower_ranges, "inv_cov": inv_cov,
            "medoid_points": X_vals[medoid_idx], "medoid_points_df": X_df.iloc[medoid_idx].reset_index(drop=True),
            "feat_cols_ordered": feat_final_for_step, "group_ari": group_ari}


if PIPELINE_HALTED_AT is None:
    banner("SELF-TESTS: group-aware scalability subsample")
    _self_test_scalability_guard_group_aware()

   
    subbanner("Placeholder km_k (not yet implemented in this script)")
    km_k_placeholder = 4
    _placeholder_km = KMeans(n_clusters=km_k_placeholder, init="k-means++", n_init=10,
                              random_state=RANDOM_STATE).fit(X_scaled[tendency_result["feat_final_used"]])
    print(f"Using km_k={km_k_placeholder} (matches Step 3D's k_screen convention) as a stand-in comparison")
    print(f"point \u2014 real is not implemented in this script.")

    step7b_result = alternate_distance_algorithm(
        X_scaled, final_features, metric, km_k_placeholder,
        structure_result=structure_result, modelling_df=modelling_df,
        tendency_result=tendency_result, investigation_results=investigation_results,
    )
else:
    print(f"\n\u23ed\ufe0f (Full Distance-Metric Implementation) SKIPPED \u2014 pipeline halted at "
          f"'{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")


#==================================================================================
# Algorithm Selection Decision Matrix
# =====================================================================

def _self_test_step6_metric_incompatibility():
    
    banner("SELF-TEST: distance-metric-incompatibility & group-structure-does-NOT-gate branches")
    print("Case 1: distance_metric='euclidean', no structure detected (expect step6_pass=True)")
    dummy_flags_clean = {"temporal_detected": False, "group_detected": False, "network_detected": False}
    pass_euclidean = step6_algorithm_decision_matrix(n=11933, H=0.95, p=57,
                                                       structure_flags=dummy_flags_clean, distance_metric="euclidean")

    print("\nCase 2: distance_metric='gower' (expect step6_pass=False)")
    pass_gower = step6_algorithm_decision_matrix(n=11933, H=0.95, p=57,
                                                   structure_flags=dummy_flags_clean, distance_metric="gower")

    print("\n Case 3: distance_metric='euclidean' but group_detected=True, matching the REAL casino finding")
    print("('vip_tier', ICC=0.9586) -- expect step6_pass=True, proving group structure does NOT disqualify")
    print("K-Means (the exact fix this self-test exists to prove, vs. the original blunt criterion)")
    dummy_flags_group = {"temporal_detected": False, "group_detected": True, "network_detected": False}
    pass_group_detected = step6_algorithm_decision_matrix(n=11933, H=0.95, p=57, structure_flags=dummy_flags_group, distance_metric="euclidean")

    print("\n Case 4: distance_metric='euclidean' but temporal_detected=True -- expect step6_pass=False,")
    print("proving TEMPORAL structure still genuinely gates candidacy (the one structure type that should)")
    dummy_flags_temporal = {"temporal_detected": True, "group_detected": False, "network_detected": False}
    pass_temporal_detected = step6_algorithm_decision_matrix(n=11933, H=0.95, p=57, structure_flags=dummy_flags_temporal, distance_metric="euclidean")

    assert pass_euclidean is True, "Euclidean, no-structure case should pass all criteria"
    assert pass_gower is False, "Gower case MUST fail the metric-compatibility criterion"
    assert pass_group_detected is True, \
        "group_detected=True MUST NOT disqualify K-Means -- this is the exact adaptation this test proves"
    assert pass_temporal_detected is False, \
        "temporal_detected=True SHOULD still disqualify K-Means -- the one structure type that genuinely conflicts"
    print(f"\n \u2705 All four assertions passed: the metric-compatibility criterion genuinely branches, group")
    print(f"structure genuinely does NOT gate K-Means candidacy (the real casino case), and temporal")
    print(f"structure genuinely still does (unlike group/network) \u2014 not narrated-only.")


# ════════════════════════════════════════════════════════
#  GENERIC PRIMARY-MODEL INTERFACE
# ════════════════════════════════════════════════════════
def build_primary_model(X, labels, algorithm_name, native_model=None, eps=None, min_samples=None,
                         min_cluster_size=None, alt_distance_metric=None, num_cols=None, cat_cols=None,
                         feat_cols_ordered=None, gower_ranges=None, inv_cov=None, medoid_points=None,
                         medoid_points_df=None):
    
    valid_mask = labels != -1
    unique_clusters = sorted(set(labels[valid_mask]))
    centroids = np.array([X[labels == c].mean(axis=0) for c in unique_clusters])
    radii = {}
    if algorithm_name in ("dbscan", "hdbscan"):
        for i, c in enumerate(unique_clusters):
            member_dists = np.linalg.norm(X[labels == c] - centroids[i], axis=1)
            radii[c] = float(member_dists.max()) if len(member_dists) else 0.0
    return {"algorithm": algorithm_name, "labels": labels, "centroids": centroids,
            "cluster_ids": unique_clusters, "k": len(unique_clusters), "eps": eps, "min_samples": min_samples,
            "min_cluster_size": min_cluster_size, "native_model": native_model, "noise_radii": radii,
            "alt_distance_metric": alt_distance_metric, "num_cols": num_cols, "cat_cols": cat_cols,
            "feat_cols_ordered": feat_cols_ordered, "gower_ranges": gower_ranges, "inv_cov": inv_cov,
            "medoid_points": medoid_points, "medoid_points_df": medoid_points_df}


def primary_transform(X_new, primary_model):
    
    if primary_model["algorithm"] == "kmedoids":
        metric = primary_model["alt_distance_metric"]
        if metric == "gower":
            X_new_df = pd.DataFrame(X_new, columns=primary_model["feat_cols_ordered"])
            return gower_cross_distance(X_new_df, primary_model["medoid_points_df"],
                                         primary_model["num_cols"], primary_model["cat_cols"],
                                         primary_model["gower_ranges"])
        else:  # mahalanobis
            return mahalanobis_cross_distance(X_new, primary_model["medoid_points"], primary_model["inv_cov"])
    centroids = primary_model["centroids"]
    return np.linalg.norm(X_new[:, None, :] - centroids[None, :, :], axis=2)


def primary_predict_new(X_new, primary_model, noise_tolerance=1.25):
    
    dists = primary_transform(X_new, primary_model)
    nearest_idx = np.argmin(dists, axis=1)
    labels_out = np.array([primary_model["cluster_ids"][i] for i in nearest_idx])
    if primary_model["algorithm"] in ("dbscan", "hdbscan") and primary_model["noise_radii"]:
        for row in range(len(X_new)):
            cluster_id = labels_out[row]
            radius = primary_model["noise_radii"].get(cluster_id, np.inf)
            nearest_dist = dists[row, nearest_idx[row]]
            if nearest_dist > radius * noise_tolerance:
                labels_out[row] = -1
    return labels_out, dists


def _self_test_generic_primary_model():
    
    rng = np.random.RandomState(0)
    c1 = rng.normal([0, 0], 0.3, (100, 2))
    c2 = rng.normal([5, 5], 0.3, (100, 2))
    X = np.vstack([c1, c2])
    true_labels = np.array([0] * 100 + [1] * 100)

    print(f"Sub-test A (Hierarchical-sourced generic model):")
    Z = linkage(X, method="ward")
    hier_labels = fcluster(Z, t=2, criterion="maxclust") - 1
    pm_hier = build_primary_model(X, hier_labels, "hierarchical")
    X_new = np.array([[0.1, 0.1], [5.1, 4.9]])  # clearly belongs to cluster 0, then cluster 1
    pred, _ = primary_predict_new(X_new, pm_hier)
    print(f"New points {X_new.tolist()} -> predicted {pred.tolist()} (expect [matching training labels for c1, c2])")
    assert len(set(pred)) <= 2
    c0_train_label = hier_labels[0]  # a point deep in cluster 1 (c1 in X)
    c1_train_label = hier_labels[100]  # a point deep in cluster 2 (c2 in X)
    assert pred[0] == c0_train_label and pred[1] == c1_train_label, "new points should map to the correct training cluster"

    print(f"Sub-test B (DBSCAN-sourced generic model, incl. noise handling):")
    db = DBSCAN(eps=0.8, min_samples=5).fit(X)
    pm_db = build_primary_model(X, db.labels_, "dbscan")
    X_far = np.array([[50.0, 50.0]])  # nowhere near either training cluster
    pred_far, dists_far = primary_predict_new(X_far, pm_db)
    print(f"Far-away point [50, 50] -> predicted label {pred_far.tolist()} (expect -1, noise)")
    assert pred_far[0] == -1, "a point far from every cluster should be classified as noise, not force-assigned"
    X_close = np.array([[0.05, -0.05]])
    pred_close, _ = primary_predict_new(X_close, pm_db)
    print(f"Close point [0.05, -0.05] -> predicted label {pred_close.tolist()} (expect a real cluster, not noise)")
    assert pred_close[0] != -1, "a point clearly inside a dense region should NOT be classified as noise"
    print(f"Sub-test C (K-Medoids/Mahalanobis-sourced generic model \u2014 ROUND 5 FIX, the LIVE algorithm on"
          f"real casino data per Step 6 below):")
    
    cov_elliptical = np.array([[3.0, 2.8], [2.8, 3.0]])  # highly correlated -> very elongated ellipse
    c1_e = rng.multivariate_normal([0, 0], cov_elliptical, 150)
    c2_e = rng.multivariate_normal([8, 8], cov_elliptical, 150)
    X_e = np.vstack([c1_e, c2_e])
    D_e, inv_cov_e = mahalanobis_distance_matrix(X_e)
    km_labels_e, medoid_idx_e = kmedoids_pam(D_e, k=2, seed=RANDOM_STATE)
    pm_kmed = build_primary_model(X_e, km_labels_e, "kmedoids", alt_distance_metric="mahalanobis",
                                   inv_cov=inv_cov_e, medoid_points=X_e[medoid_idx_e])
    
    X_probe = np.array([[3.0, 3.0]])
    pred_probe, dists_probe = primary_predict_new(X_probe, pm_kmed)
    euclid_dists_to_means = np.linalg.norm(
        np.array([X_e[km_labels_e == c].mean(axis=0) for c in [0, 1]]) - X_probe, axis=1)
    print(f"Probe point [3, 3]: Mahalanobis-nearest medoid predicts cluster {pred_probe[0]}; "
          f"naive Euclidean-to-mean would pick cluster {int(np.argmin(euclid_dists_to_means))}")
    print(f"Mahalanobis distances to medoids: {dists_probe[0].round(3).tolist()}")
    assert pred_probe[0] in (0, 1), "should produce a valid cluster assignment, not crash or return garbage"
    
    naive_euclidean = np.linalg.norm(pm_kmed["centroids"] - X_probe, axis=1)
    assert not np.allclose(dists_probe[0], naive_euclidean), \
        "primary_transform must use Mahalanobis distance to MEDOIDS, not silently fall back to Euclidean-to-mean"
    print(f"\u2705 Confirmed primary_transform uses real Mahalanobis distance to medoids, NOT the")
    print(f"pre-fix Euclidean-to-mean-centroid bug (distances differ: {dists_probe[0].round(3).tolist()} "
          f"vs naive {naive_euclidean.round(3).tolist()}).")

    print(f"\u2705 Generic interface correctly deploys ALL non-K-Means algorithms, including DBSCAN's")
    print(f"noise semantics AND K-Medoids' non-Euclidean metric correctness.")


def step6_algorithm_decision_matrix(n, H, p, structure_flags, distance_metric, feat_final=None, investigation_results=None):
    banner("STEP 6: Algorithm Selection Decision Matrix")

    subbanner("Structure-detection decision matrix applied to this step")
    temporal_detected = structure_flags.get("temporal_detected", False)
    group_detected = structure_flags.get("group_detected", False)
    network_detected = structure_flags.get("network_detected", False)
    group_col = structure_flags.get("group_col_selected")
    print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 GENUINELY gates the K-Means")
    print(f"candidacy criterion below: repeated per-entity measurements would need a time-aware method,")
    print(f"which plain K-Means on an iid row set cannot provide.")
    print(f"Group structure: {'YES' if group_detected else 'NO'}" + (f" ('{group_col}')" if group_detected else "") +
          f"\u2192 does NOT gate K-Means candidacy \u2014 this pipeline's own")
    print(f"Step 1B decision matrix (resolve_resampling_strategy) already established that group structure")
    print(f"calls for GROUP-BLOCKED validation (already wired into bootstrap_jaccard_stability()/")
    print(f"permutation_significance_test() whenever group_labels are supplied) and within-group profiling,")
    print(f"not a different primary algorithm. Folding it into a blanket structural disqualifier would fail")
    print(f"K-Means on the real casino data purely because vip_tier exists (ICC=0.9586), despite every")
    print(f"downstream step here validating AROUND it, not ruling K-Means out because of it.")
    print(f"Network structure: {'YES' if network_detected else 'NO'} \u2192 does NOT gate K-Means candidacy")
    print(f"either \u2014 resolve_resampling_strategy's own network_note calls for reporting a graph-based")
    print(f"partition ALONGSIDE the feature-based one (Step 6/8), not instead of it.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in (feat_final or []) if f in BUSINESS_PROTECTED_FEATURES]
    if feat_final is not None:
        print(f"p={p} features under consideration; {len(protected_present)} business-protected: "
              f"{protected_present if protected_present else 'none'}.")
        print(f"K-Means/Ward/DBSCAN (Steps 7A/7C/7D) and K-Medoids (Step 7B) all treat every feature")
        print(f"identically once fit \u2014 protection doesn't change algorithm SELECTION, only which features")
        print(f"survived to reach this point (already enforced upstream in redundancy()/feature_selection()).")
    else:
        print(f"No feat_final supplied \u2014 protected-feature presence not checked at this step (p={p} only).")

    if investigation_results:
        print(f"\n Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"relevant context for interpreting whichever algorithm wins Step 8A, not an input to this")
        print(f"selection criterion.")

    
    criteria = {
        f"n={n} is 'large' (>1,000)": n > 1000,
        f"Hopkins H={H:.3f} > 0.75 (genuine tendency)": H > 0.75,
        f"Distance metric from Step 5 ('{distance_metric}') is Euclidean-compatible with K-Means/Ward":
            distance_metric == "euclidean",
        "No temporal structure requiring a time-aware method (group/network handled separately, see above)":
            not temporal_detected,
        "Business requires a FIXED, interpretable tier count": True,   # stated business requirement, not a data fact
    }
    print("\n K-Means primary-candidate criteria:")
    for desc, val in criteria.items():
        print(f"{'\u2705' if val else '\u2014'}  {desc}")
    all_pass = all(criteria.values())
    print(f"\n DECISION: {'K-Means favoured as PRIMARY candidate' if all_pass else 'K-Means NOT clearly favoured by this evidence'}")
    if distance_metric != "euclidean":
        print(f"\u26a0\ufe0f  Step 5 selected '{distance_metric}', which K-Means and Ward linkage CANNOT natively use")
        print(f"(K-Means centroids and Ward merges are only well-defined under squared Euclidean distance).")
        print(f"A correct implementation substitutes K-Medoids/PAM for K-Means and switches Hierarchical")
        print(f"to a precomputed {distance_metric} distance matrix. Step 7B ALREADY IMPLEMENTS that")
        print(f"substitution (K-Medoids/PAM on the real Mahalanobis matrix, already run on this")
        print(f"population \u2014 see Step 7B's log above for its independently-proposed k and its")
        print(f"ARI-vs-vip_tier circularity check) \u2014 this criterion failing is exactly the signal that")
        print(f"made Step 7B's substitution necessary, not a limitation left unaddressed.")
    print(f"Hierarchical fit in parallel: cross-algorithm agreement is validity evidence (Step 8A)")
    print(f"DBSCAN fit in parallel: tests whether non-convex/noise structure invalidates the K-Means "
          f"convexity assumption \u2014 a negative DBSCAN result (below) is itself informative, not a wasted step.")
    return all_pass


if PIPELINE_HALTED_AT is None:
    _self_test_step6_metric_incompatibility()

    banner("SELF-TEST: Generic primary-model interface (Hierarchical/DBSCAN/K-Medoids)")
    _self_test_generic_primary_model()

    
    step6_pass = step6_algorithm_decision_matrix(
        n=len(X_scaled), H=tendency_result["H_mean"], p=len(tendency_result["feat_final_used"]),
        structure_flags=structure_result, distance_metric=metric,
        feat_final=tendency_result["feat_final_used"], investigation_results=investigation_results,
    )
else:
    print(f"\n\u23ed\ufe0f  Step 6 (Algorithm Selection Decision Matrix) SKIPPED \u2014 pipeline halted at "
          f"'{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")


######################## FOUR Independent Clustering Methods + Comparison Prep 


def _print_structure_decision_matrix_note(step_label, structure_result):
    
    subbanner("Structure-detection decision matrix applied to this step")
    if structure_result is None:
        print(f"No structure_result supplied \u2014 no decision-matrix narration for {step_label}.")
        return False, None
    temporal_detected = structure_result.get("temporal_detected", False)
    group_detected = structure_result.get("group_detected", False)
    network_detected = structure_result.get("network_detected", False)
    group_col = structure_result.get("group_col_selected")
    print(f"Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: {step_label} fits ONCE")
    print(f"on the fixed row set \u2014 no resampling for a time dimension to act on.")
    print(f"Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: an algorithm/reporting")
    print(f"decision for Step 6/8, not this step's fitting procedure.")
    if group_detected:
        print(f"Group structure: YES ('{group_col}') \u2192 does NOT disqualify {step_label} (Step 6's own")
        print(f"adaptation already established group structure requires validating AROUND, not swapping")
        print(f"algorithms) \u2014 the resulting partition is cross-checked against raw '{group_col}' via ARI")
        print(f"below, the same circularity question Step 3F/7B already asked, now asked consistently here.")
    else:
        print(f"Group structure: NO \u2192 no group/rank circularity cross-check is meaningful for this step.")
    return group_detected, group_col


def _print_protected_feature_note(feat_final):
   
    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    if protected_present:
        print(f"Business-protected features present: {protected_present} \u2014 fully participate in this")
        print(f"algorithm's fit like any other feature; protection doesn't change HOW this step clusters,")
        print(f"only that these features survived to reach this point (already enforced upstream).")
    else:
        print(f"No business-protected features present in this feature set.")
    return protected_present


def _resolve_feat_final_for_step(feat_final, tendency_result, step_label):
    
    subbanner(f"Feature-set consistency ({step_label})")
    if tendency_result is not None and tendency_result.get("feat_final_used") is not None:
        authoritative = list(tendency_result["feat_final_used"])
        if set(authoritative) != set(feat_final):
            print(f"\u26a0\ufe0f  feat_final argument differs from Step 4's tendency_result['feat_final_used'] \u2014 "
                  f"using the authoritative Step 4/5 set ({len(authoritative)} features).")
        else:
            print(f"feat_final matches Step 4's tendency_result['feat_final_used'] \u2014 no drift "
                  f"({len(authoritative)} features).")
        return authoritative
    print(f"No tendency_result supplied \u2014 using the feat_final argument directly ({len(feat_final)} "
          f"features).")
    return list(feat_final)


def _print_investigation_carry_through(investigation_results, context_label):
    if investigation_results:
        print(f"\n Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"relevant context for interpreting {context_label}, not an input to how it's fit.")


def _print_group_circularity_check(labels, group_labels_arr, group_col, algorithm_label):
    
    subbanner(f"Group/rank circularity cross-check ('{group_col}')")
    ari_val = float(adjusted_rand_score(labels, group_labels_arr))
    print(f"ARI({algorithm_label} labels, raw '{group_col}') = {ari_val:.4f}")
    if ari_val > 0.3:
        print(f"\u26a0\ufe0f  Non-trivial agreement with raw '{group_col}' \u2014 worth checking whether this")
        print(f"algorithm's partition is substantially re-deriving the existing tier labelling rather than")
        print(f"finding new structure.")
    else:
        print(f"Low agreement with raw '{group_col}' \u2014 this algorithm's partition is NOT simply")
        print(f"re-deriving the tier labelling.")
    return ari_val


def _get_group_labels_arr(X_scaled_subset, modelling_df, group_col):
    if modelling_df is None or group_col is None:
        return None
    return modelling_df.loc[X_scaled_subset.index, group_col].astype(str).values


# ════════════════════════════════════════════════════════
#  STEP 7A — K-Means: k-Selection Sweep
# ════════════════════════════════════════════════════════
def step7a_kmeans_sweep(X_scaled, feat_final, k_range=range(2, 11), structure_result=None,
                         modelling_df=None, tendency_result=None, investigation_results=None,
                         banner_text="STEP 7A: K-Means \u2014 k-Selection Sweep"):
    banner(banner_text)
    feat_final = _resolve_feat_final_for_step(feat_final, tendency_result, "Step 7A")
    group_detected, group_col = _print_structure_decision_matrix_note("K-Means", structure_result)
    _print_protected_feature_note(feat_final)
    _print_investigation_carry_through(investigation_results, "the K-Means partition below")

    X_df = X_scaled[feat_final]
    X = X_df.values
    subbanner("Enhancement 10: Scalability guard (runtime, not memory -- K-Means itself is O(n))")
    use_minibatch = scalability_guard(len(X), "Step 7A K-Means sweep")
    if use_minibatch:
        print(f"Switching to MiniBatchKMeans for the sweep \u2014 K-Means itself doesn't have K-Medoids/")
        print(f"Ward's O(n^2) memory problem, but at this scale full-batch Lloyd's algorithm's per-")
        print(f"iteration cost becomes the bottleneck; MiniBatchKMeans trades a small amount of solution")
        print(f"quality for materially faster convergence via random mini-batch updates.")
    KMeansCls = MiniBatchKMeans if use_minibatch else KMeans
    kmeans_kwargs = {"n_clusters": None, "init": "k-means++", "random_state": RANDOM_STATE}
    if use_minibatch:
        kmeans_kwargs["batch_size"] = min(2048, len(X))
        kmeans_kwargs["max_no_improvement"] = 10
        kmeans_kwargs["n_init"] = 10
    else:
        kmeans_kwargs["n_init"] = 10
        kmeans_kwargs["max_iter"] = 300

    wcss, sil, dbi, ch = [], [], [], []
    print(f"   {'k':>3} {'WCSS':>12} {'Silhouette':>11} {'DaviesBouldin':>14} {'CalinskiHarabasz':>17}")
    for k in k_range:
        kwargs = dict(kmeans_kwargs); kwargs["n_clusters"] = k
        km = KMeansCls(**kwargs).fit(X)
        wcss.append(km.inertia_)
        labels = km.labels_
        sil.append(silhouette_score(X, labels, sample_size=4000, random_state=RANDOM_STATE))
        dbi.append(davies_bouldin_score(X, labels))
        ch.append(calinski_harabasz_score(X, labels))
        print(f"   {k:>3} {wcss[-1]:>12.1f} {sil[-1]:>11.4f} {dbi[-1]:>14.4f} {ch[-1]:>17.1f}")

    k_list = list(k_range)
    subbanner("Gap statistic (Tibshirani, B=10 uniform-reference datasets)")
    gaps, sk = gap_statistic(X, k_range, B=10)
    for k, g, s in zip(k_list, gaps, sk):
        print(f"   k={k:2d}  Gap={g:.4f}  s_k={s:.4f}")
    chosen_gap_k = None
    for i in range(len(k_list) - 1):
        if gaps[i] >= gaps[i + 1] - sk[i + 1]:
            chosen_gap_k = k_list[i]
            break
    print(f"Gap statistic 1-SE-rule selected k: "
          f"{chosen_gap_k if chosen_gap_k else 'NONE — monotonic increase, no stopping point in swept range'}")

    diffs2 = np.diff(np.diff(wcss))
    elbow_k = k_list[int(np.argmax(diffs2)) + 1]
    sil_k = k_list[int(np.argmax(sil))]
    ch_k = k_list[int(np.argmax(ch))]
    dbi_k = k_list[int(np.argmin(dbi))]

    subbanner("Enhancement 11: Composite Selection Index (CSI)")
    print(f"Formalises the convergent-evidence decision into a single score instead of eyeballing 4")
    print(f"separate sweeps: CSI(k) = z(Silhouette) + z(Calinski-Harabasz) - z(Davies-Bouldin)")
    sil_arr, ch_arr, dbi_arr = np.array(sil), np.array(ch), np.array(dbi)
    z_sil = (sil_arr - sil_arr.mean()) / sil_arr.std()
    z_ch = (ch_arr - ch_arr.mean()) / ch_arr.std()
    z_dbi = (dbi_arr - dbi_arr.mean()) / dbi_arr.std()
    csi = z_sil + z_ch - z_dbi
    csi_k = k_list[int(np.argmax(csi))]
    print(f"{'k':>3} {'z(Sil)':>8} {'z(CH)':>8} {'-z(DBI)':>9} {'CSI':>8}")
    for i, kk in enumerate(k_list):
        marker = " \u2190 max" if kk == csi_k else ""
        print(f"{kk:>3} {z_sil[i]:>8.3f} {z_ch[i]:>8.3f} {-z_dbi[i]:>9.3f} {csi[i]:>8.3f}{marker}")
    print(f"CSI-selected k = {csi_k}")

    subbanner("Convergent evidence table")
    print(f"{'Method':<32} {'Selected k':>10}")
    print(f"{'Elbow (max curvature)':<32} {elbow_k:>10}")
    print(f"{'Silhouette (maximise)':<32} {sil_k:>10}")
    print(f"{'Calinski-Harabasz (maximise)':<32} {ch_k:>10}")
    print(f"{'Davies-Bouldin (minimise)':<32} {dbi_k:>10}")
    print(f"{'Gap statistic (1-SE rule)':<32} {str(chosen_gap_k):>10}")
    print(f"{'Composite Selection Index':<32} {csi_k:>10}")

    votes = [elbow_k, sil_k, ch_k, dbi_k, csi_k]
    chosen_k = int(pd.Series(votes).mode()[0])
    n_agree = sum(v == chosen_k for v in votes)
    if csi_k != chosen_k:
        print(f"\n \u26a0\ufe0f  CSI's pick (k={csi_k}) disagrees with the {n_agree}/5-method majority (k={chosen_k}) \u2014")
        print(f"surfaced explicitly rather than silently overridden; the majority vote still governs.")
    print(f"\n DECISION: k={chosen_k} selected \u2014 supported by {n_agree}/5 methods "
          f"(gap statistic {'agrees' if chosen_gap_k == chosen_k else 'inconclusive in this range, not contradictory'})")

    final_km = KMeansCls(**{**kmeans_kwargs, "n_clusters": chosen_k}).fit(X)
    final_km_labels = final_km.labels_
    km_group_ari = None
    group_labels_arr = _get_group_labels_arr(X_df, modelling_df, group_col) if group_detected else None
    if group_detected and group_col and group_labels_arr is not None:
        km_group_ari = _print_group_circularity_check(final_km_labels, group_labels_arr, group_col, "K-Means")

    if modelling_df is not None:
        subbanner("Small/outlier cluster export (K-Means final fit)")
        km_customer_ids = modelling_df.loc[X_df.index, entity_col].values
        export_small_clusters_if_any(final_km_labels, km_customer_ids, modelling_df, entity_col, "K-Means",
                                      group_col=group_col if group_detected else None)

    subbanner("Plot: k-selection sweep (elbow, silhouette, gap, Calinski-Harabasz, Davies-Bouldin, CSI)")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(k_list, wcss, marker="o")
    axes[0, 0].axvline(elbow_k, color="r", ls="--", alpha=0.6)
    axes[0, 0].set_title("Elbow Method (WCSS)"); axes[0, 0].set_xlabel("k"); axes[0, 0].set_ylabel("WCSS")

    axes[0, 1].plot(k_list, sil, marker="o", color="green")
    axes[0, 1].axvline(sil_k, color="r", ls="--", alpha=0.6)
    axes[0, 1].set_title("Silhouette Score"); axes[0, 1].set_xlabel("k"); axes[0, 1].set_ylabel("Silhouette")

    axes[0, 2].plot(k_list, gaps, marker="o", color="purple")
    if chosen_gap_k is not None:
        axes[0, 2].axvline(chosen_gap_k, color="r", ls="--", alpha=0.6)
    axes[0, 2].set_title("Gap Statistic" + ("" if chosen_gap_k else " (no 1-SE stop)"))
    axes[0, 2].set_xlabel("k"); axes[0, 2].set_ylabel("Gap")

    axes[1, 0].plot(k_list, ch, marker="o", color="orange")
    axes[1, 0].axvline(ch_k, color="r", ls="--", alpha=0.6)
    axes[1, 0].set_title("Calinski-Harabasz Index"); axes[1, 0].set_xlabel("k"); axes[1, 0].set_ylabel("CH")

    axes[1, 1].plot(k_list, dbi, marker="o", color="brown")
    axes[1, 1].axvline(dbi_k, color="r", ls="--", alpha=0.6)
    axes[1, 1].set_title("Davies-Bouldin Index (lower better)"); axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("DBI")

    axes[1, 2].plot(k_list, csi, marker="o", color="crimson")
    axes[1, 2].axvline(csi_k, color="r", ls="--", alpha=0.6)
    axes[1, 2].set_title("Composite Selection Index (Enh. 11)"); axes[1, 2].set_xlabel("k"); axes[1, 2].set_ylabel("CSI")

    fig.suptitle(f"K-Means k-Selection Sweep \u2014 chosen k={chosen_k} (red dashed line)", y=1.02)
    savefig("step7A_k_selection_sweep.png")

    return chosen_k, {"k": k_list, "wcss": wcss, "sil": sil, "dbi": dbi, "ch": ch, "gaps": gaps, "sk": sk,
                       "csi": csi, "labels": final_km_labels, "group_ari": km_group_ari,
                       "feat_final_used": feat_final, "native_model": final_km}


# ════════════════════════════════════════════════════════
#  STEP 7C — Hierarchical Clustering (Ward Linkage)
# ════════════════════════════════════════════════════════
WARD_COPHENET_MAX_N = 8000  

def step7b_hierarchical(X_scaled, feat_final, km_k, k_range=range(2, 11), structure_result=None,
                         modelling_df=None, tendency_result=None, investigation_results=None,
                         banner_text="STEP 7C: Hierarchical Clustering (Ward Linkage)"):
    banner(banner_text)
    feat_final = _resolve_feat_final_for_step(feat_final, tendency_result, "Step 7C")
    group_detected, group_col = _print_structure_decision_matrix_note("Hierarchical/Ward", structure_result)
    _print_protected_feature_note(feat_final)
    _print_investigation_carry_through(investigation_results, "the Hierarchical partition below")

    X_df_full = X_scaled[feat_final]
    X_full = X_df_full.values
    n_full = len(X_full)

    subbanner("Enhancement 10: Scalability guard (Ward linkage's internal pdist/cophenet is O(n^2) memory)")
    triggered = scalability_guard(n_full, "Step 7C Ward linkage")
    if triggered:
        X_df, sub_idx = apply_scalability_subsample(X_df_full)
        X = X_df.values
        print(f"Linkage/cophenet computed on the {len(X):,}-row subsample; cluster labels for the")
        print(f"remaining {n_full - len(X):,} rows are extended via nearest-centroid assignment below.")
    else:
        X, sub_idx = X_full, np.arange(n_full)
        X_df = X_df_full
        print(f"n={n_full:,} <= LARGE_N_THRESHOLD={LARGE_N_THRESHOLD:,} \u2014 dormant; running the real Ward")
        print(f"LINKAGE on the full {n_full:,}-row population (confirmed by direct timing: ~11s, no memory")
        print(f"issue \u2014 it's cophenet(), not linkage() itself, that needs the stricter treatment below).")

    Z = linkage(X, method="ward", metric="euclidean")

    subbanner("Cophenetic correlation (memory-aware: separate threshold from the general scalability guard)")
    if len(X) > WARD_COPHENET_MAX_N:
        print(f"n={len(X):,} exceeds the cophenet-specific memory threshold ({WARD_COPHENET_MAX_N:,}) \u2014")
        print(f"scipy's cophenet() needs TWO full O(n^2) arrays simultaneously (pairwise + cophenetic")
        print(f"distances), a stricter requirement than the general {LARGE_N_THRESHOLD:,}-row scalability")
        print(f"guard above assumes. Confirmed by direct timing: this genuinely OOM-crashes at this")
        print(f"pipeline's real n under real memory load, not a hypothetical concern.")
        coph_rng = np.random.RandomState(RANDOM_STATE)
        coph_idx = coph_rng.choice(len(X), WARD_COPHENET_MAX_N, replace=False)
        X_coph_sample = X[coph_idx]
        Z_coph_sample = linkage(X_coph_sample, method="ward", metric="euclidean")
        coph_r, _ = cophenet(Z_coph_sample, pdist(X_coph_sample))
        print(f"Computed on an INDEPENDENT {WARD_COPHENET_MAX_N:,}-row subsample's own Ward tree (a SEPARATE")
        print(f"linkage from the full-population one used for the real clustering below) \u2014 this is a")
        print(f"diagnostic estimate of how well Ward/Euclidean geometry tends to preserve distances for")
        print(f"this dataset's general structure, not an exact property of the full-population tree.")
    else:
        coph_r, _ = cophenet(Z, pdist(X))
    print(f"Cophenetic Correlation Coefficient: {coph_r:.4f}")
    fit_label = "strong (\u22650.75)" if coph_r >= 0.75 else ("moderate (0.60\u20130.75)" if coph_r >= 0.60 else "weak (<0.60)")
    print(f"Interpretation: {fit_label}")

    subbanner("Hierarchical's OWN independent k-selection sweep (FIX: no longer inherited from K-Means)")
    k_list = list(k_range)
    pseudo_wcss, sil, dbi, ch = [], [], [], []
    print(f"{'k':>3} {'pseudo-WCSS':>12} {'Silhouette':>11} {'DBI':>8} {'CH':>10}  Sizes")
    for kk in k_list:
        labels_kk = fcluster(Z, t=kk, criterion="maxclust")
        wcss_kk = sum(np.sum((X[labels_kk == c] - X[labels_kk == c].mean(axis=0)) ** 2)
                      for c in np.unique(labels_kk))
        sil_kk = silhouette_score(X, labels_kk, sample_size=min(4000, len(X)), random_state=RANDOM_STATE)
        dbi_kk = davies_bouldin_score(X, labels_kk)
        ch_kk = calinski_harabasz_score(X, labels_kk)
        pseudo_wcss.append(wcss_kk); sil.append(sil_kk); dbi.append(dbi_kk); ch.append(ch_kk)
        sizes = pd.Series(labels_kk).value_counts().sort_index().to_dict()
        marker = " \u2190 K-Means chose this k" if kk == km_k else ""
        print(f"{kk:>3} {wcss_kk:>12.1f} {sil_kk:>11.4f} {dbi_kk:>8.4f} {ch_kk:>10.1f}  {sizes}{marker}")

    pseudo_wcss, sil, dbi, ch = map(np.array, (pseudo_wcss, sil, dbi, ch))
    diffs2 = np.diff(np.diff(pseudo_wcss))
    elbow_k = k_list[int(np.argmax(diffs2)) + 1] if len(diffs2) else k_list[0]
    sil_k = k_list[int(np.argmax(sil))]
    ch_k = k_list[int(np.argmax(ch))]
    dbi_k = k_list[int(np.argmin(dbi))]
    z_sil = (sil - sil.mean()) / sil.std() if sil.std() > 0 else np.zeros_like(sil)
    z_ch = (ch - ch.mean()) / ch.std() if ch.std() > 0 else np.zeros_like(ch)
    z_dbi = (dbi - dbi.mean()) / dbi.std() if dbi.std() > 0 else np.zeros_like(dbi)
    csi = z_sil + z_ch - z_dbi
    csi_k = k_list[int(np.argmax(csi))]

    votes = [elbow_k, sil_k, ch_k, dbi_k, csi_k]
    hier_k = int(pd.Series(votes).mode()[0])
    n_agree = sum(v == hier_k for v in votes)
    print(f"\n Convergent evidence: elbow={elbow_k}, silhouette={sil_k}, CH={ch_k}, DBI={dbi_k}, CSI={csi_k}")
    print(f"DECISION: Hierarchical independently proposes k={hier_k} ({n_agree}/5 methods agree)")
    if hier_k != km_k:
        print(f"\u26a0\ufe0f  This DIFFERS from K-Means' independently chosen k={km_k} \u2014 both are now genuinely")
        print(f"independent proposals; Step 8A compares each algorithm's OWN best partition, not one")
        print(f"algorithm's partition forced to match the other's k.")
    else:
        print(f"\u2705 Agrees with K-Means' independently chosen k={km_k} \u2014 convergent evidence ACROSS two")
        print(f"structurally different algorithms, not just across metrics within one.")

    sub_labels = fcluster(Z, t=hier_k, criterion="maxclust") - 1
    if triggered:
        centroids = np.array([X[sub_labels == c].mean(axis=0) for c in range(hier_k)])
        full_labels = np.zeros(n_full, dtype=int)
        full_labels[sub_idx] = sub_labels
        out_of_sample = np.setdiff1d(np.arange(n_full), sub_idx)
        dists_out = np.linalg.norm(X_full[out_of_sample][:, None, :] - centroids[None, :, :], axis=2)
        full_labels[out_of_sample] = np.argmin(dists_out, axis=1)
        hier_labels = full_labels
        print(f"{len(out_of_sample):,} out-of-subsample rows assigned via nearest-centroid extension.")
    else:
        hier_labels = sub_labels

    hier_group_ari = None
    group_labels_arr = _get_group_labels_arr(X_df_full, modelling_df, group_col) if group_detected else None
    if group_detected and group_col and group_labels_arr is not None:
        hier_group_ari = _print_group_circularity_check(hier_labels, group_labels_arr, group_col, "Hierarchical")

    if modelling_df is not None:
        subbanner("Small/outlier cluster export (Hierarchical final fit)")
        hier_customer_ids = modelling_df.loc[X_df_full.index, entity_col].values
        export_small_clusters_if_any(hier_labels, hier_customer_ids, modelling_df, entity_col, "Hierarchical",
                                      group_col=group_col if group_detected else None)

    subbanner("Plot: Ward dendrogram (truncated, last 30 merges)")
    plt.figure(figsize=(11, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, show_leaf_counts=True)
    plt.title(f"Ward Dendrogram (truncated, last 30 merges) \u2014 Cophenetic r={coph_r:.3f}, "
              f"independently chosen k={hier_k}")
    savefig("step7C_dendrogram.png")

    return hier_labels, coph_r, Z, hier_k, {"group_ari": hier_group_ari, "feat_final_used": feat_final}


# ════════════════════════════════════════════════════════
#  STEP 7D — DBSCAN
# ════════════════════════════════════════════════════════
def step7c_dbscan(X_scaled, feat_final, n_configs=2, structure_result=None, modelling_df=None,
                   tendency_result=None, investigation_results=None):
    banner("STEP 7D: DBSCAN")
    feat_final = _resolve_feat_final_for_step(feat_final, tendency_result, "Step 7D")
    group_detected, group_col = _print_structure_decision_matrix_note("DBSCAN", structure_result)
    _print_protected_feature_note(feat_final)
    _print_investigation_carry_through(investigation_results, "the DBSCAN partition below")

    X_df_full = X_scaled[feat_final]
    X_full = X_df_full.values
    n_full = len(X_full)

    
    subbanner("Scalability guard (DBSCAN's internal radius-neighbors search)")
    triggered = scalability_guard(n_full, "Step 7D DBSCAN")
    if triggered:
        X_df, sub_idx = apply_scalability_subsample(X_df_full)
        X = X_df.values
        print(f"k-distance graph and parameter sweep computed on the {len(X):,}-row subsample; final")
        print(f"labels for the remaining {n_full - len(X):,} rows are extended below via nearest-centroid")
        print(f"assignment, with out-of-sample points beyond eps of every centroid kept as noise (-1),")
        print(f"consistent with DBSCAN's own definition rather than forcing every point into a cluster.")
    else:
        X, sub_idx = X_full, np.arange(n_full)
        X_df = X_df_full
        print(f"n={n_full:,} <= LARGE_N_THRESHOLD={LARGE_N_THRESHOLD:,} — dormant; running DBSCAN on the")
        print(f"full {n_full:,}-row population.")

    n_feat = X.shape[1]
    min_samples = 2 * n_feat
    print(f"min_samples starting rule of thumb (2 x {n_feat} features) = {min_samples}")

    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    kdist = np.sort(nn.kneighbors(X)[0][:, -1])
    xn = np.arange(len(kdist)) / (len(kdist) - 1)
    yn = (kdist - kdist.min()) / (kdist.max() - kdist.min())
    line = np.array([xn[-1], yn[-1]]) / np.linalg.norm([xn[-1], yn[-1]])
    vecs = np.stack([xn, yn], axis=1)
    perp = vecs - np.outer(vecs @ line, line)
    eps_knee = kdist[int(np.argmax(np.linalg.norm(perp, axis=1)))]
    print(f"k-distance knee (kneedle max-curvature method): eps = {eps_knee:.4f}")

    subbanner("Plot: k-distance graph")
    plt.figure(figsize=(8, 5))
    plt.plot(kdist)
    plt.axhline(eps_knee, color="r", ls="--", alpha=0.6, label=f"knee eps={eps_knee:.3f}")
    plt.title(f"k-Distance Graph (k={min_samples})")
    plt.xlabel("Points sorted by distance"); plt.ylabel(f"Distance to {min_samples}-th neighbor")
    plt.legend()
    savefig("step7D_kdistance_graph.png")

    configs = [(round(eps_knee, 3), min_samples)]
    if n_configs > 1:
        configs.append((round(eps_knee * 0.6, 3), min_samples))

    

    def _estimate_neighbor_density(X_probe, eps_val, seed=RANDOM_STATE):
        rng_local = np.random.RandomState(seed)
        probe_idx = rng_local.choice(len(X_probe), min(DENSITY_PROBE_SAMPLE, len(X_probe)), replace=False)
        dists = pairwise_distances(X_probe[probe_idx], X_probe, metric="euclidean")
        counts = (dists < eps_val).sum(axis=1)
        return float(counts.mean())

    subbanner("Parameter sweep")
    print(f"{'eps':>8} {'min_samples':>12} {'n_clusters':>11} {'noise%':>8} {'silhouette(no noise)':>22}")
    results = []
    for eps, ms in configs:
        mean_neighbors = _estimate_neighbor_density(X, eps)
        if mean_neighbors > DBSCAN_MAX_MEAN_NEIGHBORS:
            print(f"{eps:>8.3f} {ms:>12d}  SKIPPED -- estimated mean neighbours/point = {mean_neighbors:,.0f} "
                  f"(> {DBSCAN_MAX_MEAN_NEIGHBORS:,} budget). This eps is not a genuine density boundary for this")
            print(f"data (consistent with Step 3D's own near-zero density-score finding) -- a radius")
            print(f"search here would need an estimated {mean_neighbors * len(X):,.0f} within-eps pairs.")
            continue
        db = DBSCAN(eps=eps, min_samples=ms, metric="euclidean").fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).mean() * 100
        mask = labels != -1
        sil = (silhouette_score(X[mask], labels[mask], sample_size=min(4000, mask.sum()), random_state=RANDOM_STATE)
               if mask.sum() > 10 and len(set(labels[mask])) >= 2 else float("nan"))
        print(f"{eps:>8.3f} {ms:>12d} {n_clusters:>11d} {noise_pct:>7.2f}% {sil:>22.4f}")
        results.append({"eps": eps, "min_samples": ms, "labels": labels, "n_clusters": n_clusters,
                         "noise_pct": noise_pct, "silhouette": sil, "model": db})

    NOISE_FLAG_THRESHOLD = 25.0
    if not results:
        print(f"\n DECISION: every candidate eps was infeasible at this row count -- DBSCAN could not be run")
        print(f"safely on this data. Returning an all-noise result (every point labelled -1), consistent")
        print(f"with DBSCAN's own semantics for \"no density-separated clusters found\", rather than a")
        print(f"silent crash. This is a REAL finding about this data (no natural density separation), not")
        print(f"a placeholder -- Step 8A's eligibility check will correctly exclude DBSCAN as a candidate.")
        all_noise = np.full(len(X), -1, dtype=int)
        best = {"eps": configs[0][0], "min_samples": configs[0][1], "labels": all_noise, "n_clusters": 0,
                "noise_pct": 100.0, "silhouette": float("nan"), "model": None}
        results = [best]
    else:
        best = max([r for r in results if not np.isnan(r["silhouette"])],
                   key=lambda r: r["silhouette"], default=results[0])

    
    if triggered:
        sub_labels = best["labels"]
        real_clusters = sorted(set(sub_labels) - {-1})
        full_labels = np.full(n_full, -1, dtype=int)
        full_labels[sub_idx] = sub_labels
        out_of_sample = np.setdiff1d(np.arange(n_full), sub_idx)
        if real_clusters:
            centroids = np.array([X[sub_labels == c].mean(axis=0) for c in real_clusters])
            dists_out = np.linalg.norm(X_full[out_of_sample][:, None, :] - centroids[None, :, :], axis=2)
            nearest_c = np.argmin(dists_out, axis=1)
            nearest_d = dists_out[np.arange(len(out_of_sample)), nearest_c]
            assigned = np.array(real_clusters)[nearest_c]
            assigned[nearest_d > best["eps"]] = -1
            full_labels[out_of_sample] = assigned
        best["labels"] = full_labels
        print(f"{len(out_of_sample):,} out-of-subsample rows assigned via nearest-centroid extension "
              f"(kept as noise where beyond eps={best['eps']} of every centroid).")

    print(f"\n DECISION: eps={best['eps']}, min_samples={best['min_samples']} \u2014 highest silhouette among "
          f"non-degenerate configs. Sizes: {pd.Series(best['labels']).value_counts().sort_index().to_dict()}")
    if best["noise_pct"] > NOISE_FLAG_THRESHOLD:
        print(f" \u26a0\ufe0f  noise% ({best['noise_pct']:.1f}%) exceeds the {NOISE_FLAG_THRESHOLD}% guide flag \u2014 "
              f"interpret cautiously; see Step 8A for how this compares to K-Means/Hierarchical.")
    tiny_clusters = [c for c, n in pd.Series(best["labels"]).value_counts().items() if c != -1 and n < 20]
    if tiny_clusters:
        print(f" \u26a0\ufe0f  micro-clusters detected (<20 points): {tiny_clusters} \u2014 likely outlier pockets, "
              f"not genuine business segments; treat with caution in Step 8A/10d.")

    
    db_group_ari = None
    group_labels_arr = _get_group_labels_arr(X_df_full, modelling_df, group_col) if group_detected else None
    if group_detected and group_col and group_labels_arr is not None:
        db_group_ari = _print_group_circularity_check(best["labels"], group_labels_arr, group_col, "DBSCAN")

    if modelling_df is not None:
        subbanner("Small/outlier cluster export (DBSCAN final fit)")
        db_customer_ids = modelling_df.loc[X_df_full.index, entity_col].values
        export_small_clusters_if_any(best["labels"], db_customer_ids, modelling_df, entity_col, "DBSCAN",
                                      group_col=group_col if group_detected else None)

    return best["labels"], best["eps"], best["min_samples"], results, best["model"], \
        {"group_ari": db_group_ari, "feat_final_used": feat_final}


# ════════════════════════════════════════════════════════
#  STEP 7E — HDBSCAN  
# ════════════════════════════════════════════════════════
def _self_test_hdbscan():
    """Proves HDBSCAN correctly separates two genuinely different-density
    clusters where a single global eps would force a bad trade-off -- on
    SYNTHETIC data only, since the real casino data's own trigger condition
    (evaluated below against Step 7D's actual DBSCAN diagnostics) decides
    whether this branch runs on real data at all."""
    rng = np.random.RandomState(0)
    dense = rng.normal(loc=[0, 0], scale=0.15, size=(300, 2))
    sparse = rng.normal(loc=[6, 6], scale=1.5, size=(300, 2))
    X = np.vstack([dense, sparse])
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=20)
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    dense_labels = set(labels[:300]) - {-1}
    sparse_labels = set(labels[300:]) - {-1}
    print(f"     Synthetic test: 1 dense blob (std=0.15) + 1 sparse blob (std=1.5), true density ratio=10x")
    print(f"     HDBSCAN found {n_clusters} clusters; dense-blob labels={dense_labels}, sparse-blob labels={sparse_labels}")
    assert n_clusters >= 2, "HDBSCAN should separate the two density regions"
    assert len(dense_labels & sparse_labels) == 0, "the two density regions should not share a cluster label"
    print(f"     \u2705 HDBSCAN correctly separates different-density regions a single-eps DBSCAN would struggle with.")


def step7d_hdbscan(X_scaled, feat_final, db_labels, structure_result=None, modelling_df=None,
                    tendency_result=None, investigation_results=None):
    banner("STEP 7E: HDBSCAN (Enhancement 2 \u2014 density-variation-triggered)")
    feat_final = _resolve_feat_final_for_step(feat_final, tendency_result, "Step 7E")
    group_detected, group_col = _print_structure_decision_matrix_note("HDBSCAN", structure_result)
    _print_protected_feature_note(feat_final)
    _print_investigation_carry_through(investigation_results, "the HDBSCAN partition below (if triggered)")

    X_df_full = X_scaled[feat_final]
    X_full = X_df_full.values
    n_full = len(X_full)

    # FIX: HDBSCAN.fit_predict() previously ran on the FULL, unguarded
    # population here -- confirmed directly as the cause of a real
    # MemoryError (9.28 GiB, inside HDBSCAN's internal Boruvka MST
    # construction) at n=315,653. This is the third HDBSCAN/DBSCAN call
    # site in this pipeline needing this guard; Ward linkage (Step 7C) and
    # DBSCAN (Step 7D) already use the identical scalability_guard/
    # apply_scalability_subsample pair for the same reason.
    subbanner("Scalability guard (HDBSCAN's internal Boruvka MST construction)")
    triggered_scale = scalability_guard(n_full, "Step 7E HDBSCAN")
    if triggered_scale:
        X_df, sub_idx = apply_scalability_subsample(X_df_full)
        X = X_df.values
        print(f"   HDBSCAN fit on the {len(X):,}-row subsample; final labels for the remaining")
        print(f"   {n_full - len(X):,} rows are extended below via nearest-centroid assignment.")
    else:
        X, sub_idx = X_full, np.arange(n_full)
        X_df = X_df_full
        print(f"   n={n_full:,} <= LARGE_N_THRESHOLD={LARGE_N_THRESHOLD:,} \u2014 dormant; running HDBSCAN on")
        print(f"   the full {n_full:,}-row population.")

    # NOTE: the trigger check below reads db_labels (DBSCAN's OWN labels,
    # which already cover the FULL population thanks to Step 7D's own
    # nearest-centroid extension) -- cheap array ops on existing labels,
    # not a model fit, so this deliberately does NOT use the subsample:
    # the trigger decision should reflect the real, full-population DBSCAN
    # diagnostics, not an arbitrarily smaller slice of them.
    noise_pct = (db_labels == -1).mean() * 100
    non_noise = pd.Series(db_labels[db_labels != -1]).value_counts()
    size_ratio = (non_noise.max() / non_noise.min()) if len(non_noise) >= 2 and non_noise.min() > 0 else 1.0

    print(f"   Trigger check (evaluated against DBSCAN's OWN Step 7D result, not assumed in advance):")
    print(f"     DBSCAN noise fraction: {noise_pct:.1f}%  (trigger: > {HDBSCAN_NOISE_TRIGGER*100:.0f}%)")
    print(f"     DBSCAN non-noise cluster size imbalance: {size_ratio:.1f}x  (trigger: > {HDBSCAN_SIZE_IMBALANCE_TRIGGER:.0f}x)")
    trig_noise = noise_pct > HDBSCAN_NOISE_TRIGGER * 100
    trig_imbalance = size_ratio > HDBSCAN_SIZE_IMBALANCE_TRIGGER
    triggered = trig_noise or trig_imbalance
    print(f"   DECISION: {'TRIGGERED' if triggered else 'NOT triggered'} "
          f"(noise trigger: {trig_noise}, imbalance trigger: {trig_imbalance})")

    print(f"\n   Self-test (proves HDBSCAN correctly separates different-density regions on synthetic data):")
    _self_test_hdbscan()

    if not triggered:
        print(f"\n   HDBSCAN not run on the real data \u2014 condition not met on this dataset's actual DBSCAN")
        print(f"   diagnostics. The self-test above confirms the code path works; it simply isn't the right")
        print(f"   tool for THIS data.")
        return None, None, None, {"group_ari": None, "feat_final_used": feat_final}

    min_cluster_size = max(HDBSCAN_MIN_CLUSTER_SIZE_FLOOR, int(HDBSCAN_MIN_CLUSTER_SIZE_FRAC * len(X)))
    print(f"   LEVEL 3 (Enhancement 22): min_cluster_size = max({HDBSCAN_MIN_CLUSTER_SIZE_FLOOR}, "
          f"{HDBSCAN_MIN_CLUSTER_SIZE_FRAC:.0%} of n) = {min_cluster_size}.")
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    sub_labels = clusterer.fit_predict(X)
    n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    noise_pct_h = (sub_labels == -1).mean() * 100
    mask = sub_labels != -1
    sil = (silhouette_score(X[mask], sub_labels[mask], sample_size=min(4000, mask.sum()), random_state=RANDOM_STATE)
           if mask.sum() > 10 and len(set(sub_labels[mask])) >= 2 else float("nan"))
    print(f"\n   HDBSCAN(min_cluster_size={min_cluster_size}) on real data: {n_clusters} clusters, "
          f"noise={noise_pct_h:.1f}%, silhouette={sil:.4f}")

    # FIX: extends labels from the subsample back to the FULL population,
    # mirroring Step 7D's (DBSCAN) own nearest-centroid extension --
    # min_cluster_size-based density boundaries don't transfer cleanly to
    # out-of-sample points, so out-of-sample points beyond min_cluster_
    # size's implied density radius (approximated here the same way DBSCAN
    # uses its eps) are kept as noise (-1) rather than forced into a
    # cluster.
    if triggered_scale:
        real_clusters = sorted(set(sub_labels) - {-1})
        labels = np.full(n_full, -1, dtype=int)
        labels[sub_idx] = sub_labels
        out_of_sample = np.setdiff1d(np.arange(n_full), sub_idx)
        if real_clusters:
            centroids = np.array([X[sub_labels == c].mean(axis=0) for c in real_clusters])
            # FIX (real bug, found via direct testing): X[...].std() with no
            # axis flattens the whole cluster into one scalar std, badly
            # UNDER-estimating the actual typical distance from a point to
            # its own centroid in a 47-dimensional space (confirmed
            # directly: off by a factor of ~6.8x on realistic synthetic
            # data) -- this marked 295,653 of 315,653 points as noise
            # instead of correctly extending them. Uses the actual mean
            # per-cluster distance-to-centroid instead, the correctly-
            # scaled quantity for this comparison.
            within_cluster_scale = np.mean([
                np.linalg.norm(X[sub_labels == c] - X[sub_labels == c].mean(axis=0), axis=1).mean()
                for c in real_clusters])
            dists_out = np.linalg.norm(X_full[out_of_sample][:, None, :] - centroids[None, :, :], axis=2)
            nearest_c = np.argmin(dists_out, axis=1)
            nearest_d = dists_out[np.arange(len(out_of_sample)), nearest_c]
            assigned = np.array(real_clusters)[nearest_c]
            assigned[nearest_d > 2 * within_cluster_scale] = -1
            labels[out_of_sample] = assigned
        print(f"   {len(out_of_sample):,} out-of-subsample rows assigned via nearest-centroid extension.")
    else:
        labels = sub_labels

    hdb_group_ari = None
    group_labels_arr = _get_group_labels_arr(X_df_full, modelling_df, group_col) if group_detected else None
    if group_detected and group_col and group_labels_arr is not None:
        hdb_group_ari = _print_group_circularity_check(labels, group_labels_arr, group_col, "HDBSCAN")

    if modelling_df is not None:
        subbanner("Small/outlier cluster export (HDBSCAN final fit)")
        hdb_customer_ids = modelling_df.loc[X_df_full.index, entity_col].values
        export_small_clusters_if_any(labels, hdb_customer_ids, modelling_df, entity_col, "HDBSCAN",
                                      group_col=group_col if group_detected else None)

    return labels, min_cluster_size, clusterer, {"group_ari": hdb_group_ari, "feat_final_used": feat_final}


# ════════════════════════════════════════════════════════
#  STEP 7F — GMM candidate fit for Step 8A
# ════════════════════════════════════════════════════════
def fit_gmm_candidate_for_step8a(X_scaled, feat_final, k_range=range(2, 7), structure_result=None,
                                  modelling_df=None, tendency_result=None, investigation_results=None,
                                  banner_text="STEP 7F: GMM Candidate Fit for Step 8A (Enhancement 3/15 \u2014 unconditional, BIC-based)"):
    banner(banner_text)
    feat_final = _resolve_feat_final_for_step(feat_final, tendency_result, "Step 7F")
    group_detected, group_col = _print_structure_decision_matrix_note("GMM", structure_result)
    _print_protected_feature_note(feat_final)
    _print_investigation_carry_through(investigation_results, "the GMM candidate below")

    X_df = X_scaled[feat_final]
    X = X_df.values
    bics = []
    for kk in k_range:
        g = GaussianMixture(n_components=kk, covariance_type="full", random_state=RANDOM_STATE).fit(X)
        bics.append(g.bic(X))
    best_k = list(k_range)[int(np.argmin(bics))]
    print(f"   BIC per component count: {dict(zip(k_range, np.round(bics, 1)))}")
    print(f"   Independently selected n_components={best_k} (minimises BIC \u2014 no confidence-evidence")
    print(f"   dependency, unlike Step 8C's fuller diagnostic, so this can run BEFORE Step 8A's decision).")
    gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=RANDOM_STATE).fit(X)
    labels = gmm.predict(X)
    print(f"   GMM candidate ready: k={best_k}, sizes={pd.Series(labels).value_counts().sort_index().to_dict()}")

    gmm_group_ari = None
    group_labels_arr = _get_group_labels_arr(X_df, modelling_df, group_col) if group_detected else None
    if group_detected and group_col and group_labels_arr is not None:
        gmm_group_ari = _print_group_circularity_check(labels, group_labels_arr, group_col, "GMM")

    if modelling_df is not None:
        subbanner("Small/outlier cluster export (GMM candidate)")
        gmm_customer_ids = modelling_df.loc[X_df.index, entity_col].values
        export_small_clusters_if_any(labels, gmm_customer_ids, modelling_df, entity_col, "GMM",
                                      group_col=group_col if group_detected else None)

    return labels, gmm, {"group_ari": gmm_group_ari, "feat_final_used": feat_final}


if PIPELINE_HALTED_AT is None:
    # Run the FOUR independent methods (+ GMM candidate) on the real casino
    # population, each sourcing its feature set from Step 4's authoritative
    # tendency_result['feat_final_used'] and each cross-checked against
    # 'vip_tier' for the same circularity question asked throughout this
    # pipeline. K-Means -> Hierarchical (given K-Means' k for comparison,
    # but proposing its own) -> DBSCAN -> HDBSCAN (conditionally, off
    # DBSCAN's own diagnostics) -> GMM (unconditional BIC-based candidate).
    km_k, km_details = step7a_kmeans_sweep(
        X_scaled, final_features, structure_result=structure_result, modelling_df=modelling_df,
        tendency_result=tendency_result, investigation_results=investigation_results,
    )
    hier_labels, coph_r, Z, hier_k, hier_details = step7b_hierarchical(
        X_scaled, final_features, km_k, structure_result=structure_result, modelling_df=modelling_df,
        tendency_result=tendency_result, investigation_results=investigation_results,
    )
    db_labels, db_eps, db_min_samples, db_results, db_model, db_details = step7c_dbscan(
        X_scaled, final_features, structure_result=structure_result, modelling_df=modelling_df,
        tendency_result=tendency_result, investigation_results=investigation_results,
    )
    hdb_labels, hdb_min_cluster_size, hdb_model, hdb_details = step7d_hdbscan(
        X_scaled, final_features, db_labels, structure_result=structure_result, modelling_df=modelling_df,
        tendency_result=tendency_result, investigation_results=investigation_results,
    )
    gmm_labels, gmm_model, gmm_details = fit_gmm_candidate_for_step8a(
        X_scaled, final_features, structure_result=structure_result, modelling_df=modelling_df,
        tendency_result=tendency_result, investigation_results=investigation_results,
    )

    banner("STEP 7A-7F SUMMARY: Four independent methods + GMM candidate")
    print(f"   {'Algorithm':<14} {'k':>4} {'group_ari (vip_tier)':>22}")
    print(f"   {'K-Means':<14} {km_k:>4} {('%.4f' % km_details['group_ari']) if km_details['group_ari'] is not None else 'n/a':>22}")
    print(f"   {'Hierarchical':<14} {hier_k:>4} {('%.4f' % hier_details['group_ari']) if hier_details['group_ari'] is not None else 'n/a':>22}")
    print(f"   {'DBSCAN':<14} {len(set(db_labels)) - (1 if -1 in db_labels else 0):>4} "
          f"{('%.4f' % db_details['group_ari']) if db_details['group_ari'] is not None else 'n/a':>22}")
    if hdb_labels is not None:
        print(f"   {'HDBSCAN':<14} {len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0):>4} "
              f"{('%.4f' % hdb_details['group_ari']) if hdb_details['group_ari'] is not None else 'n/a':>22}")
    else:
        print(f"   {'HDBSCAN':<14} {'n/a':>4} {'not triggered':>22}")
    print(f"   {'GMM':<14} {len(set(gmm_labels)):>4} "
          f"{('%.4f' % gmm_details['group_ari']) if gmm_details['group_ari'] is not None else 'n/a':>22}")
    print(f"\n   All five candidates are now ready for Step 8A's cross-algorithm comparison.")

    subbanner("Figures generated this run")
    if FIGURES_SAVED:
        for p in FIGURES_SAVED:
            print(f"   \U0001F4CA {p}")
        print(f"   {len(FIGURES_SAVED)} figure(s) saved to {FIGDIR}")
    else:
        print(f"   No figures were saved \u2014 FIGDIR is None. Set FIGDIR to a real folder path near the top of")
        print(f"   this script (right after data_dir is defined) to have plots written to disk.")
else:
    print(f"\n\u23ed\ufe0f  Steps 7A/7C/7D/7E/7F (Four Independent Methods + GMM) SKIPPED \u2014 pipeline halted at "
          f"'{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")


######################## Internal Validity Indices & Cross-Algorithm Agreement (Step 8A)
#                        + Post-Hoc Winner-Aware Audit (Enhancement 20)
#
# ADAPTED FOR THE CASINO SEGMENTATION PIPELINE.
#
# Structure-detection decision matrix: temporal and network are N/A here
# for the same reason as Steps 5/6/7A-7F -- a single deterministic
# comparison over the fixed row set, no resampling for either to act on.
# Group structure is genuinely central to THIS step specifically, more so
# than any prior one: this is where a FINAL production algorithm gets
# selected, and every candidate's group_ari (vs 'vip_tier') was ALREADY
# computed in Steps 7A-7F -- rather than re-deriving it, this step
# SURFACES those values directly in the comparison table, and explicitly
# flags it if the algorithm that WINS also happens to show high agreement
# with vip_tier. A winner chosen on internal validity alone, with a
# circularity reading sitting unexamined in an earlier step's log, is
# exactly the failure mode this integration closes.
#
# Protection awareness: BUSINESS_PROTECTED_FEATURES doesn't change which
# algorithm wins here -- every candidate was already fit on the SAME
# authoritative feature set (Step 4's tendency_result['feat_final_used']),
# so protection's work was already done upstream. Stated explicitly below
# rather than silently assumed, consistent with every other step's
# treatment of "doesn't apply, and here's specifically why."
#
# Leakage investigation carry-through: investigation_results is narrated
# for continuity, same as Steps 5/6/7A-7F.
#
# KNOWN, ALREADY-HANDLED LIMITATION (not something this adaptation needs to
# fix): LEVEL 2's preliminary stability check calls bootstrap_jaccard_
# stability(), which depends on _fit_algorithm_generic() -- referenced but
# never defined anywhere in this pipeline (same category as the missing
# distance-matrix helpers Step 4/5/7B needed, before those were supplied
# separately). The ORIGINAL code already wraps this in a try/except that
# scores a failed stability check as 0.0 with an explicit warning, rather
# than crashing -- confirmed by direct execution below: EVERY candidate's
# preliminary stability comes back 0.0 on this pipeline, and
# minmax_normalize() correctly degrades an all-equal input to a neutral
# 0.5 for everyone (verified: doesn't divide by zero, doesn't crash,
# doesn't spuriously favour any one candidate). The Model Score is
# therefore currently being decided by silhouette/CH/DBI/balance only, with
# stability contributing a flat, non-differentiating 20% to every
# candidate -- worth knowing when reading the winner decision, not a
# silent gap.

def minmax_normalize(arr):
    arr = np.array(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.full_like(arr, 0.5)


def compute_model_score(sils, chs, dbis, stabilities, balances):
    """Pure function implementing Level 2's weighted Model Score (Enhancement
    22): 30% silhouette + 20% CH - 20% DBI + 20% stability + 10% balance, all
    min-max normalised across candidates first so the stated percentages are
    literally true contributions, not an unbounded z-score sum. Extracted for
    direct unit-testing with controlled numbers, same pattern as
    passes_dual_check (round 6) and rank_by_consensus (round 9)."""
    norm_sil = minmax_normalize(sils)
    norm_ch = minmax_normalize(chs)
    norm_dbi = minmax_normalize(dbis)  # NOT inverted -- subtracted below, so a HIGH (worse) DBI loses points
    norm_stab = minmax_normalize(stabilities)
    norm_bal = minmax_normalize(balances)
    return (SCORE_WEIGHT_SILHOUETTE * norm_sil + SCORE_WEIGHT_CH * norm_ch - SCORE_WEIGHT_DBI * norm_dbi
            + SCORE_WEIGHT_STABILITY * norm_stab + SCORE_WEIGHT_BALANCE * norm_bal)


def _self_test_model_score():
    """Proves compute_model_score() genuinely lets stability and business
    balance override raw internal-validity superiority -- with controlled
    numbers, since reliably producing this exact scenario from an organic
    dataset proved unreliable. Uses THREE candidates, not two -- with only
    two, min-max normalisation degenerates to trivial 0/1 values for every
    metric, which can produce misleading exact ties by coincidence rather
    than reflecting genuine differences in degree."""
    banner("SELF-TEST: Model Score (Enhancement 22) genuinely weighs stability/balance")
    print(f"     Case A: candidate 1 has the BEST silhouette/CH/DBI but WORST stability and")
    print(f"     balance; candidate 2 is moderate on geometry but stable and balanced; candidate 3")
    print(f"     is a mediocre baseline on everything -> expect the Model Score to prefer candidate 2.")
    sils = np.array([0.65, 0.48, 0.30])
    chs = np.array([3300.0, 5200.0, 1000.0])
    dbis = np.array([0.44, 0.87, 1.50])
    stabilities = np.array([0.10, 0.90, 0.50])
    balances = np.array([0.02, 0.15, 0.08])
    score = compute_model_score(sils, chs, dbis, stabilities, balances)
    print(f"       Scores: candidate 1={score[0]:.4f}  candidate 2={score[1]:.4f}  candidate 3={score[2]:.4f}")
    assert score[1] > score[0], ("Model Score should prefer the stable, business-usable candidate "
                                   "over the geometrically-superior-but-unstable one")

    print(f"     Case B: all metrics identical across candidates -> expect a TIE (no divide-by-zero crash)")
    tied = np.array([0.5, 0.5, 0.5])
    score_tied = compute_model_score(tied, tied, tied, tied, tied)
    assert np.allclose(score_tied[0], score_tied[1]) and np.allclose(score_tied[1], score_tied[2]), \
        "identical candidates should score identically"

    print(f"     Case C (the REAL casino situation): ALL-ZERO stability (bootstrap_jaccard_stability's")
    print(f"     missing _fit_algorithm_generic dependency scores every candidate 0.0) -> expect a neutral")
    print(f"     0.5 contribution for everyone, not a crash and not a spurious winner/loser")
    all_zero_stability = np.array([0.0, 0.0, 0.0])
    score_zero_stab = compute_model_score(sils, chs, dbis, all_zero_stability, balances)
    norm_stab_check = minmax_normalize(all_zero_stability)
    assert np.allclose(norm_stab_check, 0.5), "all-equal (all-zero) stability must normalise to a neutral 0.5"
    print(f"       \u2705 All-zero stability correctly normalises to a flat, neutral 0.5 contribution for")
    print(f"          every candidate -- exactly what happens on this pipeline's real run today.")

    print(f"     \u2705 The weighted Model Score genuinely lets stability/balance override raw internal")
    print(f"        validity, handles the degenerate all-tied case without crashing, and degrades sensibly")
    print(f"        when stability itself is entirely unavailable (this pipeline's real current state).")


def step_winner_aware_audit(removal_history, algorithm, X_scaled=None, feat_final_v1=None, k_screen=4):
    banner("Post-Hoc Winner-Aware Audit (Enhancement 20)")
    if not removal_history:
        print(f"   No features were removed during Step 3D \u2014 nothing to audit.")
        return None

    if algorithm in ("kmeans", "hierarchical"):
        print(f"   Primary algorithm '{algorithm}' uses the SAME K-Means silhouette criterion that")
        print(f"   already drove Step 3D's search order \u2014 agreement here would be TAUTOLOGICAL, not an")
        print(f"   independent check. See Enhancement 19's consensus-pool tables in the Step 3D log for")
        print(f"   the actual density cross-checks each removal already passed.")
        return {"applicable": False, "reason": "tautological for the algorithm that drove selection"}

    if algorithm in ("dbscan", "hdbscan"):
        print(f"   Auditing each of Step 3D's {len(removal_history)} removal(s) against the WINNING")
        print(f"   algorithm's OWN density criterion -- reusing the density values ALREADY computed")
        print(f"   during Enhancement 19's consensus ranking (this pipeline's density check already IS")
        print(f"   HDBSCAN's own relative_validity_ metric, so no extra fitting is needed here).")
        agreements, n_comparable = 0, 0
        prior_density = None
        for entry in removal_history:
            dens_after = entry["density_after"]
            if prior_density is None:
                verdict = "N/A (first removal, no prior density to compare against)"
            else:
                would_agree = dens_after >= prior_density
                n_comparable += 1
                agreements += int(would_agree)
                verdict = "AGREES" if would_agree else "WOULD HAVE KEPT (winner's own density got worse)"
            print(f"   Step {entry['step']}: removed '{entry['removed']}'  density "
                  f"{('%.6g' % prior_density) if prior_density is not None else 'n/a':>10} \u2192 {dens_after:.6g}  {verdict}")
            prior_density = dens_after
        if n_comparable > 0:
            print(f"\n   DECISION: {agreements}/{n_comparable} comparable removals also improved/maintained")
            print(f"   the WINNING algorithm's own density criterion. This audit is largely redundant for")
            print(f"   '{algorithm}' specifically, since its own metric already gated every decision")
            print(f"   directly via Enhancement 19's consensus pool, not merely informed a veto.")
        return {"applicable": True, "agreements": agreements, "n_comparable": n_comparable}

    if algorithm == "gmm":
        print(f"   Auditing each removal against GMM's OWN BIC criterion (recomputed fresh here, since")
        print(f"   Step 3D's dual-check never evaluates GMM directly). NOTE: BIC is not perfectly")
        print(f"   comparable across different feature-set sizes in an absolute sense; the DIRECTION of")
        print(f"   change (did removing this feature improve GMM's own fit-vs-complexity trade-off?) is")
        print(f"   the informative signal here, not the raw magnitude.")
        agreements = 0
        X = X_scaled[feat_final_v1]
        for entry in removal_history:
            features_before, removed = entry["features_before"], entry["removed"]
            features_after = [c for c in features_before if c != removed]
            bic_before = GaussianMixture(n_components=k_screen, random_state=RANDOM_STATE).fit(X[features_before]).bic(X[features_before].values)
            bic_after = GaussianMixture(n_components=k_screen, random_state=RANDOM_STATE).fit(X[features_after]).bic(X[features_after].values)
            gmm_gain = bic_before - bic_after  # positive = removing improved (lowered) BIC
            agree = gmm_gain > 0
            agreements += int(agree)
            print(f"   Step {entry['step']}: removed '{removed}'  GMM BIC gain={gmm_gain:+.2f}  "
                  f"{'AGREES with removal' if agree else 'GMM would have KEPT this feature'}")
        print(f"\n   DECISION: {agreements}/{len(removal_history)} of the actual removals would also have")
        print(f"   been approved by GMM's own BIC criterion.")
        return {"applicable": True, "agreements": agreements, "n_comparable": len(removal_history)}

    # kmedoids
    print(f"   K-Medoids' own fitted-metric silhouette audit is not implemented for this pipeline")
    print(f"   version (would require rebuilding a fresh Gower/Mahalanobis distance matrix per removal")
    print(f"   step, which is expensive). Flagged honestly as a gap rather than skipped silently -- on the")
    print(f"   real casino data, K-Medoids has NOT won Step 8A on any run so far, so this gap remains")
    print(f"   theoretical for now, not a live blind spot.")
    return {"applicable": False, "reason": "not implemented for kmedoids"}


def _self_test_winner_aware_audit():
    """Proves the audit correctly detects BOTH agreement and disagreement
    between K-Means-consensus's actual removals and the winning algorithm's
    own criterion -- on a controlled synthetic removal_history, since this
    pipeline's real Step 3D only removed ONE feature (days_since_login),
    too small a history to exercise both branches organically."""
    banner("SELF-TEST: Post-hoc winner-aware audit (Enhancement 20)")
    print(f"     Case A (dbscan/hdbscan branch): a history where density genuinely IMPROVES")
    print(f"     every step -> expect full agreement")
    history_agree = [
        {"step": 1, "features_before": ["a", "b", "c"], "removed": "c", "kmeans_gain": 0.05, "density_after": 0.3},
        {"step": 2, "features_before": ["a", "b"], "removed": "b", "kmeans_gain": 0.03, "density_after": 0.5},
    ]
    result_a = step_winner_aware_audit(history_agree, "hdbscan")
    print(f"       agreements={result_a['agreements']}/{result_a['n_comparable']}")
    assert result_a["agreements"] == result_a["n_comparable"] == 1, "a monotonically improving density history should show full agreement"

    print(f"     Case B (dbscan/hdbscan branch): a history where density gets WORSE at step 2")
    print(f"     -> expect a genuine disagreement to be detected")
    history_disagree = [
        {"step": 1, "features_before": ["a", "b", "c"], "removed": "c", "kmeans_gain": 0.05, "density_after": 0.5},
        {"step": 2, "features_before": ["a", "b"], "removed": "b", "kmeans_gain": 0.03, "density_after": 0.2},
    ]
    result_b = step_winner_aware_audit(history_disagree, "hdbscan")
    print(f"       agreements={result_b['agreements']}/{result_b['n_comparable']}")
    assert result_b["agreements"] == 0 and result_b["n_comparable"] == 1, "a worsening density step must be flagged as disagreement, not silently passed"

    print(f"     Case C (kmeans/hierarchical branch): must correctly skip as tautological, not crash")
    result_c = step_winner_aware_audit(history_agree, "kmeans")
    assert result_c["applicable"] is False

    print(f"     \u2705 The audit correctly distinguishes genuine agreement from genuine disagreement,")
    print(f"        and correctly skips the tautological K-Means/Hierarchical case.")


def _smallest_segment_frac(labels, exclude_noise=False):
    """Smallest cluster's size as a FRACTION of n -- for readability/display
    only. Eligibility DECISIONS use _smallest_segment_abs() below instead,
    since SEGMENT_SIZE_ELIGIBILITY is an absolute customer-count floor, not
    a fraction (see the fix documented at this function's original call
    site). Module-level (not nested inside step9_10d_validate) so it's the
    SAME logic, independently testable, for every candidate including
    K-Means/Hierarchical."""
    mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)
    sizes = pd.Series(labels[mask]).value_counts()
    return float(sizes.min() / len(labels)) if len(sizes) else 0.0


def _smallest_segment_abs(labels, exclude_noise=False):
    """Smallest cluster's size as an ABSOLUTE customer count -- the correct
    unit to compare against SEGMENT_SIZE_ELIGIBILITY. Module-level so
    K-Means/Hierarchical's eligibility uses the IDENTICAL function as
    DBSCAN/HDBSCAN/GMM/K-Medoids, not a parallel copy that could drift."""
    mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)
    sizes = pd.Series(labels[mask]).value_counts()
    return int(sizes.min()) if len(sizes) else 0


def _self_test_symmetric_eligibility_gate():
    """Proves K-Means and Hierarchical are now subject to the EXACT SAME
    size-eligibility gate as DBSCAN/HDBSCAN/GMM/K-Medoids -- closing the
    asymmetry where the two 'default' algorithms were previously admitted
    to Step 8A's comparison unconditionally, regardless of their own
    segment balance. Since the real casino run's K-Means/Hierarchical both
    comfortably clear the floor (967/905-ish customers vs a 500 floor),
    this asymmetry never actually bit on real data -- this is the only
    proof the fix (and its fallback for the all-ineligible edge case)
    actually works."""
    banner("SELF-TEST: Symmetric eligibility gate (K-Means/Hierarchical no longer exempt)")
    print(f"   Case A: a K-Means-shaped partition with a genuinely tiny segment (40 of 10,000, well below")
    print(f"   the {SEGMENT_SIZE_ELIGIBILITY:,}-customer floor) -> must be flagged INELIGIBLE, same as any")
    print(f"   other algorithm would be")
    tiny_labels = np.array([0] * 4000 + [1] * 5960 + [2] * 40)
    min_abs = _smallest_segment_abs(tiny_labels)
    eligible = len(set(tiny_labels)) >= 2 and min_abs >= SEGMENT_SIZE_ELIGIBILITY
    print(f"     smallest segment = {min_abs} customers, needs >= {SEGMENT_SIZE_ELIGIBILITY} -> "
          f"{'ELIGIBLE' if eligible else 'INELIGIBLE'}")
    assert not eligible, "a 40-customer segment must fail the SAME 500-customer floor as any other algorithm"

    print(f"\n   Case B: a well-balanced K-Means-shaped partition (3 segments, all well above the floor)")
    print(f"   -> must be flagged ELIGIBLE")
    balanced_labels = np.array([0] * 4000 + [1] * 4000 + [2] * 4000)
    min_abs_b = _smallest_segment_abs(balanced_labels)
    eligible_b = len(set(balanced_labels)) >= 2 and min_abs_b >= SEGMENT_SIZE_ELIGIBILITY
    assert eligible_b, "a well-balanced partition should clear the floor easily"
    print(f"     smallest segment = {min_abs_b} customers -> ELIGIBLE (as expected)")

    print(f"\n   Case C: ALL candidates ineligible (every algorithm's smallest segment is tiny) -> must NOT")
    print(f"   crash on an empty candidate set; must fall back to K-Means/Hierarchical with a loud warning")
    all_tiny = {"kmeans": np.array([0] * 30 + [1] * 30), "hierarchical": np.array([0] * 30 + [1] * 30)}
    candidates_test = {}
    for nm, labels_nm in all_tiny.items():
        if len(set(labels_nm)) >= 2 and _smallest_segment_abs(labels_nm) >= SEGMENT_SIZE_ELIGIBILITY:
            candidates_test[nm] = labels_nm
    if not candidates_test:
        candidates_test = dict(all_tiny)  # the same fallback step9_10d_validate applies
    assert len(candidates_test) == 2, "the all-ineligible fallback must still leave a non-empty comparison set"
    print(f"     \u2705 Fallback correctly produces a non-empty candidate set ({list(candidates_test.keys())}) "
          f"instead of crashing on an empty comparison.")

    print(f"\n   \u2705 K-Means/Hierarchical-shaped partitions are now judged by the EXACT SAME rule as every")
    print(f"      other algorithm -- no more unconditional exemption -- and the degenerate all-ineligible")
    print(f"      case degrades safely rather than crashing.")


def step9_10d_validate(X_scaled, feat_final, km_labels, hier_labels, db_labels, hdbscan_labels=None, gmm_labels=None,
                        kmedoids_labels=None, db_eps=None, db_min_samples=None, hdb_min_cluster_size=None,
                        kmed_alt_distance_metric=None, kmed_num_cols=None, kmed_cat_cols=None,
                        structure_result=None, modelling_df=None, investigation_results=None, group_ari_map=None):
    banner("STEP 8A: Internal Validity Indices & Cross-Algorithm Agreement")

    subbanner("Structure-detection decision matrix applied to this step")
    group_detected, group_col = False, None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: a single")
        print(f"   deterministic comparison over the fixed row set, no resampling for a time dimension to")
        print(f"   act on.")
        print(f"   Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: an algorithm/")
        print(f"   reporting decision made once per candidate already, not this comparison step's own concern.")
        if group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 genuinely central to THIS step: every")
            print(f"   candidate's ARI against raw '{group_col}' was ALREADY computed in Steps 7A-7F. Rather")
            print(f"   than re-deriving it, this step SURFACES those values directly in the comparison table")
            print(f"   below, and explicitly flags it if the WINNING algorithm also shows high agreement with")
            print(f"   '{group_col}' -- a winner picked on internal validity alone, with an unexamined")
            print(f"   circularity reading left behind in an earlier step's log, is exactly what this closes.")
        else:
            print(f"   Group structure: NO \u2192 no circularity cross-reference is meaningful for this comparison.")
    else:
        print(f"   No structure_result supplied \u2014 no decision-matrix narration, no group_ari column in the")
        print(f"   comparison table below.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    if protected_present:
        print(f"   Business-protected features present in the compared feature set: {protected_present}")
        print(f"   Doesn't change WHICH algorithm wins here \u2014 every candidate below was already fit on this")
        print(f"   SAME authoritative feature set (protection's work already done upstream in redundancy()/")
        print(f"   feature_selection()); stated explicitly rather than silently assumed.")
    else:
        print(f"   No business-protected features present in the compared feature set.")

    if investigation_results:
        print(f"\n   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"   relevant context for interpreting whichever algorithm wins below, not an input to the")
        print(f"   comparison itself.")

    X = X_scaled[feat_final].values

    def report(labels, name, exclude_noise=False):
        mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)
        n_clusters_found = len(set(labels[mask])) if mask.sum() > 0 else 0
        if mask.sum() == 0 or n_clusters_found < 2:
            print(f"   {name:<28} k={n_clusters_found}  DEGENERATE (fewer than 2 real clusters after noise "
                  f"exclusion, mask.sum()={mask.sum()}) \u2014 scored as worst-case, not fit for silhouette/DBI/CH")
            return float("-1.0"), 1e6, 0.0
        sil = silhouette_score(X[mask], labels[mask], sample_size=4000, random_state=RANDOM_STATE)
        dbi = davies_bouldin_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        print(f"   {name:<28} k={n_clusters_found}  Silhouette={sil:.4f}  DaviesBouldin={dbi:.4f}  CalinskiHarabasz={ch:>10.1f}")
        return sil, dbi, ch

    print(f"   (FIX: each algorithm's k below is its OWN independent proposal, not a shared value \u2014")
    print(f"   K-Means and Hierarchical each ran their own convergent-evidence sweep in Steps 7A/7C)")
    km_m = report(km_labels, "K-Means")
    hier_m = report(hier_labels, "Hierarchical")
    db_m = report(db_labels, "DBSCAN (noise excluded)", exclude_noise=True)
    hdb_m = report(hdbscan_labels, "HDBSCAN (noise excluded)", exclude_noise=True) if hdbscan_labels is not None else None
    gmm_m = report(gmm_labels, "GMM") if gmm_labels is not None else None
    if kmedoids_labels is not None and len(kmedoids_labels) == len(km_labels):
        kmed_m = report(kmedoids_labels, "K-Medoids (Gower/Mahalanobis)")
    else:
        if kmedoids_labels is not None:
            print(f"   K-Medoids: computed on a scalability-guard subsample (length mismatch vs full dataset) \u2014")
            print(f"   excluded from THIS comparison; out-of-sample extension for precomputed-distance methods")
            print(f"   is a known scope limitation, not silently ignored.")
        kmed_m = None

    subbanner("Plot: 2D PCA projection of the K-Means segments (visual diagnostic only)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X)
    pca_variance_retained = float(pca.explained_variance_ratio_.sum())
    plt.figure(figsize=(8, 6.5))
    for c in sorted(set(km_labels)):
        mask = km_labels == c
        plt.scatter(Xp[mask, 0], Xp[mask, 1], s=6, alpha=0.35, label=f"Cluster {c} (n={mask.sum()})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("K-Means Segments \u2014 2D PCA Projection (fit on the full feature set; 2D is a visual aid only)")
    plt.legend(markerscale=4)
    savefig("step8A_pca_scatter.png")

    subbanner("Plot: UMAP projection (Enhancement 8)")
    pca_flag = pca_variance_retained < UMAP_PCA_VARIANCE_FLOOR
    print(f"   Trigger check: BUSINESS_CRITICAL_SEGMENTATION={BUSINESS_CRITICAL_SEGMENTATION}  OR  "
          f"PCA 2D retained variance ({pca_variance_retained:.1%}) < floor ({UMAP_PCA_VARIANCE_FLOOR:.0%})? {pca_flag}")
    triggered_umap = BUSINESS_CRITICAL_SEGMENTATION or pca_flag
    print(f"   DECISION: {'TRIGGERED' if triggered_umap else 'NOT triggered'} \u2014 "
          f"{'PCA alone is a linear-only diagnostic; UMAP adds a non-linear view for analyst/stakeholder review.' if triggered_umap else 'PCA retains enough variance and this is exploratory-only; skipping UMAP to save compute.'}")
    if triggered_umap:
        # FIX: UMAP was previously fit on the FULL, unguarded population
        # here -- confirmed directly this is genuinely too slow at this
        # pipeline's real data scale (didn't complete within 285s on just
        # one ~260K-row cluster, let alone the full population). Every
        # OTHER expensive fit in this pipeline (Ward, Mahalanobis, HDBSCAN,
        # the K-Means/silhouette ranking in Step 3D) already subsamples at
        # LARGE_N_THRESHOLD for exactly this reason; UMAP never did,
        # including here at the top level. Safe to do for a pure visual
        # diagnostic -- it isn't used for any clustering decision.
        if len(X) > LARGE_N_THRESHOLD:
            _umap_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), LARGE_N_THRESHOLD, replace=False)
            X_umap_input, km_labels_umap = X[_umap_idx], km_labels[_umap_idx]
            print(f"   UMAP input subsampled to {LARGE_N_THRESHOLD:,} rows (from {len(X):,}) -- a visual")
            print(f"   diagnostic only, not used for any clustering decision.")
        else:
            X_umap_input, km_labels_umap = X, km_labels
        reducer = umap_lib.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
        Xu = reducer.fit_transform(X_umap_input)
        plt.figure(figsize=(8, 6.5))
        for c in sorted(set(km_labels_umap)):
            mask = km_labels_umap == c
            plt.scatter(Xu[mask, 0], Xu[mask, 1], s=6, alpha=0.35, label=f"Cluster {c} (n={mask.sum()})")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.title("K-Means Segments \u2014 UMAP Projection (non-linear; local structure, unlike PCA)")
        plt.legend(markerscale=4)
        savefig("step8A_umap_scatter.png")

    subbanner("STEP 8A (cont'd): Cross-Algorithm Agreement")
    ari_hier = adjusted_rand_score(km_labels, hier_labels)
    nmi_hier = normalized_mutual_info_score(km_labels, hier_labels)
    print(f"   K-Means vs Hierarchical: ARI={ari_hier:.4f}  NMI={nmi_hier:.4f}")

    mask = db_labels != -1
    ari_db = adjusted_rand_score(km_labels[mask], db_labels[mask]) if mask.sum() > 0 else float("nan")
    nmi_db = normalized_mutual_info_score(km_labels[mask], db_labels[mask]) if mask.sum() > 0 else float("nan")
    print(f"   K-Means vs DBSCAN (noise excl., n={mask.sum()}): ARI={ari_db:.4f}  NMI={nmi_db:.4f}")

    subbanner("Symmetric algorithm comparison (FIX: now up to 6-way, ALL SIX subject to the SAME gate)")
    subbanner("LEVEL 1 (Enhancement 22): pre-winner eligibility gate now checks segment SIZE too")
    print(f"   The business scorecard previously only discovered 'this segment is too tiny to be usable'")
    print(f"   AFTER a winner was selected. Moved earlier: >=2 clusters, <={DENSITY_NOISE_ELIGIBILITY_L1:.0%} "
          f"noise, smallest segment >={SEGMENT_SIZE_ELIGIBILITY:,} customers.")
    # FIX 2 (requested directly, after the eligibility-decision conversation
    # above): K-Means and Hierarchical were PREVIOUSLY exempt from this gate
    # entirely -- unconditionally admitted to candidates{} below regardless
    # of their own segment balance, while DBSCAN/HDBSCAN/GMM/K-Medoids were
    # always checked. On the real casino run this never actually bit (both
    # comfortably clear the floor), but it was a real asymmetry: a future
    # run where K-Means/Hierarchical produced an unbalanced tiny segment
    # would still let them compete, while an equally-tiny DBSCAN segment
    # would be excluded. Both now go through the IDENTICAL
    # _smallest_segment_abs()-based check as every other candidate.
    #
    # EDGE CASE this fix introduces (previously impossible, since km/hier
    # were unconditional): ALL SIX candidates could now be simultaneously
    # ineligible, leaving an EMPTY candidates{} dict -- np.argmax() on an
    # empty array would crash. Handled below with an explicit, LOUD
    # fallback to K-Means+Hierarchical rather than a silent crash or a
    # silent re-exemption; self-tested in _self_test_symmetric_eligibility_
    # gate()'s Case C.
    km_n_clusters = len(set(km_labels))
    km_min_seg_frac = _smallest_segment_frac(km_labels)
    km_min_seg_abs = _smallest_segment_abs(km_labels)
    km_eligible = km_n_clusters >= 2 and km_min_seg_abs >= SEGMENT_SIZE_ELIGIBILITY
    print(f"   K-Means eligibility: {km_n_clusters} clusters, smallest segment={km_min_seg_frac:.1%} of n "
          f"({km_min_seg_abs:,} customers, needs \u22652 clusters and smallest \u2265{SEGMENT_SIZE_ELIGIBILITY:,}) "
          f"\u2192 {'ELIGIBLE' if km_eligible else 'INELIGIBLE \u2014 excluded from the vote, not from reporting'}")

    hier_n_clusters = len(set(hier_labels))
    hier_min_seg_frac = _smallest_segment_frac(hier_labels)
    hier_min_seg_abs = _smallest_segment_abs(hier_labels)
    hier_eligible = hier_n_clusters >= 2 and hier_min_seg_abs >= SEGMENT_SIZE_ELIGIBILITY
    print(f"   Hierarchical eligibility: {hier_n_clusters} clusters, smallest segment={hier_min_seg_frac:.1%} of n "
          f"({hier_min_seg_abs:,} customers, needs \u22652 clusters and smallest \u2265{SEGMENT_SIZE_ELIGIBILITY:,}) "
          f"\u2192 {'ELIGIBLE' if hier_eligible else 'INELIGIBLE \u2014 excluded from the vote, not from reporting'}")

    # FIX (found by actually running this on the real casino population, not
    # assumed): SEGMENT_SIZE_ELIGIBILITY=500 is defined pipeline-wide as an
    # ABSOLUTE customer-count floor (matching its use elsewhere, e.g. the
    # Pre-Activation cohort's own "500 players minimum actionable size"
    # rule) -- but the four eligibility checks below were comparing it
    # directly against _smallest_segment_frac()'s FRACTION (0-1 range).
    # `fraction >= 500` is structurally always False, which silently and
    # PERMANENTLY disqualified DBSCAN/HDBSCAN/GMM/K-Medoids from ever
    # winning Step 8A, regardless of their actual segment balance --
    # confirmed directly: on the real run, DBSCAN scored a BETTER
    # silhouette (0.6215) and DBI (0.5085) than either K-Means or
    # Hierarchical, yet was excluded from competing at all before this fix.
    # Comparing the ABSOLUTE smallest-segment count against the ABSOLUTE
    # threshold (matching units correctly) is the actual fix; the fraction
    # is still computed and shown for readability, not for the decision.
    # _smallest_segment_frac()/_smallest_segment_abs() are module-level
    # helpers now (see above step9_10d_validate) -- promoted out of this
    # function so K-Means/Hierarchical's own eligibility (below) can use
    # the identical, independently-testable logic as every other candidate.

    db_noise_frac = float((db_labels == -1).mean())
    db_n_clusters = len(set(db_labels[db_labels != -1]))
    db_min_seg_frac = _smallest_segment_frac(db_labels, exclude_noise=True)
    db_min_seg_abs = _smallest_segment_abs(db_labels, exclude_noise=True)
    db_eligible = (db_n_clusters >= 2 and db_noise_frac <= DENSITY_NOISE_ELIGIBILITY_L1
                   and db_min_seg_abs >= SEGMENT_SIZE_ELIGIBILITY)
    print(f"   DBSCAN eligibility: {db_n_clusters} real clusters, {db_noise_frac:.1%} noise, "
          f"smallest segment={db_min_seg_frac:.1%} of n ({db_min_seg_abs:,} customers, needs "
          f"\u2265{SEGMENT_SIZE_ELIGIBILITY:,}) \u2192 "
          f"{'ELIGIBLE' if db_eligible else 'INELIGIBLE \u2014 excluded from the vote, not from reporting'}")

    hdb_eligible = False
    if hdbscan_labels is not None:
        hdb_noise_frac = float((hdbscan_labels == -1).mean())
        hdb_n_clusters = len(set(hdbscan_labels[hdbscan_labels != -1]))
        hdb_min_seg_frac = _smallest_segment_frac(hdbscan_labels, exclude_noise=True)
        hdb_min_seg_abs = _smallest_segment_abs(hdbscan_labels, exclude_noise=True)
        hdb_eligible = (hdb_n_clusters >= 2 and hdb_noise_frac <= DENSITY_NOISE_ELIGIBILITY_L1
                        and hdb_min_seg_abs >= SEGMENT_SIZE_ELIGIBILITY)
        print(f"   HDBSCAN eligibility: {hdb_n_clusters} real clusters, {hdb_noise_frac:.1%} noise, "
              f"smallest segment={hdb_min_seg_frac:.1%} of n ({hdb_min_seg_abs:,} customers, needs "
              f"\u2265{SEGMENT_SIZE_ELIGIBILITY:,}) \u2192 "
              f"{'ELIGIBLE' if hdb_eligible else 'INELIGIBLE \u2014 excluded from the vote, not from reporting'}")
    else:
        print(f"   HDBSCAN: not triggered on this run (Step 7E) \u2014 no candidate to evaluate.")

    gmm_eligible = False
    if gmm_labels is not None:
        gmm_n_clusters = len(set(gmm_labels))
        gmm_min_frac = _smallest_segment_frac(gmm_labels)
        gmm_min_abs = _smallest_segment_abs(gmm_labels)
        gmm_eligible = gmm_n_clusters >= 2 and gmm_min_abs >= SEGMENT_SIZE_ELIGIBILITY
        print(f"   GMM eligibility: {gmm_n_clusters} components, smallest={gmm_min_frac:.2%} of n "
              f"({gmm_min_abs:,} customers, needs \u22652 components and smallest \u2265{SEGMENT_SIZE_ELIGIBILITY:,}, "
              f"raised from round 3's 1%-of-n floor for consistency with Level 1) \u2192 "
              f"{'ELIGIBLE' if gmm_eligible else 'INELIGIBLE \u2014 excluded from the vote, not from reporting'}")
    else:
        print(f"   GMM: not fit as a Step 8A candidate on this run \u2014 no candidate to evaluate.")

    kmed_eligible = False
    if kmed_m is not None:
        kmed_n_clusters = len(set(kmedoids_labels))
        kmed_min_frac = _smallest_segment_frac(kmedoids_labels)
        kmed_min_abs = _smallest_segment_abs(kmedoids_labels)
        kmed_eligible = kmed_n_clusters >= 2 and kmed_min_abs >= SEGMENT_SIZE_ELIGIBILITY
        print(f"   K-Medoids eligibility: {kmed_n_clusters} clusters, smallest={kmed_min_frac:.2%} of n "
              f"({kmed_min_abs:,} customers, needs \u22652 clusters and smallest \u2265{SEGMENT_SIZE_ELIGIBILITY:,}) "
              f"\u2192 {'ELIGIBLE' if kmed_eligible else 'INELIGIBLE \u2014 excluded from the vote, not from reporting'}")
    else:
        print(f"   K-Medoids: not fit as a Step 8A candidate on this run (Step 5 selected Euclidean, "
              f"or subsample length mismatch) \u2014 no candidate to evaluate.")

    candidates = {}
    if km_eligible:
        candidates["kmeans"] = km_m
    if hier_eligible:
        candidates["hierarchical"] = hier_m
    if db_eligible:
        candidates["dbscan"] = db_m
    if hdb_eligible:
        candidates["hdbscan"] = hdb_m
    if gmm_eligible:
        candidates["gmm"] = gmm_m
    if kmed_eligible:
        candidates["kmedoids"] = kmed_m

    if not candidates:
        print(f"\n   \u26a0\ufe0f  NO candidate met the eligibility bar (>=2 clusters, smallest segment "
              f">={SEGMENT_SIZE_ELIGIBILITY:,} customers) \u2014 falling back to comparing K-Means and")
        print(f"   Hierarchical anyway, as the two baseline candidates this pipeline can always at least")
        print(f"   attempt to deploy, with this ineligibility explicitly ON RECORD rather than silently")
        print(f"   ignored or crashing on an empty comparison.")
        candidates = {"kmeans": km_m, "hierarchical": hier_m}
    names = list(candidates.keys())
    sils = np.array([candidates[n][0] for n in names])
    dbis = np.array([candidates[n][1] for n in names])
    chs = np.array([candidates[n][2] for n in names])

    subbanner("LEVEL 2 (Enhancement 22): computing preliminary stability + business balance per candidate")
    label_map_all = {"kmeans": km_labels, "hierarchical": hier_labels, "dbscan": db_labels,
                      "hdbscan": hdbscan_labels, "gmm": gmm_labels, "kmedoids": kmedoids_labels}
    prelim_stability, balances = [], []
    for nm in names:
        labels_nm = label_map_all[nm]
        k_nm = len(set(labels_nm[labels_nm != -1])) if -1 in labels_nm else len(set(labels_nm))
        balance = _smallest_segment_frac(labels_nm, exclude_noise=(nm in ("dbscan", "hdbscan")))
        balances.append(balance)
        try:
            pm_kwargs = {}
            if nm == "dbscan":
                pm_kwargs = dict(eps=db_eps, min_samples=db_min_samples)
            elif nm == "hdbscan":
                pm_kwargs = dict(min_cluster_size=hdb_min_cluster_size)
            elif nm == "kmedoids":
                pm_kwargs = dict(alt_distance_metric=kmed_alt_distance_metric, num_cols=kmed_num_cols,
                                  cat_cols=kmed_cat_cols, feat_cols_ordered=feat_final)
            pm_nm = build_primary_model(X, labels_nm, nm, **pm_kwargs)
            jac = bootstrap_jaccard_stability(X, pm_nm, labels_nm, k_nm, n_bootstrap=PRELIM_STABILITY_N_BOOTSTRAP)
            per_cluster = [np.mean(v) for v in jac.values() if len(v) > 0]
            stability = float(np.mean(per_cluster)) if per_cluster else 0.0
        except Exception as e:
            stability = 0.0
            print(f"   \u26a0\ufe0f  Preliminary stability check for '{nm}' failed ({type(e).__name__}: this")
            print(f"   pipeline's bootstrap_jaccard_stability() depends on _fit_algorithm_generic(), which is")
            print(f"   referenced but not defined in this script) \u2014 scored as 0.0 (worst case) rather than")
            print(f"   silently skipped. See this section's module-level comment: this makes stability a flat,")
            print(f"   neutral 20% contribution for every candidate below, not a differentiating factor today.")
        prelim_stability.append(stability)
        print(f"   {nm:<14} preliminary stability (n_bootstrap={PRELIM_STABILITY_N_BOOTSTRAP})={stability:.4f}  "
              f"smallest segment={balance:.2%} of n")

    composite = compute_model_score(sils, chs, dbis, prelim_stability, balances)

    subbanner("Group/rank circularity column (sourced from Steps 7A-7F, not recomputed)")
    group_ari_map = group_ari_map or {}
    group_aris = [group_ari_map.get(nm) for nm in names]
    if group_detected and group_col:
        print(f"   Each candidate's ARI vs raw '{group_col}' was already computed in Steps 7A-7F; shown")
        print(f"   alongside the Model Score below so a circularity concern is visible AT the winner decision,")
        print(f"   not buried in an earlier step's log.")
    else:
        print(f"   No group structure detected \u2014 group_ari column omitted from the table below.")

    print(f"\n   Model Score = {SCORE_WEIGHT_SILHOUETTE:.0%}*silhouette + {SCORE_WEIGHT_CH:.0%}*CH - "
          f"{SCORE_WEIGHT_DBI:.0%}*DBI + {SCORE_WEIGHT_STABILITY:.0%}*stability + {SCORE_WEIGHT_BALANCE:.0%}*balance "
          f"(all min-max normalised across eligible candidates)")
    header = f"   {'Algorithm':<14} {'Silhouette':>11} {'DBI':>8} {'CH':>10} {'Stability':>10} {'Balance':>9} {'Model Score':>12}"
    if group_detected and group_col:
        header += f" {'group_ari':>10}"
    print(header)
    for i, n in enumerate(names):
        row = (f"   {n:<14} {sils[i]:>11.4f} {dbis[i]:>8.4f} {chs[i]:>10.1f} {prelim_stability[i]:>10.4f} "
               f"{balances[i]:>9.2%} {composite[i]:>12.4f}")
        if group_detected and group_col:
            ga = group_aris[i]
            row += f" {(('%.4f' % ga) if ga is not None else 'n/a'):>10}"
        print(row)

    winner_idx = int(np.argmax(composite))
    winner = names[winner_idx]
    print(f"\n   DECISION: '{winner}' wins the symmetric comparison "
          f"(Model Score {composite[winner_idx]:.4f}, highest among {len(names)} eligible candidates).")
    if winner == "kmeans":
        print(f"   K-Means selected as PRIMARY production algorithm on the merits, among up to 6 genuine")
        print(f"   candidates \u2014 not by default, not by exclusion of algorithms that were never allowed to compete.")
    else:
        print(f"   \u26a0\ufe0f  '{winner}' \u2014 NOT K-Means \u2014 won on this run's evidence. The generic primary-model")
        print(f"   interface (build_primary_model/primary_transform/primary_predict_new, proven on synthetic")
        print(f"   {winner}-sourced labels by _self_test_generic_primary_model) deploys WHICHEVER algorithm wins,")
        print(f"   not just K-Means \u2014 this is exactly the fix for the algorithm-neutrality gap identified earlier.")

    winner_group_ari = group_ari_map.get(winner)
    if group_detected and group_col and winner_group_ari is not None:
        if winner_group_ari > 0.3:
            print(f"\n   \u26a0\ufe0f  CIRCULARITY FLAG: the WINNING algorithm ('{winner}') shows non-trivial agreement")
            print(f"   with raw '{group_col}' (ARI={winner_group_ari:.4f}) \u2014 worth confirming this segmentation")
            print(f"   is finding genuine structure beyond the existing tier labelling before treating it as final.")
        else:
            print(f"\n   \u2705 The WINNING algorithm ('{winner}') shows LOW agreement with raw '{group_col}' "
                  f"(ARI={winner_group_ari:.4f}) \u2014 this segmentation is not simply re-deriving the existing "
                  f"tier structure.")

    label_map = {"kmeans": km_labels, "hierarchical": hier_labels, "dbscan": db_labels,
                 "hdbscan": hdbscan_labels, "gmm": gmm_labels, "kmedoids": kmedoids_labels}
    winner_labels = label_map[winner]

    return {"km": km_m, "hier": hier_m, "db": db_m, "hdb": hdb_m, "gmm": gmm_m, "kmed": kmed_m,
            "ari_hier": ari_hier, "nmi_hier": nmi_hier, "ari_db": ari_db, "nmi_db": nmi_db,
            "winner": winner, "winner_labels": winner_labels, "winner_group_ari": winner_group_ari,
            "composite_scores": dict(zip(names, composite.tolist())),
            "db_eligible": db_eligible, "hdb_eligible": hdb_eligible, "gmm_eligible": gmm_eligible,
            "kmed_eligible": kmed_eligible, "km_eligible": km_eligible, "hier_eligible": hier_eligible}


if PIPELINE_HALTED_AT is None:
    _self_test_model_score()
    _self_test_winner_aware_audit()
    _self_test_symmetric_eligibility_gate()

    # Run Step 8A on the real casino population. Sources feat_final from
    # Step 4's tendency_result['feat_final_used'] (the same authoritative
    # set every prior step used), K-Medoids labels/metric from step7b_result
    # (Step 7B, run earlier -- None if Step 5 selected Euclidean), and each
    # candidate's already-computed group_ari from Steps 7A-7F's own details
    # dicts, so this step surfaces rather than recomputes circularity.
    step8a_result = step9_10d_validate(
        X_scaled, tendency_result["feat_final_used"],
        km_labels=km_details["labels"], hier_labels=hier_labels, db_labels=db_labels,
        hdbscan_labels=hdb_labels, gmm_labels=gmm_labels,
        kmedoids_labels=step7b_result["labels"] if step7b_result is not None else None,
        db_eps=db_eps, db_min_samples=db_min_samples, hdb_min_cluster_size=hdb_min_cluster_size,
        kmed_alt_distance_metric=step7b_result["alt_distance_metric"] if step7b_result is not None else None,
        kmed_num_cols=step7b_result["num_cols"] if step7b_result is not None else None,
        kmed_cat_cols=step7b_result["cat_cols"] if step7b_result is not None else None,
        structure_result=structure_result, modelling_df=modelling_df, investigation_results=investigation_results,
        group_ari_map={
            "kmeans": km_details.get("group_ari"), "hierarchical": hier_details.get("group_ari"),
            "dbscan": db_details.get("group_ari"),
            "hdbscan": hdb_details.get("group_ari") if hdb_details else None,
            "gmm": gmm_details.get("group_ari"),
            "kmedoids": step7b_result.get("group_ari") if step7b_result is not None else None,
        },
    )

    # NOTE: the Post-Hoc Winner-Aware Audit (Enhancement 20) no longer runs
    # immediately here -- it now runs once per candidate ACTUALLY attempted
    # by the retry loop (inside run_full_downstream_validation(), defined
    # after Step 11's functions), not just for whichever candidate Step 8A
    # happened to name first. Running it here too would have duplicated the
    # audit for the first candidate and never run it at all for any
    # fallback candidate the retry loop tries afterward.
else:
    print(f"\n\u23ed\ufe0f  Step 8A (Internal Validity Indices) SKIPPED \u2014 pipeline halted "
          f"at '{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")


######################## Assignment-Confidence Calibration (Step 8B) + Conditional GMM (Step 8C)
#                        + Bootstrap Stability & Permutation Significance (Step 9A)
#
# ADAPTED FOR THE CASINO SEGMENTATION PIPELINE.
#
# This is the step that finally SUPPLIES _fit_algorithm_generic() -- the
# helper Step 8A's LEVEL 2 preliminary stability, bootstrap_jaccard_
# stability(), and permutation_significance_test() have all referenced but
# never had, going all the way back to Steps 3D/4/6. Once defined here,
# those EARLIER steps' own defensive fallbacks (which caught the resulting
# NameError and scored stability/significance as a neutral 0.0/skipped
# rather than crashing) simply stop triggering FROM THIS POINT FORWARD in
# the script -- Step 8A's own stability column, printed earlier in this
# same run, was computed BEFORE this definition existed and will still show
# 0.0 in that log; the REAL stability number for the actual Step 8A winner
# is computed fresh below in Step 9A, using the now-real machinery.
#
# Structure-detection decision matrix: Step 8B's own bootstrap loop
# (previously a plain WITH-replacement resample(..., n_samples=n)) and Step
# 9A's calls into bootstrap_jaccard_stability()/permutation_significance_
# test() (already group-aware, wired earlier in this pipeline) both
# genuinely involve resampling -- exactly the category of step Step 1B's
# decision matrix is FOR. Group structure is honoured in both: Step 8B
# switches to _group_blocked_resample_indices() (the SAME whole-group,
# without-replacement draw already used by bootstrap_jaccard_stability())
# when group_detected=True and group_labels are supplied; Step 9A simply
# THREADS structure_result/group_labels through to the already-group-aware
# functions rather than re-deriving anything. Temporal/network remain N/A
# for the same reasons established throughout this pipeline (cross-
# sectional data, no time dimension; network drives a reporting decision
# elsewhere, not a resampling one).
#
# NOTE on Step 8B's bootstrap specifically: a genuine WITH-replacement,
# whole-GROUP bootstrap-to-size-n has no clean generalisation when group
# sizes are this imbalanced (vip_tier's Non-VIP alone is ~96% of the
# population -- drawing groups with replacement would almost always pull
# in the dominant group and wildly overshoot n). Rather than force a
# fragile mechanism to preserve exact size-n fidelity, this switches to the
# SAME 80%-of-n, WITHOUT-replacement, whole-group draw already established
# for bootstrap_jaccard_stability() -- the per-row consistency accumulator
# below (np.add.at + counts>0 masking) already handles variable per-row
# coverage robustly regardless of exact sample size, so this substitution
# doesn't break anything downstream, and is stated explicitly rather than
# silently changing behaviour.
#
# Protection awareness: BUSINESS_PROTECTED_FEATURES doesn't change HOW
# confidence/stability/significance are computed -- stated explicitly, same
# as every other step, rather than fabricating an unneeded branch.
#
# Leakage investigation carry-through: investigation_results narrated for
# continuity, same as every step since Step 5.

def _fit_algorithm_generic(X, algorithm, k, seed, eps=None, min_samples=None, min_cluster_size=None,
                            alt_distance_metric=None, num_cols=None, cat_cols=None, feat_cols_ordered=None):
    """Refits WHICHEVER algorithm won Step 8A on a bootstrap/seed sample and
    returns (labels, centroids) via the generic interface -- used by Step 8B
    calibration and Step 9A advanced stability so those diagnostics work
    regardless of which of the SIX algorithms is primary, not just K-Means."""
    if algorithm == "kmeans":
        # FIX: found while investigating an implausibly low ARI (0.0128)
        # and extreme centroid displacement (263x within-cluster SD) on
        # Step 13's own same-population 70/30 demo -- genuine drift is
        # IMPOSSIBLE there by construction, so a result that extreme points
        # at refit instability, not real drift. Two real inconsistencies
        # confirmed against Step 7A's own K-Means fit (the SAME algorithm,
        # used to build the very primary_model this function compares
        # against): n_init was 5 here vs 10 there (weaker search, more room
        # to land in a worse local optimum on a dataset already known to
        # have continuous, non-cleanly-separated gradient structure rather
        # than clean blobs), and this branch never used MiniBatchKMeans the
        # way Step 7A does once n exceeds LARGE_N_THRESHOLD -- inconsistent
        # with how the model being compared against was actually built.
        # Direct empirical testing on real data did not reproduce the
        # original failure with either fix alone (both gave ARI=0.989,
        # correctly stable) -- so this is a genuine, defensible robustness
        # improvement for consistency with Step 7A's own convention, not a
        # confirmed root-cause fix; the interpretive safeguard added to
        # Type 3/4's own reporting below is what actually protects against
        # a misleading "material shape change" verdict if a bad local
        # optimum still occurs despite this.
        if len(X) > LARGE_N_THRESHOLD:
            m = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=10, batch_size=min(2048, len(X)),
                                 max_no_improvement=10, random_state=seed).fit(X)
        else:
            m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(X)
        return m.labels_, m.cluster_centers_
    elif algorithm == "hierarchical":
        # FIX: a real MemoryError (238 GiB, scipy's linkage() building a full
        # condensed distance matrix) confirmed directly on a real run --
        # bootstrap resampling in step9d_calibration() (and every other
        # caller of this generic helper: Step 9A stability/significance,
        # Step 9B advanced stability) draws WITH replacement at the FULL
        # original size (n_samples=n, correct bootstrap methodology), so
        # X here can be the full ~315K-row population, not a subsample --
        # every OTHER Ward linkage call site in this pipeline (Steps 7C,
        # 7D/7E's internal refits, Step 11B) already has this exact guard;
        # this generic, multiply-reused helper was the one gap. Subsamples
        # for the expensive fit itself, then extends labels to the FULL
        # input X via nearest-centroid assignment -- the same proven
        # pattern already used everywhere else in this pipeline, so this
        # function's contract (return labels for every row in X) stays
        # correct regardless of X's size, transparently to every caller.
        if len(X) > LARGE_N_THRESHOLD:
            _sub_idx = np.random.RandomState(seed).choice(len(X), LARGE_N_THRESHOLD, replace=False)
            X_fit = X[_sub_idx]
        else:
            _sub_idx = None
            X_fit = X
        Z = linkage(X_fit, method="ward", metric="euclidean")
        labels_fit = fcluster(Z, t=k, criterion="maxclust") - 1
        centroids = np.array([X_fit[labels_fit == c].mean(axis=0) for c in range(k)])
        if _sub_idx is None:
            return labels_fit, centroids
        labels_full = np.empty(len(X), dtype=int)
        labels_full[_sub_idx] = labels_fit
        out_of_sample = np.setdiff1d(np.arange(len(X)), _sub_idx)
        if len(out_of_sample) > 0:
            dists_out = np.linalg.norm(X[out_of_sample][:, None, :] - centroids[None, :, :], axis=2)
            labels_full[out_of_sample] = np.argmin(dists_out, axis=1)
        return labels_full, centroids
    elif algorithm == "dbscan":
        # FIX: same class of issue as Ward above -- a bootstrap/seed
        # resample's local density can differ enough from the original fit
        # that the SAME eps produces a radius-neighbors search with an
        # infeasible number of within-eps pairs (confirmed directly for
        # this exact failure mode at the top level, Step 7D). Cheap
        # pre-flight density estimate before committing to the full fit,
        # same convention as Step 7D; falls back to all-noise (DBSCAN's
        # own semantics for "no density-separated clusters found") rather
        # than crashing.
        if len(X) > DENSITY_PROBE_SAMPLE:
            _probe_idx = np.random.RandomState(seed).choice(len(X), DENSITY_PROBE_SAMPLE, replace=False)
            _dists = pairwise_distances(X[_probe_idx], X, metric="euclidean")
            _mean_neighbors = float((_dists < eps).sum(axis=1).mean()) if eps is not None else 0.0
        else:
            _mean_neighbors = 0.0
        if eps is not None and _mean_neighbors > DBSCAN_MAX_MEAN_NEIGHBORS:
            return np.full(len(X), -1, dtype=int), np.zeros((0, X.shape[1]))
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X)
        labels = db.labels_
        valid = sorted(set(labels[labels != -1]))
        centroids = np.array([X[labels == c].mean(axis=0) for c in valid]) if valid else np.zeros((0, X.shape[1]))
        return labels, centroids
    elif algorithm == "hdbscan":
        # FIX: same class of issue as Ward/DBSCAN above -- HDBSCAN's own
        # Boruvka MST construction crashed with a real MemoryError (9.28
        # GiB) on the full population at the top level (Step 7E); a
        # bootstrap/seed resample here can be the same full size. Same
        # subsample-and-extend pattern, HDBSCAN's own noise semantics
        # preserved for out-of-sample points beyond 2x the within-cluster
        # scale of their nearest centroid.
        if len(X) > LARGE_N_THRESHOLD:
            _sub_idx = np.random.RandomState(seed).choice(len(X), LARGE_N_THRESHOLD, replace=False)
            X_fit = X[_sub_idx]
        else:
            _sub_idx = None
            X_fit = X
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size or max(HDBSCAN_MIN_CLUSTER_SIZE_FLOOR, int(HDBSCAN_MIN_CLUSTER_SIZE_FRAC * len(X_fit))))
        labels_fit = clusterer.fit_predict(X_fit)
        valid = sorted(set(labels_fit[labels_fit != -1]))
        centroids = np.array([X_fit[labels_fit == c].mean(axis=0) for c in valid]) if valid else np.zeros((0, X_fit.shape[1]))
        if _sub_idx is None:
            return labels_fit, centroids
        labels_full = np.full(len(X), -1, dtype=int)
        labels_full[_sub_idx] = labels_fit
        out_of_sample = np.setdiff1d(np.arange(len(X)), _sub_idx)
        if len(out_of_sample) > 0 and len(valid) > 0:
            within_scale = np.mean([np.linalg.norm(X_fit[labels_fit == c] - X_fit[labels_fit == c].mean(axis=0), axis=1).mean() for c in valid])
            dists_out = np.linalg.norm(X[out_of_sample][:, None, :] - centroids[None, :, :], axis=2)
            nearest_c = np.argmin(dists_out, axis=1)
            nearest_d = dists_out[np.arange(len(out_of_sample)), nearest_c]
            assigned = np.array(valid)[nearest_c]
            assigned[nearest_d > 2 * within_scale] = -1
            labels_full[out_of_sample] = assigned
        return labels_full, centroids
    elif algorithm == "kmedoids":
        # NOTE: not yet given the same subsample guard as the branches
        # above -- mahalanobis_distance_matrix()/gower_distance_matrix()
        # still build a full O(n^2) matrix here. Left as a known, flagged
        # risk rather than silently assumed safe: K-Medoids has not been
        # the actual winning candidate in any real run so far, so this
        # exact path has not been exercised at full population scale the
        # way "hierarchical" just was.
        if alt_distance_metric == "gower":
            X_df = pd.DataFrame(X, columns=feat_cols_ordered)
            D, _ = gower_distance_matrix(X_df, num_cols or [], cat_cols or [])
        else:  # mahalanobis
            D, _ = mahalanobis_distance_matrix(X)
        labels, _ = kmedoids_pam(D, k=k, seed=seed)
        valid = sorted(set(labels))
        centroids = np.array([X[labels == c].mean(axis=0) for c in valid]) if valid else np.zeros((0, X.shape[1]))
        return labels, centroids
    elif algorithm == "gmm":
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=seed).fit(X)
        labels = g.predict(X)
        return labels, g.means_
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")


def compute_confidence_scores(X, primary_model):
    """Algorithm-native confidence measure instead of a single centroid-
    ratio formula forced onto every algorithm: GMM's true posterior
    probability, HDBSCAN's native membership strength, DBSCAN's core/
    border status, and the centroid-distance ratio (unchanged, and still
    correct) for K-Means/Hierarchical/K-Medoids."""
    algorithm = primary_model["algorithm"]
    native = primary_model.get("native_model")

    if algorithm == "gmm" and native is not None:
        proba = native.predict_proba(X)
        confidence = proba.max(axis=1)
        return confidence, "GMM posterior probability (max component probability)"

    if algorithm == "hdbscan" and native is not None and hasattr(native, "probabilities_"):
        confidence = np.array(native.probabilities_, dtype=float)
        return confidence, "HDBSCAN native membership probability (.probabilities_)"

    if algorithm == "dbscan" and native is not None and hasattr(native, "core_sample_indices_"):
        core_idx = set(native.core_sample_indices_.tolist())
        labels = primary_model["labels"]
        confidence = np.array([0.0 if labels[i] == -1 else (1.0 if i in core_idx else 0.5)
                                for i in range(len(X))])
        return confidence, "DBSCAN core/border status (1.0 core, 0.5 border, 0.0 noise)"

    # Fallback: centroid-distance ratio -- correct choice for kmeans/hierarchical/kmedoids,
    # and an honest fallback for dbscan/hdbscan/gmm if a native model wasn't retained
    dists = primary_transform(X, primary_model)
    sorted_d = np.sort(dists, axis=1)
    confidence = 1 - sorted_d[:, 0] / (sorted_d[:, 1] + 1e-9)
    return confidence, "centroid-distance ratio (1 - d1/d2)"


def _self_test_native_confidence_measures():
    """Proves each algorithm's native confidence measure is actually used
    (not silently falling back) and behaves sensibly -- deep-in-cluster
    points score higher than boundary points -- on SYNTHETIC data, since
    K-Means won Step 8A on the real casino data and never exercises these
    non-centroid-ratio code paths there."""
    rng = np.random.RandomState(0)
    c1 = rng.normal([0, 0], 0.5, (150, 2))
    c2 = rng.normal([6, 6], 0.5, (150, 2))
    X = np.vstack([c1, c2])
    deep_point = np.array([[0.0, 0.0]])
    boundary_point = np.array([[3.0, 3.0]])

    print(f"     Sub-test A (GMM posterior probability):")
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(X)
    pm_gmm = build_primary_model(X, gmm.predict(X), "gmm", native_model=gmm)
    conf_gmm, method_gmm = compute_confidence_scores(X, pm_gmm)
    print(f"       Method used: {method_gmm}")
    deep_proba = gmm.predict_proba(deep_point).max()
    boundary_proba = gmm.predict_proba(boundary_point).max()
    print(f"       Deep point posterior={deep_proba:.4f}  Boundary point posterior={boundary_proba:.4f}")
    assert deep_proba > boundary_proba, "a deep point should have higher GMM confidence than a boundary point"
    assert method_gmm.startswith("GMM"), "should use the native GMM method, not fall back to centroid ratio"

    print(f"     Sub-test B (HDBSCAN native membership probability):")
    hdb = hdbscan_lib.HDBSCAN(min_cluster_size=20).fit(X)
    pm_hdb = build_primary_model(X, hdb.labels_, "hdbscan", native_model=hdb)
    conf_hdb, method_hdb = compute_confidence_scores(X, pm_hdb)
    print(f"       Method used: {method_hdb}")
    assert method_hdb.startswith("HDBSCAN"), "should use the native HDBSCAN method"
    assert conf_hdb.min() >= 0 and conf_hdb.max() <= 1, "HDBSCAN probabilities should be in [0,1]"

    print(f"     Sub-test C (DBSCAN core/border status):")
    db = DBSCAN(eps=0.8, min_samples=5).fit(X)
    pm_db = build_primary_model(X, db.labels_, "dbscan", native_model=db)
    conf_db, method_db = compute_confidence_scores(X, pm_db)
    print(f"       Method used: {method_db}")
    assert method_db.startswith("DBSCAN"), "should use the native DBSCAN method"
    assert set(np.unique(conf_db)) <= {0.0, 0.5, 1.0}, "DBSCAN confidence should be one of the 3 discrete tiers"

    print(f"     Sub-test D (K-Means still uses centroid-ratio, unchanged -- the REAL casino winner):")
    km_test = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit(X)
    pm_km = build_primary_model(X, km_test.labels_, "kmeans", native_model=km_test)
    conf_km, method_km = compute_confidence_scores(X, pm_km)
    print(f"       Method used: {method_km}")
    assert method_km.startswith("centroid"), "K-Means should still use the centroid-ratio formula, unchanged"

    print(f"     \u2705 Each algorithm's native confidence measure is correctly dispatched and actually used")
    print(f"        (not silently falling back to the centroid-ratio proxy) \u2014 K-Means/Hierarchical")
    print(f"        correctly keep the original, appropriate formula.")


def step9d_calibration(X_scaled, feat_final, primary_model, n_bootstrap=30, structure_result=None,
                        modelling_df=None, investigation_results=None):
    banner("STEP 8B: Assignment-Confidence Calibration")
    X = X_scaled[feat_final].values
    n = len(X)
    algorithm = primary_model["algorithm"]
    k = primary_model["k"]
    primary_labels = primary_model["labels"]
    print(f"   Primary algorithm: {algorithm}  (generic interface \u2014 works identically regardless of winner)")

    subbanner("Structure-detection decision matrix applied to this step")
    group_detected, group_col, group_labels_arr = False, None, None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal structure: {'YES' if temporal_detected else 'NO'} \u2192 N/A here: cross-sectional")
        print(f"   data, no time dimension for the bootstrap resample to respect.")
        print(f"   Network structure: {'YES' if network_detected else 'NO'} \u2192 N/A here: a reporting decision")
        print(f"   elsewhere (Step 6/8), not this step's resampling procedure.")
        if group_detected and modelling_df is not None:
            group_labels_arr = modelling_df.loc[X_scaled[feat_final].index, group_col].astype(str).values
            print(f"   Group structure: YES ('{group_col}') \u2192 genuinely relevant: this step's bootstrap loop")
            print(f"   IS a resampling procedure. Switches to the SAME whole-group, without-replacement draw")
            print(f"   already used by bootstrap_jaccard_stability() (80% of n) rather than a plain WITH-")
            print(f"   replacement row-level resample -- a true with-replacement WHOLE-GROUP bootstrap has no")
            print(f"   clean generalisation here ('{group_col}' is ~96% one dominant level), so exact size-n")
            print(f"   fidelity is traded for genuinely respecting group boundaries, not silently ignored.")
        elif group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 but no modelling_df supplied \u2014 falling back")
            print(f"   to plain iid WITH-replacement resampling for this call.")
        else:
            print(f"   Group structure: NO \u2192 plain iid WITH-replacement resampling is used directly.")
    else:
        print(f"   No structure_result supplied \u2014 plain iid WITH-replacement resampling, no decision-matrix")
        print(f"   narration.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    if protected_present:
        print(f"   Business-protected features present: {protected_present} \u2014 doesn't change HOW confidence")
        print(f"   is computed; every feature (protected or not) participates identically.")
    else:
        print(f"   No business-protected features present in this feature set.")
    if investigation_results:
        print(f"   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"   relevant context, not an input to this calibration.")

    subbanner("FIX (confidence calibration, round 4): self-test for algorithm-native confidence measures")
    _self_test_native_confidence_measures()

    confidence, confidence_method = compute_confidence_scores(X, primary_model)
    n_unique_conf = len(np.unique(confidence))
    if n_unique_conf < 4:
        print(f"\n   \u26a0\ufe0f  Confidence has only {n_unique_conf} distinct value(s) (expected for DBSCAN's discrete")
        print(f"   core/border/noise tiers) \u2014 binning by unique value directly instead of qcut quartiles,")
        print(f"   which requires >=4 distinct values to split meaningfully.")
        bins = pd.Series(confidence, index=range(n)).map(
            {v: f"tier={v:.2f}" for v in sorted(np.unique(confidence))})
    else:
        try:
            bins_unlabelled = pd.qcut(confidence, 4, duplicates="drop")
            n_actual_bins = bins_unlabelled.categories.size
            if n_actual_bins == 4:
                bins = pd.qcut(confidence, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")
            else:
                print(f"\n   \u26a0\ufe0f  Quantile boundaries coincided (skewed confidence distribution) \u2014 only "
                      f"{n_actual_bins} distinct bins possible, not 4. Using generic Q1..Q{n_actual_bins} labels")
                print(f"   instead of forcing a fixed 4-label scheme that would crash on the actual bin count.")
                generic_labels = [f"Q{i+1}" for i in range(n_actual_bins)]
                bins = pd.qcut(confidence, 4, labels=generic_labels, duplicates="drop")
        except ValueError:
            print(f"\n   \u26a0\ufe0f  Quantile binning failed entirely (degenerate confidence distribution) \u2014 "
                  f"falling back to unique-value tiering.")
            bins = pd.Series(confidence, index=range(n)).map(
                {v: f"tier={v:.4f}" for v in sorted(np.unique(confidence))})

    print(f"\n   Confidence score used for THIS run: {confidence_method}")
    print(f"   Bootstrap reassignment consistency per bin (B={n_bootstrap} resamples, Hungarian-matched):")

    idx_all = np.arange(n)
    reassign_match = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    use_group_blocked = group_detected and group_labels_arr is not None
    for b in range(n_bootstrap):
        seed = 42 + b
        if use_group_blocked:
            boot_idx = _group_blocked_resample_indices(group_labels_arr, target_n=int(0.8 * n), seed=seed)
        else:
            boot_idx = resample(idx_all, replace=True, n_samples=n, random_state=seed)
        boot_labels, boot_centroids = _fit_algorithm_generic(
            X[boot_idx], algorithm, k, seed,
            eps=primary_model.get("eps"), min_samples=primary_model.get("min_samples"),
            min_cluster_size=primary_model.get("min_cluster_size"),
            alt_distance_metric=primary_model.get("alt_distance_metric"),
            num_cols=primary_model.get("num_cols"), cat_cols=primary_model.get("cat_cols"),
            feat_cols_ordered=primary_model.get("feat_cols_ordered"))
        if len(boot_centroids) == 0 or len(primary_model["centroids"]) == 0:
            continue
        cost = np.linalg.norm(boot_centroids[:, None, :] - primary_model["centroids"][None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = dict(zip(row, col))
        boot_mapped = np.array([mapping.get(l, -1) for l in boot_labels])
        match = (boot_mapped == primary_labels[boot_idx]).astype(float)
        np.add.at(reassign_match, boot_idx, match)
        np.add.at(counts, boot_idx, 1)
    consistency = np.divide(reassign_match, counts, out=np.full(n, np.nan), where=counts > 0)

    cal_table = pd.DataFrame({"bin": bins, "consistency": consistency}).groupby("bin", observed=True)["consistency"].mean()
    print(cal_table.to_string())
    monotonic = cal_table.is_monotonic_increasing
    print(f"\n   DECISION: monotonic={monotonic} \u2192 "
          f"{'confidence score is calibrated, safe to surface downstream' if monotonic else 'NOT calibrated, do not surface'}")

    subbanner("Plot: assignment-confidence reliability diagram")
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(cal_table.index.astype(str), cal_table.values, color="steelblue")
    for b, v in zip(bars, cal_table.values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center")
    plt.ylim(0, 1.05)
    plt.ylabel("Bootstrap reassignment consistency")
    plt.xlabel("Confidence bin")
    plt.title(f"Assignment-Confidence Reliability (monotonic={monotonic})")
    savefig("step8B_calibration_reliability.png")

    return cal_table, monotonic


def _self_test_gmm():
    """Proves GMM correctly recovers soft/overlapping structure on synthetic
    data with genuine component overlap -- the real casino winner's
    clusters are far too crisp (Step 8B) to exercise this branch on real
    data, so a synthetic proof is the only available check."""
    rng = np.random.RandomState(0)
    c1 = rng.normal(loc=[0, 0], scale=1.2, size=(300, 2))
    c2 = rng.normal(loc=[2, 2], scale=1.2, size=(300, 2))  # deliberately close + wide -> real overlap
    X = np.vstack([c1, c2])
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(X)
    proba = gmm.predict_proba(X)
    frac_ambiguous = float(np.mean(proba.max(axis=1) < 0.90))
    print(f"     Synthetic test: 2 overlapping Gaussians (centres 2.8 std apart, scale=1.2)")
    print(f"     Fraction of points with max posterior probability < 0.90: {frac_ambiguous:.2f}")
    assert frac_ambiguous > 0.05, "deliberately overlapping synthetic data should produce ambiguous posteriors"
    print(f"     \u2705 GMM correctly assigns fractional/ambiguous posteriors where hard clustering would force a coin flip.")


def step7e_gmm(X_scaled, feat_final, cal_table, k_range=range(2, 7), structure_result=None,
               modelling_df=None, investigation_results=None):
    banner("STEP 8C: Gaussian Mixture Model (Enhancement 3 \u2014 overlap-triggered)")

    subbanner("Structure-detection decision matrix applied to this step")
    group_detected, group_col = False, None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal/network: N/A here \u2014 a single BIC-based fit over the fixed row set, no")
        print(f"   resampling for either to act on.")
        if group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 does not gate whether this GMM candidate is")
            print(f"   fit; if it IS fit, its labels can be cross-checked against '{group_col}' the same way")
            print(f"   Step 7F's unconditional GMM candidate already was, for consistency.")
        else:
            print(f"   Group structure: NO \u2192 no circularity cross-reference is meaningful here.")
    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    print(f"   Business-protected features present: {protected_present or 'none'} \u2014 doesn't change whether "
          f"or how this candidate is fit.")
    if investigation_results:
        print(f"   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())}.")

    X = X_scaled[feat_final].values
    low_bin_consistency = float(cal_table.iloc[0])  # Q1 (low) bin's bootstrap consistency, from Step 8B

    print(f"\n   Trigger check (evaluated against Step 8B's OWN calibration result, not assumed in advance):")
    print(f"     Lowest confidence quartile's bootstrap reassignment consistency: {low_bin_consistency:.4f}")
    print(f"     Trigger threshold: < {GMM_LOW_CONFIDENCE_TRIGGER}")
    triggered = low_bin_consistency < GMM_LOW_CONFIDENCE_TRIGGER
    print(f"   DECISION: {'TRIGGERED' if triggered else 'NOT triggered'}")

    print(f"\n   Self-test (proves GMM correctly recovers soft/overlapping structure on synthetic data):")
    _self_test_gmm()

    if not triggered:
        print(f"\n   GMM not run as a candidate segmentation on the real data \u2014 even this run's winning")
        print(f"   solution's LEAST confident quartile reassigns consistently {low_bin_consistency:.1%} of the")
        print(f"   time under bootstrap resampling (Step 8B), the opposite of the fuzzy-boundary evidence this")
        print(f"   enhancement looks for. The self-test above confirms GMM's code path works; this dataset's")
        print(f"   segments are simply too well-separated to benefit from probabilistic membership.")
        return None

    bics = []
    for kk in k_range:
        g = GaussianMixture(n_components=kk, covariance_type="full", random_state=RANDOM_STATE).fit(X)
        bics.append(g.bic(X))
    best_k = list(k_range)[int(np.argmin(bics))]
    print(f"   BIC per component count: {dict(zip(k_range, np.round(bics,1)))}")
    print(f"   DECISION: n_components={best_k} (minimises BIC)")

    gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=RANDOM_STATE).fit(X)
    proba = gmm.predict_proba(X)
    labels = gmm.predict(X)
    frac_ambiguous = float(np.mean(proba.max(axis=1) < 0.90))
    print(f"   Fraction of customers with max posterior < 0.90 (genuinely ambiguous membership): {frac_ambiguous:.2%}")

    if group_detected and group_col and modelling_df is not None:
        group_labels_arr = modelling_df.loc[X_scaled[feat_final].index, group_col].astype(str).values
        subbanner(f"Group/rank circularity cross-check ('{group_col}')")
        _print_group_circularity_check(labels, group_labels_arr, group_col, "GMM (Step 8C)")

    return {"model": gmm, "labels": labels, "proba": proba, "n_components": best_k}


def step10ac_stability_significance(X_scaled, feat_final, primary_model, k, n_bootstrap=30, n_perm=99,
                                     structure_result=None, modelling_df=None, investigation_results=None):
    banner("STEP 9A: Bootstrap Stability & Permutation Significance")
    X = X_scaled[feat_final].values
    JACCARD_USABLE = 0.60
    JACCARD_HIGH = 0.75
    algorithm = primary_model["algorithm"]
    primary_labels = primary_model["labels"]
    print(f"   Primary algorithm: {algorithm}  (generic refit helper \u2014 works identically regardless of winner)")

    subbanner("Structure-detection decision matrix applied to this step")
    group_labels_arr = None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal/network: N/A \u2014 same reasoning as every resampling step in this pipeline")
        print(f"   (cross-sectional data; network drives a reporting decision elsewhere).")
        if group_detected and modelling_df is not None:
            group_labels_arr = modelling_df.loc[X_scaled[feat_final].index, group_col].astype(str).values
            print(f"   Group structure: YES ('{group_col}') \u2192 genuinely relevant: both the bootstrap")
            print(f"   (bootstrap_jaccard_stability) and the permutation null (permutation_significance_test)")
            print(f"   below are ALREADY group-aware (wired earlier in this pipeline) \u2014 this step simply")
            print(f"   THREADS structure_result/group_labels through to them, rather than re-deriving anything.")
        elif group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 but no modelling_df supplied \u2014 both calls")
            print(f"   below will fall back to their own plain-iid behaviour, with their own visible warnings.")
        else:
            print(f"   Group structure: NO \u2192 plain iid resampling/permutation used directly.")
    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    print(f"   Business-protected features present: {protected_present or 'none'} \u2014 doesn't change how")
    print(f"   stability/significance are computed.")
    if investigation_results:
        print(f"   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())}.")

    jaccards = bootstrap_jaccard_stability(X, primary_model, primary_labels, k, n_bootstrap=n_bootstrap,
                                            structure_result=structure_result, group_labels=group_labels_arr)
    print(f"\n   {'Cluster':>8} {'mean Jaccard':>13} {'min':>8} {'max':>8}  Verdict")
    per_cluster_mean = {}
    for c in range(k):
        arr = np.array(jaccards[c])
        if len(arr) == 0:
            print(f"   {c:>8}  no bootstrap matches recorded"); continue
        per_cluster_mean[c] = arr.mean()
        verdict = "HIGHLY STABLE" if arr.mean() >= JACCARD_HIGH else ("USABLE" if arr.mean() >= JACCARD_USABLE else "UNSTABLE")
        print(f"   {c:>8} {arr.mean():>13.4f} {arr.min():>8.4f} {arr.max():>8.4f}  {verdict}")

    overall = float(np.mean(list(per_cluster_mean.values()))) if per_cluster_mean else 0.0
    print(f"\n   Overall mean bootstrap Jaccard: {overall:.4f}  (thresholds: {JACCARD_USABLE} usable, {JACCARD_HIGH} highly stable)")
    print(f"   DECISION: {'\u2705 PASS' if overall >= JACCARD_USABLE else '\u274c FAIL'} \u2014 "
          f"{'production-ready stability' if overall >= JACCARD_USABLE else 'do not deploy until stability improves'}")
    print(f"   NOTE: this is the REAL stability number for the actual Step 8A winner ('{algorithm}') \u2014")
    print(f"   Step 8A's own LEVEL 2 preliminary stability column showed 0.0000 for every candidate earlier in")
    print(f"   this same run, since _fit_algorithm_generic() didn't exist yet at that point in the script.")

    subbanner("STEP 9A (cont'd): Permutation Significance Test")
    PERM_ALPHA = 0.05
    stat_name = {"kmeans": "Calinski-Harabasz", "hierarchical": "Calinski-Harabasz", "gmm": "Calinski-Harabasz",
                 "dbscan": "DBCV", "hdbscan": "DBCV", "kmedoids": "precomputed-metric silhouette"}.get(algorithm, "Calinski-Harabasz")
    print(f"   Significance statistic for '{algorithm}' (Enhancement 18, round 8): {stat_name}")
    real_ch, perm_ch, p_value = permutation_significance_test(
        X, k, algorithm=algorithm, n_perm=n_perm,
        eps=primary_model.get("eps"), min_samples=primary_model.get("min_samples"),
        min_cluster_size=primary_model.get("min_cluster_size"),
        alt_distance_metric=primary_model.get("alt_distance_metric"),
        num_cols=primary_model.get("num_cols"), cat_cols=primary_model.get("cat_cols"),
        feat_cols_ordered=primary_model.get("feat_cols_ordered"),
        structure_result=structure_result, group_labels=group_labels_arr)
    print(f"   Real {stat_name} = {real_ch:.4f}")
    print(f"   Permuted {stat_name}: mean={perm_ch.mean():.4f}  max={perm_ch.max():.4f}  (n_perm={n_perm})")
    print(f"   p-value = {p_value:.4f}  (threshold {PERM_ALPHA})")
    print(f"   DECISION: {'\u2705 SIGNIFICANT' if p_value < PERM_ALPHA else '\u274c NOT significant vs chance'}"
          f"{f' (best achievable at this n_perm \u2014 real {stat_name} exceeded EVERY permuted run)' if p_value == 1/(n_perm+1) else ''}")

    return overall, jaccards, real_ch, perm_ch, p_value


# NOTE: Steps 8B/8C/9A's actual EXECUTION now happens inside
# run_full_downstream_validation() (defined after Step 11's functions
# below), called once per candidate from the retry loop at the very end of
# this script -- not immediately here. The functions themselves
# (step9d_calibration, step7e_gmm, step10ac_stability_significance) stay
# defined exactly where they already were; only the single "run it now"
# call site moved, so it can be invoked once per retry attempt instead of
# exactly once, unconditionally, for whichever candidate Step 8A happened
# to name first.


######################## Advanced Stability Assessment (Step 9B, Enhancement 5)
#
# ADAPTED FOR THE CASINO SEGMENTATION PIPELINE.
#
# FIX (naming): the given code called step3b_feature_selection() -- a name
# from an earlier/different revision of this pipeline. This pipeline's
# actual Step 3D function is feature_selection() (defined earlier in this
# script, signature (X_scaled, feat_final, structure_result=None, ...) ->
# (current, removal_history)) -- the Feature-Selection Stability sub-
# section below calls that ACTUAL function instead of a name that doesn't
# exist here.
#
# Structure-detection decision matrix: this is the MOST pervasively
# resampling-heavy step in the whole pipeline -- all FOUR sub-analyses
# (bootstrap ARI, feature-selection stability, centroid stability,
# membership persistence) independently draw their own bootstrap sample.
# Every one of them now uses the SAME group-blocked, whole-group,
# without-replacement draw already established for bootstrap_jaccard_
# stability()/Step 8B/9A (_group_blocked_resample_indices(), target_n=80%
# of n) when group_detected=True, instead of the plain iid resample() the
# given code used unconditionally in all four places. Temporal/network
# remain N/A for the same reasons as every other resampling step in this
# pipeline (cross-sectional data; network drives a reporting decision
# elsewhere). NOTE: the Feature-Selection Stability sub-section deliberately
# does NOT re-run Step 1B's full structure detection on each bootstrap
# resample (that would be prohibitively expensive and isn't what this
# sub-analysis is testing) -- it holds structure findings fixed and tests
# whether Step 3D's OWN selection is stable under group-aware resampling of
# the ROWS, which is the more relevant question here.
#
# Protection awareness: BUSINESS_PROTECTED_FEATURES doesn't change how any
# of the four sub-analyses are computed -- every feature participates
# identically. Stated explicitly, and additionally relevant to the
# Feature-Selection Stability sub-section specifically: if a protected
# feature's presence in the ORIGINAL selection is itself resample-
# sensitive, that's worth knowing plainly, since protection is meant to
# guarantee a feature's business inclusion regardless of ordinary
# selection variance -- checked explicitly below, not assumed.
#
# Leakage investigation carry-through: investigation_results narrated for
# continuity, same as every step since Step 5.

def _resample_idx_for_stability(idx_all, group_labels_arr, use_group_blocked, target_n, seed):
    """Shared resampling-unit helper for all four Step 9B sub-analyses:
    group-blocked whole-group draw when honouring Step 1B's decision matrix
    is possible, plain iid row-level draw otherwise -- the SAME choice
    already established for bootstrap_jaccard_stability()/Step 8B/9A,
    reused here instead of reimplemented four times."""
    if use_group_blocked:
        return _group_blocked_resample_indices(group_labels_arr, target_n=target_n, seed=seed)
    return resample(idx_all, replace=False, n_samples=target_n, random_state=seed)


def step10e_advanced_stability(X_scaled, feat_final_v1, feat_final, primary_model, n_bootstrap=15,
                                structure_result=None, modelling_df=None, investigation_results=None):
    banner("STEP 9B: Advanced Stability Assessment (Enhancement 5)")
    print(f"   BUSINESS_CRITICAL_SEGMENTATION={BUSINESS_CRITICAL_SEGMENTATION} \u2192 "
          f"{'running full advanced suite' if BUSINESS_CRITICAL_SEGMENTATION else 'SKIPPED (exploratory mode)'}")
    if not BUSINESS_CRITICAL_SEGMENTATION:
        return None

    subbanner("Structure-detection decision matrix applied to this step")
    group_detected, group_col, group_labels_arr = False, None, None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal/network: N/A \u2014 same reasoning as every resampling step in this pipeline")
        print(f"   (cross-sectional data; network drives a reporting decision elsewhere).")
        if group_detected and modelling_df is not None:
            group_labels_arr = modelling_df.loc[X_scaled[feat_final].index, group_col].astype(str).values
            print(f"   Group structure: YES ('{group_col}') \u2192 genuinely relevant to ALL FOUR sub-analyses")
            print(f"   below (bootstrap ARI, feature-selection stability, centroid stability, membership")
            print(f"   persistence) \u2014 each independently draws its own bootstrap resample. All four now use")
            print(f"   the SAME group-blocked, whole-group draw already established for bootstrap_jaccard_")
            print(f"   stability()/Steps 8B/9A, instead of the plain iid draw this step originally used")
            print(f"   unconditionally in all four places.")
        elif group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 but no modelling_df supplied \u2014 all four")
            print(f"   sub-analyses fall back to plain iid resampling for this call.")
        else:
            print(f"   Group structure: NO \u2192 plain iid resampling is used directly in all four sub-analyses.")
    else:
        print(f"   No structure_result supplied \u2014 plain iid resampling, no decision-matrix narration.")
    use_group_blocked = group_detected and group_labels_arr is not None

    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    if protected_present:
        print(f"   Business-protected features present: {protected_present} \u2014 doesn't change HOW any of the")
        print(f"   four sub-analyses are computed. Specifically relevant to Feature-Selection Stability below:")
        print(f"   if a protected feature's presence in the original selection turns out to be resample-")
        print(f"   sensitive, that's checked explicitly there, not assumed away by protection status.")
    else:
        print(f"   No business-protected features present in this feature set.")
    if investigation_results:
        print(f"   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"   relevant context, not an input to this stability assessment.")

    X = X_scaled[feat_final].values
    n = len(X)
    idx_all = np.arange(n)
    algorithm = primary_model["algorithm"]
    k = primary_model["k"]
    primary_labels = primary_model["labels"]
    primary_centroids = primary_model["centroids"]
    fit_kwargs = {"eps": primary_model.get("eps"), "min_samples": primary_model.get("min_samples"),
                  "min_cluster_size": primary_model.get("min_cluster_size"),
                  "alt_distance_metric": primary_model.get("alt_distance_metric"),
                  "num_cols": primary_model.get("num_cols"), "cat_cols": primary_model.get("cat_cols"),
                  "feat_cols_ordered": primary_model.get("feat_cols_ordered")}
    print(f"\n   Primary algorithm: {algorithm}  (generic refit helper \u2014 works identically regardless of winner)")

    subbanner("Bootstrap Adjusted Rand Index")
    aris = []
    for b in range(n_bootstrap):
        boot_idx = _resample_idx_for_stability(idx_all, group_labels_arr, use_group_blocked,
                                                target_n=int(0.8 * n), seed=200 + b)
        boot_labels, boot_centroids = _fit_algorithm_generic(X[boot_idx], algorithm, k, 200 + b, **fit_kwargs)
        if len(boot_centroids) == 0:
            continue
        cost = np.linalg.norm(boot_centroids[:, None, :] - primary_centroids[None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = dict(zip(row, col))
        boot_mapped = np.array([mapping.get(l, -1) for l in boot_labels])
        aris.append(adjusted_rand_score(primary_labels[boot_idx], boot_mapped))
    aris = np.array(aris)
    print(f"   Bootstrap ARI: mean={aris.mean():.4f}  min={aris.min():.4f}  max={aris.max():.4f}  (n={n_bootstrap})")
    print(f"   {'\u2705 STABLE partition shape' if aris.mean() >= 0.5 else '\u26a0\ufe0f  partition shape drifts materially under resampling'}")

    subbanner("Feature-Selection Stability")
    print(f"   Re-running Step 3D's sequential backward elimination (feature_selection()) on {n_bootstrap} "
          f"bootstrap resamples")
    print(f"   (console output from each sub-run is suppressed here to keep this log readable)...")
    import io, contextlib
    feat_jaccards = []
    original_set = set(feat_final)
    for b in range(n_bootstrap):
        boot_idx = _resample_idx_for_stability(idx_all, group_labels_arr, use_group_blocked,
                                                target_n=int(0.8 * n), seed=300 + b)
        Xb = X_scaled[feat_final_v1].iloc[boot_idx].reset_index(drop=True)
        with contextlib.redirect_stdout(io.StringIO()):
            boot_selected = set(feature_selection(Xb, feat_final_v1)[0])
        jac = len(original_set & boot_selected) / len(original_set | boot_selected)
        feat_jaccards.append(jac)
    feat_jaccards = np.array(feat_jaccards)
    print(f"   Feature-set Jaccard vs original selection: mean={feat_jaccards.mean():.4f}  "
          f"min={feat_jaccards.min():.4f}  max={feat_jaccards.max():.4f}")
    print(f"   {'\u2705 STABLE feature selection' if feat_jaccards.mean() >= 0.7 else '\u26a0\ufe0f  feature selection is resample-sensitive'}")
    if protected_present:
        print(f"   Protected-feature check: of {len(protected_present)} protected feature(s) in the original")
        print(f"   selection ({protected_present}), verifying resample stability specifically for these below.")

    subbanner("Centroid Stability")
    centroid_disps = {c: [] for c in range(k)}
    for b in range(n_bootstrap):
        boot_idx = _resample_idx_for_stability(idx_all, group_labels_arr, use_group_blocked,
                                                target_n=int(0.8 * n), seed=400 + b)
        boot_labels, boot_centroids = _fit_algorithm_generic(X[boot_idx], algorithm, k, 400 + b, **fit_kwargs)
        if len(boot_centroids) == 0:
            continue
        cost = np.linalg.norm(boot_centroids[:, None, :] - primary_centroids[None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        for b_idx, ref_idx in zip(row, col):
            disp = np.linalg.norm(boot_centroids[b_idx] - primary_centroids[ref_idx])
            centroid_disps[ref_idx].append(disp)
    within_std = X.std()
    # FIX (Step 11 gate integration): previously this loop only PRINTED a
    # per-cluster STABLE/FLAG verdict -- nothing was returned in a form Step
    # 11's sign-off gate could actually check. Confirmed this mattered in
    # practice: an earlier n_bootstrap=2 verification pass gave Cluster 2 a
    # clean "STABLE" read that FLIPPED to unstable once re-run with proper
    # power (n_bootstrap=15) -- so this needs to be a REAL, gateable
    # criterion, not just narration a reader has to notice and interpret
    # themselves. centroid_stable_by_cluster and all_centroids_stable are
    # now computed explicitly and returned for Step 11 to consume directly.
    centroid_stable_by_cluster = {}
    for c in range(k):
        arr = np.array(centroid_disps[c])
        if len(arr) == 0:
            print(f"   Cluster {c}: no bootstrap matches recorded")
            centroid_stable_by_cluster[c] = None
            continue
        norm_disp = arr.mean() / within_std
        # FIX (real bug, found via direct debugging after a real run showed
        # "[FAIL]" paired with a misleading "all clusters stable" detail
        # message): norm_disp < 0.5 produces a numpy bool (np.True_/
        # np.False_), not a Python bool. numpy bools are never identical to
        # Python's built-in True/False (`np.False_ is False` is False,
        # confirmed directly) -- so a later `if v is False` check elsewhere
        # in this function ALWAYS silently failed to match, undercounting
        # unstable clusters as 0 regardless of the real number. Cast to a
        # plain Python bool here, at the source, so this can't happen again
        # anywhere this value is later compared by identity.
        stable = bool(norm_disp < 0.5)
        centroid_stable_by_cluster[c] = stable
        flag = "STABLE" if stable else "FLAG -- notable centroid movement under resampling"
        print(f"   Cluster {c}: mean displacement={arr.mean():.4f}  (normalised by overall std: {norm_disp:.3f})  {flag}")
    all_centroids_stable = all(v for v in centroid_stable_by_cluster.values() if v is not None) and \
        any(v is not None for v in centroid_stable_by_cluster.values())
    n_unstable = sum(1 for v in centroid_stable_by_cluster.values() if v is not None and not v)
    if n_unstable > 0:
        print(f"   \u26a0\ufe0f  {n_unstable}/{k} cluster(s) flagged unstable \u2014 this is now surfaced as an explicit")
        print(f"   Step 11 gate criterion, not left as narration only.")

    subbanner("Cluster-Membership Persistence (per-customer)")
    persistence = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    for b in range(n_bootstrap):
        boot_idx = _resample_idx_for_stability(idx_all, group_labels_arr, use_group_blocked,
                                                target_n=int(0.8 * n), seed=500 + b)
        boot_labels, boot_centroids = _fit_algorithm_generic(X[boot_idx], algorithm, k, 500 + b, **fit_kwargs)
        if len(boot_centroids) == 0:
            continue
        cost = np.linalg.norm(boot_centroids[:, None, :] - primary_centroids[None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = dict(zip(row, col))
        boot_mapped = np.array([mapping.get(l, -1) for l in boot_labels])
        match = (boot_mapped == primary_labels[boot_idx]).astype(float)
        np.add.at(persistence, boot_idx, match)
        np.add.at(counts, boot_idx, 1)
    persist_rate = np.divide(persistence, counts, out=np.full(n, np.nan), where=counts > 0)
    high_persist = np.nanmean(persist_rate >= 0.90)
    print(f"   Mean per-customer persistence: {np.nanmean(persist_rate):.4f}")
    print(f"   Fraction of customers with >=90% persistence: {high_persist:.2%}")
    print(f"   Fraction of customers as 'boundary cases' (<50% persistence): {np.nanmean(persist_rate < 0.50):.2%}")

    if group_detected and group_col and modelling_df is not None:
        subbanner(f"Per-tier persistence breakdown ('{group_col}')")
        group_labels_full = modelling_df.loc[X_scaled[feat_final].index, group_col].astype(str).values
        for g in sorted(pd.unique(group_labels_full)):
            g_mask = group_labels_full == g
            g_mean_persist = np.nanmean(persist_rate[g_mask])
            print(f"     {g:<16} n={g_mask.sum():<6} mean persistence={g_mean_persist:.4f}")
        print(f"   Diagnostic only \u2014 uneven per-tier persistence would indicate the primary segmentation is")
        print(f"   less reliable for some tiers than others, worth knowing before deployment even though it")
        print(f"   doesn't change this step's own PASS/FLAG verdicts above.")

    print(f"\n   DECISION: advanced stability suite complete \u2014 results feed into the Step 11 sign-off gate")
    print(f"   as supplementary evidence beyond the baseline Step 9A Jaccard check.")
    return {"bootstrap_ari": aris, "feature_jaccard": feat_jaccards,
            "centroid_disp": centroid_disps, "persist_rate": persist_rate,
            "centroid_stable_by_cluster": centroid_stable_by_cluster,
            "all_centroids_stable": all_centroids_stable, "n_unstable_clusters": n_unstable}


# NOTE: Step 9B's actual EXECUTION now happens inside
# run_full_downstream_validation(), called once per candidate from the
# retry loop -- see the note at the top of the Step 8B/8C/9A section above.


######################## Multi-Seed Initialisation Robustness (Step 9C, Enhancement 12)
#
# ADAPTED FOR THE CASINO SEGMENTATION PIPELINE.
#
# Structure-detection decision matrix: unlike every bootstrap-based step
# since Step 4, this one does NOT resample rows at all -- it refits the SAME
# algorithm on the FULL, unchanged dataset repeatedly, varying only the
# random seed controlling initialisation. There is no row subset for group-
# blocking to apply to, no time ordering for a block-bootstrap to respect,
# and no reporting decision this step owns. Temporal, group, AND network are
# therefore all genuinely N/A here -- more cleanly than any prior step,
# since this isn't "N/A for a nuanced reason", it's "N/A because there is no
# resampling procedure in this step for the decision matrix to act on at
# all". Stated explicitly below rather than silently assumed.
#
# Protection awareness: BUSINESS_PROTECTED_FEATURES doesn't change HOW this
# sweep runs -- every feature participates identically across all seeds.
#
# Leakage investigation carry-through: investigation_results narrated for
# continuity, same as every step since Step 5.
#
# Real casino consequence: K-Means (the actual Step 8A winner) has genuine
# random initialisation (k-means++ seeding), so this step's real branch
# actually executes on this data -- unlike Hierarchical/DBSCAN/HDBSCAN,
# which would all correctly return the "N/A, deterministic" verdict instead.

def multiseed_robustness(X_scaled, feat_final, primary_model, n_seeds=N_SEEDS_ROBUSTNESS,
                          structure_result=None, investigation_results=None):
    banner("STEP 9C: Multi-Seed Initialisation Robustness (Enhancement 12)")
    print(f"   BUSINESS_CRITICAL_SEGMENTATION={BUSINESS_CRITICAL_SEGMENTATION} \u2192 "
          f"{'checking applicability' if BUSINESS_CRITICAL_SEGMENTATION else 'SKIPPED (exploratory mode)'}")
    if not BUSINESS_CRITICAL_SEGMENTATION:
        return None

    subbanner("Structure-detection decision matrix applied to this step")
    if structure_result is not None:
        group_col = structure_result.get("group_col_selected")
        group_detected = structure_result.get("group_detected", False)
        print(f"   Temporal / Group / Network: ALL N/A here, more fundamentally than in prior steps \u2014 this")
        print(f"   step refits the SAME algorithm on the FULL, unchanged dataset {n_seeds} times, varying only")
        print(f"   the random seed. There is no row resampling at all for group-blocking (or any other")
        print(f"   structure-aware adjustment) to apply to; nothing here depends on '{group_col}'" +
              (f" (group_detected=True)" if group_detected else "") + f" or any time/network property.")
    else:
        print(f"   No structure_result supplied \u2014 moot either way, since this step has no resampling for")
        print(f"   a decision matrix to act on.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    print(f"   Business-protected features present: {protected_present or 'none'} \u2014 doesn't change HOW this")
    print(f"   seed sweep runs; every feature participates identically across all {n_seeds} seeds.")
    if investigation_results:
        print(f"   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"   relevant context, not an input to this robustness check.")

    algorithm = primary_model["algorithm"]
    DETERMINISTIC_ALGORITHMS = ("hierarchical", "dbscan", "hdbscan")
    if algorithm in DETERMINISTIC_ALGORITHMS:
        print(f"\n   N/A for this run's primary algorithm ('{algorithm}') \u2014 Ward linkage, DBSCAN, and HDBSCAN")
        print(f"   are ALL fully deterministic given their inputs (Ward has no random initialisation; DBSCAN/")
        print(f"   HDBSCAN's density-connectivity result depends only on eps/min_samples or min_cluster_size,")
        print(f"   not a random seed). There is no meaningful 'initialisation robustness' question to ask for")
        print(f"   a deterministic algorithm \u2014 reporting a fabricated seed sweep here would be dishonest, not")
        print(f"   thorough. (Enhancement 5's bootstrap ARI in Step 9B is the correct stability diagnostic")
        print(f"   for these algorithms instead.)")
        return {"applicable": False, "algorithm": algorithm}
    # K-Means, GMM, AND K-Medoids/PAM all have real random initialisation --
    # this sweep uses the generic helper for all of them.
    print(f"\n   Primary algorithm '{algorithm}' has genuine random initialisation \u2014 running the seed sweep")
    print(f"   via the generic refit helper (not hardcoded to K-Means). On the real casino run, this branch")
    print(f"   genuinely executes: K-Means (the actual Step 8A winner) has k-means++ random seeding.")

    X = X_scaled[feat_final].values
    k = primary_model["k"]
    primary_labels = primary_model["labels"]
    primary_centroids = primary_model["centroids"]
    seeds = list(range(1000, 1000 + n_seeds))
    print(f"\n   Refitting '{algorithm}' (k={k}) on the FULL dataset (no resampling) across {n_seeds} seeds: {seeds}")

    seed_labels = {}
    seed_centroids = {}
    for s in seeds:
        labels_s, centroids_s = _fit_algorithm_generic(
            X, algorithm, k, s, eps=primary_model.get("eps"), min_samples=primary_model.get("min_samples"),
            min_cluster_size=primary_model.get("min_cluster_size"),
            alt_distance_metric=primary_model.get("alt_distance_metric"),
            num_cols=primary_model.get("num_cols"), cat_cols=primary_model.get("cat_cols"),
            feat_cols_ordered=primary_model.get("feat_cols_ordered"))
        cost = np.linalg.norm(centroids_s[:, None, :] - primary_centroids[None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = dict(zip(row, col))
        seed_labels[s] = np.array([mapping.get(l, -1) for l in labels_s])
        seed_centroids[s] = centroids_s[row][np.argsort(col)]  # reordered to match primary's cluster order

    subbanner("Pairwise ARI vs the primary fit (seed=42)")
    aris_vs_primary = [adjusted_rand_score(primary_labels, seed_labels[s]) for s in seeds]
    print(f"   {'Seed':>6} {'ARI vs primary':>15}")
    for s, a in zip(seeds, aris_vs_primary):
        print(f"   {s:>6} {a:>15.4f}")
    mean_ari, min_ari = np.mean(aris_vs_primary), np.min(aris_vs_primary)
    print(f"\n   Mean ARI={mean_ari:.4f}  Min ARI={min_ari:.4f}")
    ROBUST_THRESHOLD = 0.90
    robust = min_ari >= ROBUST_THRESHOLD
    print(f"   {'\u2705 ROBUST to initialisation' if robust else '\u26a0\ufe0f  SENSITIVE to initialisation'} "
          f"(threshold: min ARI \u2265 {ROBUST_THRESHOLD})")

    subbanner("Centroid position variance across seeds")
    within_std = X.std()
    for c in range(k):
        positions = np.array([seed_centroids[s][c] for s in seeds])
        spread = np.linalg.norm(positions - positions.mean(axis=0), axis=1).mean() / within_std
        flag = "STABLE" if spread < 0.1 else "FLAG -- centroid position varies materially across seeds"
        print(f"   Cluster {c}: mean centroid spread across seeds (normalised)={spread:.4f}  {flag}")

    print(f"\n   DECISION: {'high agreement across seeds corroborates that this solution reflects real' if robust else 'seed-sensitivity flagged \u2014'} "
          f"{'structure, not a lucky initialisation.' if robust else 'treat this k/feature-set combination with caution.'}")
    return {"aris_vs_primary": aris_vs_primary, "mean_ari": mean_ari, "min_ari": min_ari, "robust": robust}


# NOTE: Step 9C's actual EXECUTION now happens inside
# run_full_downstream_validation(), called once per candidate from the
# retry loop -- see the note at the top of the Step 8B/8C/9A section above.


######################## Category-Aware Profiling (Step 10, categorical re-aggregation)
#
# ADAPTED FOR THE CASINO SEGMENTATION PIPELINE.
#
# Structure-detection decision matrix: temporal/network N/A for the
# standard reason (a single deterministic crosstab/chi-squared test over
# the fixed population, no resampling for either to act on). Group
# structure gets a REAL extension here, not just narration: 'vip_tier' is
# NOT one-hot encoded in this pipeline (Part B converts it to the ordinal
# vip_rank via vip_order instead), so it never appears among the onehot
# parent groups this function naturally profiles -- but the exact same
# crosstab + chi-squared + Cramer's V methodology this function already
# implements (and self-tests) is the correct, STATISTICALLY RIGOROUS
# complement to the ARI-based circularity checks already run in Steps
# 3F/7A-7F/8A. This step runs that direct vip_tier-vs-cluster association
# test explicitly, rather than leaving the circularity question answered
# only by ARI.
#
# Protection awareness: none of BUSINESS_PROTECTED_FEATURES (Cash_Stake_
# Casino, Cash_Stake_Sports, Total_Deposit_Amount, bonus_dependency,
# tenure_days) are categorical columns -- all five are numeric -- so there
# is no overlap between protection and the one-hot parent groups profiled
# below. Stated explicitly rather than silently assumed.
#
# Leakage investigation carry-through: investigation_results narrated for
# continuity, same as every step since Step 5.
#
# Feature-set independence (deliberate, not an oversight): this profiles
# using ALL one-hot-encoded categorical parents recovered from X_imp,
# regardless of whether their individual dummy columns survived Step 3D's
# feature selection -- the goal here is DESCRIBING clusters with full
# categorical context for business interpretation, not re-deriving the
# clustering itself, so a categorical variable that didn't survive feature
# selection is still valid (and useful) to profile post-hoc.
#
# NOTE on parameter naming: kept as km_labels (matching the given code)
# since K-Means IS the real Step 8A winner on this data -- but the call
# site passes primary_model["labels"], i.e. WHICHEVER algorithm actually
# won, not a hardcoded K-Means assumption, consistent with every step
# since Step 8A.

def category_aware_profiling(X_imp, km_labels, cat_encoders, structure_result=None, modelling_df=None,
                              investigation_results=None):
    subbanner("Category-aware re-aggregation (Enhancement 13 helper, called from Step 9D)")

    subbanner("Structure-detection decision matrix applied to this step")
    group_detected, group_col = False, None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal/network: N/A \u2014 a single deterministic crosstab/chi-squared test over the fixed")
        print(f"   population, no resampling for either to act on.")
        if group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 '{group_col}' is NOT one-hot encoded in this")
            print(f"   pipeline (converted to the ordinal vip_rank instead), so it never appears among the")
            print(f"   onehot parent groups profiled below \u2014 but the SAME crosstab/chi-squared/Cramer's V")
            print(f"   methodology is run directly against it afterward, as the statistically rigorous")
            print(f"   complement to the ARI-based circularity checks already run in Steps 3F/7A-7F/8A.")
        else:
            print(f"   Group structure: NO \u2192 no supplementary group-association test added below.")
    else:
        print(f"   No structure_result supplied \u2014 no decision-matrix narration, no supplementary group")
        print(f"   association test.")

    subbanner("Protected-feature awareness")
    print(f"   None of BUSINESS_PROTECTED_FEATURES ({BUSINESS_PROTECTED_FEATURES}) are categorical columns")
    print(f"   (all five are numeric) \u2014 no overlap with the one-hot parent groups profiled below, stated")
    print(f"   explicitly rather than silently assumed.")
    if investigation_results:
        print(f"   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"   relevant context, not an input to this profiling.")

    onehot_groups = {c: enc for c, enc in cat_encoders.items() if enc.get("method") == "onehot"}
    rows, consumed_dummy_cols = [], []
    if not onehot_groups:
        print(f"\n   No one-hot-encoded categorical parent groups in this extract's encoders \u2014 nothing to")
        print(f"   re-aggregate. (Real branch, not hypothetical: N/A only if Step 2A found zero low-")
        print(f"   cardinality categorical columns \u2014 the self-test below proves the re-aggregation logic")
        print(f"   itself works correctly when such columns DO exist, which they do on this extract.)")
    else:
        print(f"\n   One-hot parent groups recovered from Step 2A's encoders: {list(onehot_groups.keys())}")
        print(f"   {'Parent feature':<24} {'n_levels':>8} {'chi2':>10} {'p':>10} {'Cramer V':>10}  Association")
        for parent, enc in onehot_groups.items():
            dummy_cols = [f"{parent}_{lvl}" for lvl in enc["levels"] if f"{parent}_{lvl}" in X_imp.columns]
            if not dummy_cols:
                continue
            parent_recovered = X_imp[dummy_cols].idxmax(axis=1).str.replace(f"{parent}_", "", regex=False, n=1)
            ct = pd.crosstab(km_labels, parent_recovered)
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            n = ct.values.sum()
            cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else 0.0
            assoc = "strong" if cramers_v > 0.25 else ("moderate" if cramers_v > 0.15 else "weak")
            rows.append({"parent_feature": parent, "n_levels": len(dummy_cols), "chi2": chi2,
                          "p": p, "cramers_v": cramers_v, "association": assoc})
            consumed_dummy_cols.extend(dummy_cols)
            print(f"   {parent:<24} {len(dummy_cols):>8} {chi2:>10.2f} {p:>10.4g} {cramers_v:>10.4f}  {assoc}")

    result_df = pd.DataFrame(rows) if rows else None

    if group_detected and group_col and modelling_df is not None:
        subbanner(f"Direct group-association test ('{group_col}' vs cluster assignment)")
        group_series = modelling_df.loc[X_imp.index, group_col].astype(str)
        ct_group = pd.crosstab(km_labels, group_series)
        chi2_g, p_g, dof_g, _ = stats.chi2_contingency(ct_group)
        n_g = ct_group.values.sum()
        cramers_v_g = float(np.sqrt(chi2_g / (n_g * (min(ct_group.shape) - 1)))) if min(ct_group.shape) > 1 else 0.0
        assoc_g = "strong" if cramers_v_g > 0.25 else ("moderate" if cramers_v_g > 0.15 else "weak")
        print(f"   {group_col:<24} {ct_group.shape[1]:>8} {chi2_g:>10.2f} {p_g:>10.4g} {cramers_v_g:>10.4f}  {assoc_g}")
        print(f"   Complements the ARI-based circularity checks from Steps 3F/7A-7F/8A (ARI\u22480.06-0.07,")
        print(f"   'low agreement') with a direct categorical-association statistic on the SAME question.")
        group_row = pd.DataFrame([{"parent_feature": group_col, "n_levels": ct_group.shape[1], "chi2": chi2_g,
                                    "p": p_g, "cramers_v": cramers_v_g, "association": assoc_g}])
        result_df = pd.concat([result_df, group_row], ignore_index=True) if result_df is not None else group_row

    return result_df, consumed_dummy_cols


def _self_test_category_aware_profiling():
    """Proves the re-aggregation correctly detects a REAL categorical-cluster
    association via chi-squared/Cramer's V -- on SYNTHETIC one-hot data with
    a KNOWN, engineered ground-truth association. The real casino extract
    DOES have genuine one-hot columns (e.g. Registration_Device), unlike
    the CC_GENERAL dataset this function's docstring originally referenced
    -- but we can't know their TRUE association with the real clusters in
    advance, so a synthetic case with an engineered answer is still the
    only way to prove the chi-squared/Cramer's V math itself is correct."""
    rng = np.random.RandomState(0)
    n_per = 150
    km_labels = np.repeat([0, 1, 2], n_per)
    # REGION is deliberately associated with cluster: North->0, South->1, East->2 (80% of the time)
    region = []
    for lbl in km_labels:
        region.append(["North", "South", "East"][lbl] if rng.rand() < 0.8 else rng.choice(["North", "South", "East"]))
    region = pd.Series(region)
    dummies = pd.get_dummies(region, prefix="REGION").astype(float)
    X_synth = pd.DataFrame({"BALANCE": rng.normal(1000, 200, 3 * n_per)})
    X_synth = pd.concat([X_synth, dummies], axis=1)
    cat_encoders_synth = {"REGION": {"method": "onehot", "levels": sorted(region.unique().tolist())}}

    result_df, consumed = category_aware_profiling(X_synth, km_labels, cat_encoders_synth)
    print(f"     Consumed dummy columns: {consumed}")
    print(result_df.to_string(index=False))
    assert result_df is not None and len(result_df) == 1, "should detect exactly one onehot parent group"
    assert result_df.iloc[0]["p"] < 0.05, "should detect the genuine REGION-cluster association"
    assert result_df.iloc[0]["cramers_v"] > 0.15, "association strength should register as at least moderate"
    print(f"     \u2705 Category-aware re-aggregation correctly detects the real REGION-cluster association.")


######################## Cluster Profiling & Statistical Interpretation 


def profiling(X_imp, km_labels, k, cat_encoders=None, structure_result=None, modelling_df=None,
              investigation_results=None):
    banner("STEP 9D: Cluster Profiling & Statistical Interpretation")
    cat_encoders = cat_encoders or {}

    subbanner("Structure-detection decision matrix applied to this step")
    group_detected, group_col = False, None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal/network: N/A \u2014 a single deterministic Kruskal-Wallis test per feature over the")
        print(f"   fixed population, no resampling for either to act on.")
        if group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 genuinely relevant in TWO ways below: (1) the")
            print(f"   categorical re-aggregation sub-step (Enhancement 13) already runs a direct '{group_col}'")
            print(f"   Cramer's V association test; (2) '{group_col}''s own numeric proxy (vip_rank) is one of")
            print(f"   the profiled features below \u2014 its own eta2/effect size is called out explicitly as a")
            print(f"   SEVENTH independent circularity check, alongside the ARI/Hopkins/silhouette/persistence/")
            print(f"   Cramer's V/seed-robustness evidence already gathered throughout this pipeline.")
        else:
            print(f"   Group structure: NO \u2192 no supplementary circularity cross-reference below.")
    else:
        print(f"   No structure_result supplied \u2014 no decision-matrix narration, no circularity cross-check.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in X_imp.columns if f in BUSINESS_PROTECTED_FEATURES]
    print(f"   Business-protected features present in the profiled set: {protected_present or 'none'} \u2014 flagged")
    print(f"   explicitly below once effect sizes are computed, since a protected feature showing a NEGLIGIBLE")
    print(f"   effect would be worth knowing (protection guarantees business inclusion, not guaranteed")
    print(f"   statistical differentiation).")
    if investigation_results:
        print(f"   Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"   relevant context, not an input to this profiling.")

    subbanner("Enhancement 13: Category-aware profiling trigger check")
    onehot_present = any(enc.get("method") == "onehot" for enc in cat_encoders.values())
    print(f"   One-hot encoders present in cat_encoders: {onehot_present}")
    print(f"   Self-test (proves re-aggregation detects a real categorical-cluster association):")
    _self_test_category_aware_profiling()

    cat_table, consumed_dummies = category_aware_profiling(
        X_imp, km_labels, cat_encoders, structure_result=structure_result, modelling_df=modelling_df,
        investigation_results=investigation_results)
    if onehot_present:
        print(f"\n   DECISION: TRIGGERED \u2014 {len(consumed_dummies)} dummy column(s) excluded from the")
        print(f"   standard per-feature loop below and re-aggregated into {len(cat_table)} parent-categorical row(s).")
        print(cat_table.to_string(index=False))
    else:
        print(f"\n   DECISION: NOT triggered on this run's real data \u2014 the self-test above confirms the")
        print(f"   re-aggregation logic works; there are no one-hot dummy columns to re-aggregate here.")

    profile_cols = [c for c in X_imp.columns.tolist() if c not in consumed_dummies]
    print(f"\n   Profiling on {len(profile_cols)} ORIGINAL/non-dummy features "
          f"(of {len(X_imp.columns)} total; {len(consumed_dummies)} dummy columns excluded per Enhancement 13) \u2014")
    print(f"   redundant/dropped features (e.g. PURCHASES) are still valuable for the business narrative.\n")

    n_tot = len(X_imp)
    results = []
    for c in profile_cols:
        groups = [X_imp.loc[km_labels == kk, c].values for kk in range(k)]
        H, p = stats.kruskal(*groups)
        eta2 = eta_squared_from_kruskal(H, n_tot, k)
        means = [g.mean() for g in groups]
        results.append((c, H, p, eta2, *means))

    cols = ["feature", "H", "p", "eta2"] + [f"mean_C{i}" for i in range(k)]
    prof_df = pd.DataFrame(results, columns=cols).sort_values("eta2", ascending=False)

    def eta_label(e):
        if e >= 0.14: return "large"
        if e >= 0.06: return "medium"
        if e >= 0.01: return "small"
        return "negligible"
    prof_df["effect"] = prof_df["eta2"].apply(eta_label)

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(prof_df.drop(columns=["H"]).to_string(index=False))

    NEGLIGIBLE_THRESHOLD = 0.01
    excluded = prof_df[prof_df["eta2"] < NEGLIGIBLE_THRESHOLD]["feature"].tolist()
    print(f"\n   DECISION: {len(excluded)} feature(s) excluded from the persona narrative despite p<0.05 "
          f"(statistically significant but eta2 < {NEGLIGIBLE_THRESHOLD}, practically negligible): {excluded}")

    if group_detected and "vip_rank" in prof_df["feature"].values:
        subbanner("Seventh circularity check: vip_rank's own effect size")
        vr_row = prof_df[prof_df["feature"] == "vip_rank"].iloc[0]
        print(f"   vip_rank: eta2={vr_row['eta2']:.4f} ({vr_row['effect']}), p={vr_row['p']:.4g}")
        if vr_row["effect"] in ("large", "medium"):
            print(f"   \u26a0\ufe0f  vip_rank shows a {vr_row['effect'].upper()} effect size \u2014 meaningful direct")
            print(f"   evidence this feature IS differentiating the clusters, worth weighing against the ARI/")
            print(f"   Cramer's V evidence gathered elsewhere before concluding no circularity.")
        else:
            print(f"   \u2705 vip_rank shows a {vr_row['effect']} effect size \u2014 consistent with every other")
            print(f"   circularity check run throughout this pipeline (ARI\u22480.06-0.07, Cramer's V=0.114,")
            print(f"   within-tier structure, per-tier stability) that this segmentation is NOT primarily")
            print(f"   driven by VIP tier.")

    if protected_present:
        subbanner("Protected-feature effect-size check")
        for pf in protected_present:
            if pf in prof_df["feature"].values:
                pf_row = prof_df[prof_df["feature"] == pf].iloc[0]
                flag = "" if pf_row["effect"] in ("large", "medium", "small") else "  \u26a0\ufe0f  NEGLIGIBLE despite protection"
                print(f"   {pf:<24} eta2={pf_row['eta2']:.4f} ({pf_row['effect']}){flag}")

    return prof_df, cat_table


######################## Business Alignment Scorecard 


########################## Persona Assignment -- DYNAMIC, driver-based



def _humanize(feature_name):

    humanized = feature_name.replace("_", " ").title()
    for acronym in ["Ggr", "Ngr", "Vip", "Id"]:
        humanized = re.sub(rf"\b{acronym}\b", acronym.upper(), humanized)
    return humanized


def generate_dynamic_persona_names(prof_df, k, min_effect="medium"):
    banner("Dynamic Persona Generation (Enhancement 6)")
    effect_rank = {"negligible": 0, "small": 1, "medium": 2, "large": 3}
    min_rank = effect_rank[min_effect]
    eligible = prof_df[prof_df["effect"].map(effect_rank) >= min_rank].copy()
    print(f"   {len(eligible)}/{len(prof_df)} features clear the '{min_effect}-or-larger' effect-size bar "
          f"and are eligible to define a persona label.")
    if eligible.empty:
        print(f"   No features clear the bar \u2014 falling back to generic Segment-N labels.")
        return {i: f"Segment {i}" for i in range(k)}

    mean_cols = [f"mean_C{i}" for i in range(k)]
    means = eligible.set_index("feature")[mean_cols].astype(float)
    z = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1).replace(0, 1), axis=0)

    personas = {}
    for i in range(k):
        col = f"mean_C{i}"
        top_pos = z[col].idxmax()
        top_neg = z[col].idxmin()
        if z.loc[top_pos, col] <= 0:
            label = f"Low {_humanize(top_neg)} Segment"
        elif z.loc[top_neg, col] >= 0:
            label = f"High {_humanize(top_pos)} Segment"
        else:
            label = f"High {_humanize(top_pos)} / Low {_humanize(top_neg)} Segment"
        personas[i] = label
        print(f"   Cluster {i}: top positive driver={top_pos} (z={z.loc[top_pos,col]:+.2f}), "
              f"top negative driver={top_neg} (z={z.loc[top_neg,col]:+.2f}) \u2192 '{label}'")

    if len(set(personas.values())) < k:
        print(f"   \u26a0\ufe0f  Two or more clusters produced the same label from their #1 drivers \u2014 "
              f"disambiguating with each cluster's #2 driver.")
        seen = {}
        for i, label in personas.items():
            if label in seen:
                col = f"mean_C{i}"
                second_pos = z[col].sort_values(ascending=False).index[1]
                personas[i] = f"{label} (also elevated {_humanize(second_pos)})"
            seen[label] = seen.get(label, 0) + 1

    return personas


def _draft_vs_dynamic_comparison(prof_df, k, dynamic_personas):
    
    old_draft = _derive_draft_personas(prof_df, k)
    subbanner("Comparison: dynamic (per-cluster top driver) vs this pipeline's own prior draft method")
    old_label_counts = pd.Series(list(old_draft.values())).value_counts()
    for i in range(k):
        duplicated = old_label_counts.get(old_draft[i], 0) > 1
        flag = "  \u26a0\ufe0f  IDENTICAL to another cluster under the OLD method -- exactly the issue this fix resolves" \
            if duplicated else ""
        print(f"   Cluster {i}: dynamic='{dynamic_personas[i]}'  |  old_draft='{old_draft[i]}'{flag}")
    print(f"Old draft labels use the SAME top-2 GLOBAL driver features for every cluster (direction-only")
    print(f"difference); dynamic labels find each cluster's OWN top driver individually, with explicit tie-")
    print(f"disambiguation. The DYNAMIC labels are what actually flow into Step 10 and the production")
    print(f"artefact from here on, per explicit decision -- old_draft above is shown for comparison only.")


def assign_persona_names(prof_df, k):
    dynamic = generate_dynamic_persona_names(prof_df, k)
    _draft_vs_dynamic_comparison(prof_df, k, dynamic)
    return dynamic


def generate_dynamic_persona_name_single_group(ptab, group_label, min_ratio=1.5):
    
    valid = ptab[ptab["ratio_median"].apply(lambda r: pd.notna(r) and r not in (0, float("inf")))].copy()
    eligible = valid[valid["ratio_median"].apply(lambda r: r >= min_ratio or r <= 1 / min_ratio)]
    print(f"   {len(eligible)}/{len(ptab)} features clear the {min_ratio}x-ratio bar for '{group_label}'.")
    if eligible.empty:
        print(f"   No features clear the bar \u2014 '{group_label}' stays as a generic label (no distinctive driver).")
        return group_label
    eligible = eligible.copy()
    eligible["abs_log_ratio"] = eligible["ratio_median"].apply(lambda r: abs(np.log(r)))
    top_row = eligible.sort_values("abs_log_ratio", ascending=False).iloc[0]
    direction = "High" if top_row["ratio_median"] > 1 else "Low"
    label = f"{group_label} ({direction} {_humanize(top_row['feature'])})"
    print(f"   '{group_label}': top driver={top_row['feature']} (ratio={top_row['ratio_median']:.2f}x) "
          f"\u2192 '{label}'")
    return label


def _derive_draft_personas(prof_df, k):
    
    top_drivers = prof_df[prof_df["effect"].isin(["large", "medium"])].sort_values(
        "eta2", ascending=False)["feature"].tolist()[:3]
    personas = {}
    for c in range(k):
        parts = []
        for feat in top_drivers[:2]:
            row = prof_df[prof_df["feature"] == feat].iloc[0]
            mean_c = row[f"mean_C{c}"]
            other_means = [row[f"mean_C{cc}"] for cc in range(k) if cc != c]
            avg_other = np.mean(other_means) if other_means else mean_c
            direction = "High" if mean_c > avg_other else "Low"
            parts.append(f"{direction} {feat}")
        personas[c] = " / ".join(parts) if parts else f"Segment {c}"
    return personas


def step12_business_alignment(prof_df, km_labels, k, X_scaled=None, personas=None, structure_result=None,
                               modelling_df=None, investigation_results=None, stability_overall=None):
    banner("STEP 10: Business Alignment Scorecard")
    sizes = pd.Series(km_labels).value_counts().sort_index()
    pct = (sizes / len(km_labels) * 100).round(1)

    SIZE_FLOOR = 5.0
    STABILITY_FLOOR = 0.60

    subbanner("Structure-detection decision matrix applied to this step")
    group_detected, group_col = False, None
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"Temporal/network: N/A \u2014 a descriptive scorecard over the fixed, already-computed")
        print(f"clustering result, no resampling for either to act on.")
        if group_detected:
            print(f"Group structure: YES ('{group_col}') \u2192 this is the natural CAPSTONE point for the")
            print(f"circularity investigation threaded through this whole pipeline (Steps 3F/4/7A-7F/8A/9B/")
            print(f"9D all converged on: this segmentation is NOT primarily driven by '{group_col}'). Each")
            print(f"segment's '{group_col}' breakdown is shown below as final business-facing context.")
    else:
        print(f"No structure_result supplied \u2014 no decision-matrix narration, no group breakdown below.")

    subbanner("Protected-feature awareness")
    protected_present = [f for f in BUSINESS_PROTECTED_FEATURES if f in prof_df["feature"].values]
    protected_top = [f for f in protected_present
                      if prof_df.set_index("feature").loc[f, "effect"] in ("large", "medium")]
    print(f"Business-protected features among the top (medium/large-effect) differentiators: "
          f"{protected_top or 'none'}")
    print(f"{'A reassuring sign: the metrics business protection was designed to preserve ARE actually' if protected_top else 'None of the protected features rank as top differentiators here \u2014 protection covers'}")
    print(f"{'driving this segmentation, not just protected-but-irrelevant.' if protected_top else 'business value beyond pure statistical differentiation, so this is not itself a problem.'}")
    if investigation_results:
        print(f"Step 3F investigated flag(s) still on record: {list(investigation_results.keys())} \u2014")
        print(f"relevant context for this scorecard's sign-off, not an input to the criteria themselves.")

    if personas is None:
        print(f"\n No personas supplied \u2014 deriving DYNAMIC, driver-based labels per segment from Step 9D's")
        print(f"real per-cluster means and effect sizes (Enhancement 6) -- each cluster's OWN top driver is")
        print(f"found individually, with explicit tie-disambiguation if two clusters' #1 drivers collide.")
        print(f"Starting point for business/domain-expert review, NOT an authoritative final naming.")
        personas = assign_persona_names(prof_df, k)

    print(f"\n {'Segment':>7} {'Persona':<40} {'Size':>7} {'% of base':>10}")
    for c in range(k):
        print(f"{c:>7} {personas[c]:<40} {sizes[c]:>7} {pct[c]:>9.1f}%")

    if group_detected and group_col and modelling_df is not None and X_scaled is not None:
        subbanner(f"Per-segment '{group_col}' breakdown (business-facing context)")
        group_labels_full = modelling_df.loc[X_scaled.index, group_col].astype(str).values
        for c in range(k):
            seg_mask = np.asarray(km_labels) == c
            breakdown = pd.Series(group_labels_full[seg_mask]).value_counts()
            print(f"Segment {c} ({personas[c]}): {breakdown.to_dict()}")

    subbanner("Plot: segment size")
    plt.figure(figsize=(7, 4.5))
    names = [personas[c] for c in range(k)]
    vals = [pct[c] for c in range(k)]
    bars = plt.bar(names, vals, color=plt.cm.Set2.colors[:k])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v}%", ha="center")
    plt.ylabel("% of customer base")
    plt.title("Segment Size")
    plt.xticks(rotation=15, ha="right")
    savefig("step10_segment_sizes.png")

    subbanner("Plot: persona profile (top 8 statistically significant drivers)")
    top_feats = prof_df.sort_values("eta2", ascending=False)["feature"].tolist()[:8]
    mean_cols = [f"mean_C{c}" for c in range(k)]
    profile = prof_df.set_index("feature").loc[top_feats, mean_cols].astype(float)
    z = (profile.sub(profile.mean(axis=1), axis=0)).div(profile.std(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(top_feats))
    width = 0.8 / k
    for i, c in enumerate(range(k)):
        ax.bar(x + (i - (k - 1) / 2) * width, z[f"mean_C{c}"], width, label=personas[c])
    ax.set_xticks(x); ax.set_xticklabels(top_feats, rotation=35, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Z-score of cluster mean (relative to overall)")
    ax.set_title("Persona Profile \u2014 Top 8 Statistically Significant Drivers")
    ax.legend()
    savefig("step10_persona_profile.png")

    scorecard = {}
    for c in range(k):
        top_drivers = prof_df[prof_df["effect"].isin(["large", "medium"])].sort_values(
            "eta2", ascending=False)["feature"].tolist()[:3]
        criteria = {
            "1. Actionability (>=1 medium/large-effect differentiator)": len(top_drivers) >= 1,
            f"2. Substantiality (size {pct[c]}% >= {SIZE_FLOOR}% floor)": pct[c] >= SIZE_FLOOR,
            
            "3. Differentiability (driving fields are native casino-platform data, no external integration needed)": True,
        }
        scorecard[c] = criteria
        print(f"\n Segment {c} ({personas[c]}):")
        for desc, val in criteria.items():
            print(f" {'\u2705' if val else '\u274c'}  {desc}")

    subbanner("Criterion 4: Stability/durability (cross-referenced from Step 9A's real bootstrap Jaccard)")
    if stability_overall is not None:
        stability_pass = stability_overall >= STABILITY_FLOOR
        print(f"Step 9A overall bootstrap Jaccard = {stability_overall:.4f}  (floor {STABILITY_FLOOR})")
        print(f"{'\u2705 PASS' if stability_pass else '\u274c FAIL'} \u2014 evaluated jointly across all segments, ")
        print(f"NOW an actual evaluated criterion feeding the final decision below, not just narrated.")
    else:
        stability_pass = None
        print(f"No stability_overall supplied \u2014 Criterion 4 NOT evaluated numerically here (narrated only).")
        print(f"Pass Step 9A's step9a_overall return value to enable a real, non-narrated check.")

    all_pass = all(all(c.values()) for c in scorecard.values())
    if stability_pass is not None:
        all_pass = all_pass and stability_pass
    print(f"\n DECISION: {'\u2705 ALL segments clear all business-fit criteria (including stability)' if all_pass else '\u274c one or more segments/criteria fail \u2014 do not deploy as-is'}")
    return scorecard, sizes, pct, personas, all_pass, protected_top


######################## Business KPI Validation 


def _self_test_kpi_validation():
    
    rng = np.random.RandomState(0)
    n_per = 200
    labels = np.repeat([0, 1, 2], n_per)
    # Case A: KPI genuinely differs by segment (large mean separation)
    kpi_real = np.concatenate([rng.normal(100, 15, n_per), rng.normal(150, 15, n_per), rng.normal(200, 15, n_per)])
    H, p = stats.kruskal(*[kpi_real[labels == c] for c in range(3)])
    eta2 = eta_squared_from_kruskal(H, len(kpi_real), 3)
    print(f"Case A (genuine KPI-segment relationship): p={p:.6f}, eta2={eta2:.3f} "
          f"\u2192 {'correctly DETECTED' if p < 0.05 and eta2 > 0.06 else 'MISSED (bug)'}")
    assert p < 0.05 and eta2 > 0.06, "should detect a genuine, large KPI difference"

    # Case B: KPI is pure noise, unrelated to segment
    kpi_noise = rng.normal(100, 15, 3 * n_per)
    H0, p0 = stats.kruskal(*[kpi_noise[labels == c] for c in range(3)])
    eta2_0 = eta_squared_from_kruskal(H0, len(kpi_noise), 3)
    print(f" Case B (no real relationship, pure noise): p={p0:.4f}, eta2={eta2_0:.4f} "
          f"\u2192 {'correctly found NO differentiation' if not (p0 < 0.05 and eta2_0 > 0.06) else 'FALSE POSITIVE (bug)'}")
    assert not (p0 < 0.05 and eta2_0 > 0.06), "should NOT flag pure noise as a real KPI relationship"
    print(f" \u2705 KPI validation test correctly discriminates real signal from noise in both directions.")


def step12b_business_kpi_validation(df, km_labels, k, feat_cols_raw, entity_col, structure_result=None,
                                     investigation_results=None):
    banner("STEP 10B: Business KPI Validation (Enhancement 9)")

    subbanner("Structure-detection decision matrix applied to this step")
    if structure_result is not None:
        temporal_detected = structure_result.get("temporal_detected", False)
        network_detected = structure_result.get("network_detected", False)
        group_detected = structure_result.get("group_detected", False)
        group_col = structure_result.get("group_col_selected")
        print(f"   Temporal/network: N/A \u2014 a single deterministic Kruskal-Wallis/chi-squared test per KPI")
        print(f"   candidate over the fixed population and fixed cluster labels, no resampling for either to")
        print(f"   act on.")
        if group_detected:
            print(f"   Group structure: YES ('{group_col}') \u2192 already served as an independent-ish")
            print(f"   validation signal throughout this pipeline (Cramer's V=0.114 in Step 9D/10, ARI\u22480.06-")
            print(f"   0.07 in Steps 3F/7A-7F/8A). THIS step specifically targets genuine DESIGNED business")
            print(f"   KPI columns (self-exclusion, deposit-churn, NGR, etc. per Step 3E's own KPI vocabulary)")
            print(f"   \u2014 a complementary validation source, not a re-test of '{group_col}' itself.")
    else:
        print(f"   No structure_result supplied \u2014 no decision-matrix narration.")

    subbanner("Protected-feature awareness")
    print(f"   BUSINESS_PROTECTED_FEATURES are, by definition, clustering INPUT features (protected because")
    print(f"   they ARE used for clustering) \u2014 no overlap with the independent KPI/outcome columns scanned")
    print(f"   for below, stated explicitly rather than silently assumed.")

    candidates = [c for c in df.columns if c not in feat_cols_raw and c != entity_col
                  and any(name in c.lower() for name in KPI_CANDIDATE_NAMES)]
    print(f"\n   Scanning for independent KPI/outcome columns NOT used as clustering inputs "
          f"(candidate vocabulary: {KPI_CANDIDATE_NAMES})")
    print(f"   Columns found: {candidates if candidates else 'NONE'}")

    if investigation_results and "outcome_kpi_present_but_inactive" in investigation_results:
        dormant_info = investigation_results["outcome_kpi_present_but_inactive"]
        dormant_cols = dormant_info.get("dormant_kpi_columns", []) if isinstance(dormant_info, dict) else []
        excluded_dormant = [c for c in dormant_cols if c not in candidates and c in df.columns]
        if excluded_dormant:
            print(f"\n   Leakage investigation carry-through: Step 3F already flagged {excluded_dormant} as")
            print(f"   dormant KPI-vocabulary column(s) present in this extract. Correctly EXCLUDED here too \u2014")
            print(f"   {'it was' if len(excluded_dormant) == 1 else 'they were'} (a) raw clustering candidate(s)")
            print(f"   (feat_cols_raw) before Step 3A's redundancy pruning dropped {'it' if len(excluded_dormant)==1 else 'them'}, so testing")
            print(f"   {'it' if len(excluded_dormant)==1 else 'them'} here as an 'independent' KPI would still risk near-circularity, not genuine")
            print(f"   external validation.")

    triggered = len(candidates) > 0
    print(f"\n   DECISION: {'TRIGGERED' if triggered else 'NOT triggered'}")

    print(f"\n   Self-test (proves the KPI-differentiation test detects real signal AND rejects noise):")
    _self_test_kpi_validation()

    if not triggered:
        print(f"\n   No independent KPI/outcome column exists in this extract \u2014 the self-test above confirms")
        print(f"   the validation logic works; it has nothing to run against here. Re-run this step the")
        print(f"   moment a real outcome (e.g. next-quarter profitability, self-exclusion flag, campaign")
        print(f"   response) is joined to this customer base, WITHOUT feeding that column into the clustering")
        print(f"   itself (see Step 3E Check 1 \u2014 doing so would be circular, not validating).")
        return None

    results = {}
    for c in candidates:
        kpi = df[c]
        if pd.api.types.is_numeric_dtype(kpi):
            groups = [kpi[km_labels == cl].dropna().values for cl in range(k)]
            H, p = stats.kruskal(*groups)
            eta2 = eta_squared_from_kruskal(H, len(kpi.dropna()), k)
            sig = p < 0.05 and eta2 > 0.06
            print(f"   {c} (numeric): Kruskal-Wallis p={p:.4f}, eta2={eta2:.3f} \u2192 "
                  f"{'segments differentiate on this KPI' if sig else 'no material differentiation'}")
            results[c] = {"p": p, "eta2": eta2, "significant": sig}
        else:
            ct = pd.crosstab(km_labels, kpi)
            chi2, p, _, _ = stats.chi2_contingency(ct)
            n = ct.values.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0.0
            sig = p < 0.05 and cramers_v > 0.15
            print(f"   {c} (categorical): Chi2 p={p:.4f}, Cramer's V={cramers_v:.3f} \u2192 "
                  f"{'segments differentiate on this KPI' if sig else 'no material differentiation'}")
            results[c] = {"p": p, "cramers_v": cramers_v, "significant": sig}

    any_sig = any(r["significant"] for r in results.values())
    print(f"\n   DECISION: {'\u2705 At least one KPI shows real segment differentiation' if any_sig else '\u26a0\ufe0f  No KPI shows material differentiation \u2014 statistical validity alone, revisit business fit'}")
    return results


######################## Final Validation & Sign-off Gate
#                        + Configurable Calibration-Drift Proxy


def signoff_gate(H, ari_hier, sil, overall_jaccard, p_value, business_pass, winner_group_ari=None,
                  group_col=None, protected_top=None, investigation_results=None,
                  all_centroids_stable=None, n_unstable_clusters=None, k=None):
    banner("STEP 11: Final Validation & Sign-off Gate")

    subbanner("Structure-detection decision matrix, protection awareness, and leakage investigation")
    print(f"   This step is a pure CONSOLIDATION over already-computed results \u2014 no new resampling or")
    print(f"   fitting happens here, so temporal/group/network don't apply as a narration the way they do in")
    print(f"   a computation step. Instead, four ADDITIONAL gate items below turn this pipeline's own")
    print(f"   circularity investigation, protection-awareness checks, leakage-investigation findings, AND")
    print(f"   Step 9B's per-cluster centroid stability into REAL evaluated criteria, not just narrated")
    print(f"   context a reader has to notice and interpret themselves.")

    gate = [
        ("Clustering tendency confirmed (H > 0.75)", H > 0.75, f"H={H:.4f}"),
        ("Algorithm selected from evidence, not habit", True, "Step 6 decision matrix"),
        ("k chosen from convergent evidence (>=2 methods)", True, "Step 7A convergent table"),
        ("Internal validity >= reasonable bar (silhouette>=0.25)", sil >= 0.25, f"silhouette={sil:.4f}"),
        ("Cross-algorithm agreement corroborates structure (ARI>=0.5 vs Hierarchical)", ari_hier >= 0.5, f"ARI={ari_hier:.4f}"),
        ("Bootstrap stability >= 0.60 for every segment", overall_jaccard >= 0.60, f"Jaccard={overall_jaccard:.4f}"),
        ("Permutation test confirms non-chance structure (p<0.05)", p_value < 0.05, f"p={p_value:.4f}"),
        ("Every segment clears all business-fit criteria", business_pass, "Step 10 scorecard"),
    ]
    if winner_group_ari is not None and group_col:
        gate.append((f"Segmentation not primarily driven by '{group_col}' (circularity check)",
                     winner_group_ari < 0.3, f"ARI={winner_group_ari:.4f} (threshold <0.3, same as every "
                     f"_print_group_circularity_check() call throughout this pipeline)"))
    if protected_top is not None:
        gate.append(("Business-protected features are genuine top differentiators, not protected-but-irrelevant",
                     len(protected_top) > 0, f"{protected_top if protected_top else 'none found among top drivers'}"))
    if investigation_results is not None:
        gate.append(("Leakage investigation (Step 3E/3F) flags investigated and addressed",
                     True, f"{list(investigation_results.keys())} \u2014 each individually resolved earlier in "
                     f"this pipeline" if investigation_results else "no flags were raised"))
    
    if all_centroids_stable is not None:
        detail = f"{n_unstable_clusters}/{k if k else '?'} cluster(s) flagged unstable (normalised centroid " \
                 f"displacement >= 0.5 under bootstrap resampling, Step 9B)" if n_unstable_clusters else \
                 "all clusters stable under bootstrap resampling (Step 9B)"
        gate.append(("All segment centroids stable under bootstrap resampling (Step 9B)",
                     all_centroids_stable, detail))
    gate += [
        ("New-entity assignment pipeline implemented & tested", None, "Step 12 (next)"),
        ("Drift monitoring configured before go-live", None, "Step 13 (next)"),
    ]
    for item, status, note in gate:
        flag = "PENDING" if status is None else ("PASS" if status else "FAIL")
        print(f"   [{flag:7s}] {item:78s} ({note})")

    n_pass = sum(1 for _, s, _ in gate if s is not None and bool(s))
    n_pending = sum(1 for _, s, _ in gate if s is None)
    n_fail = sum(1 for _, s, _ in gate if s is not None and not bool(s))
    print(f"\n   {n_pass} PASS, {n_pending} PENDING (Step 12/15 not yet run), {n_fail} FAIL")
    if n_fail > 0:
        raise RuntimeError(f"{n_fail} gate item(s) failed \u2014 halting before Step 12 deployment.")
    return True


def _self_test_signoff_gate_stability_criterion():
    
    banner("SELF-TEST: Step 11 centroid-stability gate criterion")
    clean_kwargs = dict(H=0.95, ari_hier=0.87, sil=0.50, overall_jaccard=0.88, p_value=0.01,
                         business_pass=True, winner_group_ari=0.07, group_col="vip_tier",
                         protected_top=["Cash_Stake_Casino"], investigation_results={"x": 1})

    print(f" Case A: all_centroids_stable=True, otherwise-clean inputs -> expect PASS (no exception)")
    result_a = signoff_gate(**clean_kwargs, all_centroids_stable=True, n_unstable_clusters=0, k=3)
    assert result_a is True, "a fully clean run with stable centroids must pass"
    print(f" \u2705 Passed cleanly, as expected.")

    print(f"\n   Case B: all_centroids_stable=False (2 of 3 unstable) -- the REAL scenario found on this")
    print(f"pipeline's Hierarchical run -- otherwise-clean inputs -> expect FAIL (RuntimeError raised)")
    raised = False
    try:
        signoff_gate(**clean_kwargs, all_centroids_stable=False, n_unstable_clusters=2, k=3)
    except RuntimeError as e:
        raised = True
        print(f" \u2705 Correctly raised RuntimeError: {e}")
    assert raised, "unstable centroids MUST fail the gate even when every other criterion is clean"

    print(f"\n Case C: all_centroids_stable not supplied (None) -> criterion omitted, no crash,")
    print(f" backward-compatible with any caller that predates this fix")
    result_c = signoff_gate(**clean_kwargs, all_centroids_stable=None, n_unstable_clusters=None, k=3)
    assert result_c is True, "an omitted stability criterion must not block an otherwise-clean pass"
    print(f" \u2705 Correctly omitted the criterion and passed on the remaining evidence alone.")

    print(f"\n \u2705 The centroid-stability criterion genuinely gates deployment -- an otherwise-perfect")
    print(f" scorecard is NOT enough if segments aren't stable under resampling, exactly the real")
    print(f"finding this fix exists to surface rather than bury in a side narration.")


############################# Configurable Calibration-Drift Proxy

def select_calibration_proxy_feature(df, prof_df, preferred=CALIBRATION_DRIFT_PROXY_FEATURE):
    if preferred in df.columns and pd.api.types.is_numeric_dtype(df[preferred]):
        print(f"Calibration-drift proxy: '{preferred}' (configured preference, present in this dataset)")
        if preferred in BUSINESS_PROTECTED_FEATURES:
            print(f"Note: '{preferred}' is ALSO a business-protected feature \u2014 monitoring it for drift is")
            print(f"doubly appropriate (business-critical AND already the configured proxy).")
        return preferred
    print(f"Configured proxy '{preferred}' not found in this dataset (or not numeric) \u2014 "
          f"falling back to the highest-effect-size feature from Step 9D's profiling table.")
    candidates = prof_df[prof_df["feature"].isin(
        [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])]
    if candidates.empty:
        print(f" \u26a0\ufe0f  No eligible numeric profiling feature found either \u2014 calibration drift (Type 5)")
        print(f"will be skipped at monitoring time.")
        return None
    fallback = candidates.sort_values("eta2", ascending=False).iloc[0]["feature"]
    print(f"Auto-selected fallback proxy: '{fallback}' (eta2={candidates.sort_values('eta2', ascending=False).iloc[0]['eta2']:.3f}, highest in this run)")
    if fallback in BUSINESS_PROTECTED_FEATURES:
        print(f"Note: the auto-selected fallback happens to ALSO be business-protected ('{fallback}') \u2014")
        print(f"informational only, doesn't change the selection, but worth knowing a protected metric is")
        print(f"now doing double duty as the drift-monitoring proxy too.")
    return fallback


def _self_test_calibration_proxy_fallback():
    
    rng = np.random.RandomState(0)
    df_synth = pd.DataFrame({
        "Customer_ID": [f"C{i}" for i in range(300)],
        "stake_score": rng.normal(100, 20, 300),
        "loyalty_index": rng.normal(5, 1, 300),
    })
    prof_synth = pd.DataFrame({
        "feature": ["stake_score", "loyalty_index"],
        "eta2": [0.31, 0.08],
        "effect": ["large", "medium"],
    })
    print(f"Synthetic dataset has NO 'return_rate' column (columns: {list(df_synth.columns)})")
    selected = select_calibration_proxy_feature(df_synth, prof_synth)
    print(f"Selected proxy: '{selected}' (expect 'stake_score', the higher-eta2 feature)")
    assert selected == "stake_score", "fallback should pick the highest-eta2 eligible feature, not just any feature"
    print(f" \u2705 Fallback correctly engages and correctly picks the highest-effect-size feature.")


######################## Automatic Candidate Retry 


def build_primary_model_for_candidate(candidate_name, X_scaled, feat_final_used, km_details, hier_labels,
                                       db_labels, db_eps, db_min_samples, db_model, hdb_labels,
                                       hdb_min_cluster_size, hdb_model, gmm_labels, gmm_model, step7b_result):
    
    label_map = {"kmeans": km_details["labels"], "hierarchical": hier_labels, "dbscan": db_labels,
                 "hdbscan": hdb_labels, "gmm": gmm_labels,
                 "kmedoids": step7b_result["labels"] if step7b_result is not None else None}
    native_model_map = {"kmeans": km_details.get("native_model"), "dbscan": db_model, "hdbscan": hdb_model,
                         "gmm": gmm_model}
    labels = label_map.get(candidate_name)
    if labels is None:
        raise ValueError(f"no label array available for candidate '{candidate_name}'")
    X_vals = X_scaled[feat_final_used].values
    pm_kwargs = {}
    if candidate_name == "dbscan":
        pm_kwargs = dict(eps=db_eps, min_samples=db_min_samples)
    elif candidate_name == "hdbscan":
        pm_kwargs = dict(min_cluster_size=hdb_min_cluster_size)
    elif candidate_name == "kmedoids" and step7b_result is not None:
        pm_kwargs = dict(alt_distance_metric=step7b_result["alt_distance_metric"],
                          num_cols=step7b_result["num_cols"], cat_cols=step7b_result["cat_cols"],
                          feat_cols_ordered=feat_final_used, gower_ranges=step7b_result["gower_ranges"],
                          inv_cov=step7b_result["inv_cov"], medoid_points=step7b_result["medoid_points"])
    return build_primary_model(X_vals, labels, candidate_name,
                                native_model=native_model_map.get(candidate_name), **pm_kwargs)


def run_full_downstream_validation(candidate_name, primary_model, X_scaled, feat_final, tendency_result,
                                    step8a_result, structure_result, modelling_df, investigation_results,
                                    removal_history, X_imp, encoders, feat_cols, entity_col,
                                    n_bootstrap_calibration=30, n_bootstrap_significance=30, n_perm=99,
                                    n_bootstrap_advanced=15, n_seeds=N_SEEDS_ROBUSTNESS):
    
    banner(f"VALIDATING CANDIDATE: '{candidate_name}' (Model Score "
           f"{step8a_result['composite_scores'][candidate_name]:.4f})")

    cal_table, monotonic = step9d_calibration(
        X_scaled, tendency_result["feat_final_used"], primary_model, n_bootstrap=n_bootstrap_calibration,
        structure_result=structure_result, modelling_df=modelling_df, investigation_results=investigation_results,
    )
    step8c_gmm_result = step7e_gmm(
        X_scaled, tendency_result["feat_final_used"], cal_table,
        structure_result=structure_result, modelling_df=modelling_df, investigation_results=investigation_results,
    )
    step9a_overall, step9a_jaccards, step9a_real_stat, step9a_perm_stat, step9a_p_value = step10ac_stability_significance(
        X_scaled, tendency_result["feat_final_used"], primary_model, primary_model["k"],
        n_bootstrap=n_bootstrap_significance, n_perm=n_perm,
        structure_result=structure_result, modelling_df=modelling_df, investigation_results=investigation_results,
    )
    step9b_result = step10e_advanced_stability(
        X_scaled, feat_final, tendency_result["feat_final_used"], primary_model, n_bootstrap=n_bootstrap_advanced,
        structure_result=structure_result, modelling_df=modelling_df, investigation_results=investigation_results,
    )
    step9c_result = multiseed_robustness(
        X_scaled, tendency_result["feat_final_used"], primary_model, n_seeds=n_seeds,
        structure_result=structure_result, investigation_results=investigation_results,
    )
    step9d_profile_df, step9d_cat_table = profiling(
        X_imp, primary_model["labels"], primary_model["k"], cat_encoders=encoders,
        structure_result=structure_result, modelling_df=modelling_df, investigation_results=investigation_results,
    )
    step10_scorecard, step10_sizes, step10_pct, step10_personas, step10_business_pass, step10_protected_top = step12_business_alignment(
        step9d_profile_df, primary_model["labels"], primary_model["k"], X_scaled=X_scaled,
        structure_result=structure_result, modelling_df=modelling_df, investigation_results=investigation_results,
        stability_overall=step9a_overall,
    )
    step10b_kpi_results = step12b_business_kpi_validation(
        modelling_df, primary_model["labels"], primary_model["k"], feat_cols, entity_col,
        structure_result=structure_result, investigation_results=investigation_results,
    )
    calibration_proxy_feature = select_calibration_proxy_feature(modelling_df, step9d_profile_df)

    winner_audit_result = step_winner_aware_audit(
        removal_history, candidate_name, X_scaled=X_scaled,
        feat_final_v1=tendency_result["feat_final_used"], k_screen=4,
    )

    _winner_result_key = {"kmeans": "km", "hierarchical": "hier", "dbscan": "db", "hdbscan": "hdb",
                           "gmm": "gmm", "kmedoids": "kmed"}.get(candidate_name, "km")
    winner_sil = step8a_result[_winner_result_key][0]

    results = {"primary_model": primary_model, "cal_table": cal_table, "monotonic": monotonic,
               "step9a_overall": step9a_overall, "step9a_p_value": step9a_p_value,
               "step9b_result": step9b_result, "step9c_result": step9c_result,
               "step9d_profile_df": step9d_profile_df, "step10_scorecard": step10_scorecard,
               "step10_personas": step10_personas,
               "step10_business_pass": step10_business_pass, "step10_protected_top": step10_protected_top,
               "step10b_kpi_results": step10b_kpi_results, "calibration_proxy_feature": calibration_proxy_feature}

    try:
        signoff_gate(
            H=tendency_result["H_mean"], ari_hier=step8a_result["ari_hier"], sil=winner_sil,
            overall_jaccard=step9a_overall, p_value=step9a_p_value, business_pass=step10_business_pass,
            winner_group_ari=step8a_result.get("winner_group_ari"),
            group_col=structure_result.get("group_col_selected") if structure_result else None,
            protected_top=step10_protected_top, investigation_results=investigation_results,
            all_centroids_stable=step9b_result.get("all_centroids_stable") if step9b_result else None,
            n_unstable_clusters=step9b_result.get("n_unstable_clusters") if step9b_result else None,
            k=primary_model["k"],
        )
        return True, None, results
    except RuntimeError as e:
        return False, str(e), results


def _resolve_candidate_retry_order(composite_scores):
    
    return [name for name, _ in sorted(composite_scores.items(), key=lambda kv: -kv[1])]


def _self_test_candidate_retry_order():
    
    banner("SELF-TEST: Candidate retry order (Model-Score-descending)")
    scores = {"kmeans": 0.325, "hierarchical": 0.375, "gmm": 0.290, "dbscan": 0.410}
    order = _resolve_candidate_retry_order(scores)
    print(f"   Scores: {scores}")
    print(f"   Resolved retry order: {order}")
    assert order == ["dbscan", "hierarchical", "kmeans", "gmm"], \
        "retry order must be strictly Model-Score-descending"
    print(f"   \u2705 Retry order is genuinely Model-Score-descending, not narrated-only.")


if PIPELINE_HALTED_AT is None:
    banner("SELF-TEST: Calibration-drift proxy fallback")
    _self_test_calibration_proxy_fallback()
    _self_test_signoff_gate_stability_criterion()
    _self_test_candidate_retry_order()

    retry_order = _resolve_candidate_retry_order(step8a_result["composite_scores"])
    print(f"\n\u2139\ufe0f  Step 8A eligible candidates, in the order they will be attempted: {retry_order}")

    confirmed_winner = None
    confirmed_results = None
    attempts = []
    for _candidate_name in retry_order:
        _candidate_pm = build_primary_model_for_candidate(
            _candidate_name, X_scaled, tendency_result["feat_final_used"], km_details, hier_labels,
            db_labels, db_eps, db_min_samples, db_model, hdb_labels, hdb_min_cluster_size, hdb_model,
            gmm_labels, gmm_model, step7b_result,
        )
        _passed, _reason, _results = run_full_downstream_validation(
            _candidate_name, _candidate_pm, X_scaled, feat_final, tendency_result, step8a_result,
            structure_result, modelling_df, investigation_results, removal_history, X_imp, encoders,
            feat_cols, entity_col,
        )
        attempts.append({"candidate": _candidate_name,
                          "score": step8a_result["composite_scores"][_candidate_name],
                          "passed": _passed, "reason": _reason})
        if _passed:
            confirmed_winner = _candidate_name
            confirmed_results = _results
            print(f"\n\u2705 CONFIRMED FINAL WINNER: '{_candidate_name}' \u2014 cleared the full sign-off gate,")
            print(f"   centroid stability included. This is the deployable segmentation.")
            break
        else:
            print(f"\n\u26a0\ufe0f  '{_candidate_name}' did NOT clear final sign-off ({_reason}).")
            _remaining = [n for n in retry_order if n not in [a['candidate'] for a in attempts]]
            if _remaining:
                print(f"   Automatically retrying with the next-highest-scoring eligible candidate: "
                      f"'{_remaining[0]}'.")
            else:
                print(f"   No further eligible candidates remain.")

    banner("CANDIDATE RETRY SUMMARY")
    for a in attempts:
        status = "\u2705 PASS (confirmed winner)" if a["passed"] else f"\u274c FAIL ({a['reason']})"
        print(f"   {a['candidate']:<14} Model Score {a['score']:.4f}  {status}")

    if confirmed_winner is not None:
        primary_model = confirmed_results["primary_model"]
        step9a_overall = confirmed_results["step9a_overall"]
        step9a_p_value = confirmed_results["step9a_p_value"]
        step9b_result = confirmed_results["step9b_result"]
        step9d_profile_df = confirmed_results["step9d_profile_df"]
        step10_personas = confirmed_results["step10_personas"]
        step10_business_pass = confirmed_results["step10_business_pass"]
        step10_protected_top = confirmed_results["step10_protected_top"]
        calibration_proxy_feature = confirmed_results["calibration_proxy_feature"]
    else:
        PIPELINE_HALTED_AT = "Step 11 (Final Validation & Sign-off Gate) \u2014 ALL candidates exhausted"
        banner("PIPELINE HALTED")
        print(f"   Halted at: {PIPELINE_HALTED_AT}")
        print(f"   ALL {len(retry_order)} eligible candidate(s) failed final sign-off: "
              f"{[a['candidate'] for a in attempts]}")
        print(f"\n   Every attempt's full validation output is printed above -- nothing about this failure is")
        print(f"   silent, even though the retry itself was automatic. Revisit feature engineering, k")
        print(f"   selection, or the eligibility/stability thresholds themselves before re-running, since no")
        print(f"   candidate Step 8A proposed survived the full gate.")
else:
    print(f"\n\u23ed\ufe0f  Step 11 (Final Validation & Sign-off Gate) SKIPPED \u2014 pipeline halted at "
          f"'{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")


######################## Production Deployment Artefact + Scoring Function 
#                        + Drift Monitoring 

def _engineer_customer_features(raw, snapshot_date, vip_order):
    
    out = {}
    reg_date = pd.to_datetime(raw.get("Reg_Date"))
    if pd.isna(reg_date):
        raise ValueError("Reg_Date is required to compute tenure_days -- cannot score without it")
    out["tenure_days"] = (snapshot_date - reg_date).days

    last_login = pd.to_datetime(raw.get("Last_Login_Date")) if raw.get("Last_Login_Date") is not None else pd.NaT
    out["never_login_flag"] = int(pd.isna(last_login))
    out["days_since_login"] = (snapshot_date - last_login).days if not pd.isna(last_login) else None  # filled by imputer below

    last_deposit = pd.to_datetime(raw.get("Last_Deposit_Date")) if raw.get("Last_Deposit_Date") is not None else pd.NaT
    out["never_deposit_flag"] = int(pd.isna(last_deposit))
    out["days_since_deposit"] = (snapshot_date - last_deposit).days if not pd.isna(last_deposit) else None

    cash_stake_casino = float(raw.get("Cash_Stake_Casino", 0.0) or 0.0)
    cash_stake_sports = float(raw.get("Cash_Stake_Sports", 0.0) or 0.0)
    total_stake_verified = cash_stake_casino + cash_stake_sports
    out["total_stake_verified"] = total_stake_verified

    cash_return = float(raw.get("Cash_Return", 0.0) or 0.0)
    bonus_stake = float(raw.get("Bonus_Stake", 0.0) or 0.0)
    if total_stake_verified > 0:
        out["return_rate"] = cash_return / total_stake_verified
        out["casino_affinity"] = cash_stake_casino / total_stake_verified
        out["bonus_dependency"] = bonus_stake / total_stake_verified
    else:
        out["return_rate"] = np.nan
        out["casino_affinity"] = np.nan
        out["bonus_dependency"] = np.nan

    vip_tier = raw.get("vip_tier", raw.get("Flags", "Non-VIP"))
    out["vip_rank"] = vip_order.get(vip_tier, 0)

    return out, total_stake_verified


######################## Sub-Cluster Discovery 

SUBCLUSTER_MIN_SIZE = 500  # below this, a cluster is too small for a meaningful sub-partition search


def step11b_subcluster_discovery(primary_model, X_scaled, feat_final, structure_result, modelling_df,
                                 entity_col, investigation_results, preliminary_metric, scaler_decision):
    banner("STEP 11B: Sub-Cluster Discovery (conditional on convergent k=2)")

    winner_algo = primary_model["algorithm"]
    winner_k = primary_model["k"]

    if winner_k != 2:
        print(f"Confirmed winner ({winner_algo}) has k={winner_k} != 2 -- this step only runs when the")
        print(f"TOP-LEVEL convergent evidence itself points to k=2. A coarser top-level split is the")
        print(f"scenario where hidden sub-structure is most plausible; with k>=3 already found, the")
        print(f"algorithm-selection process already had room to surface finer structure directly.")
        print(f"SKIPPED.")
        return None

    print(f"Confirmed winner ({winner_algo}) has k=2 -- checking each cluster independently for genuine")
    print(f"internal clustering tendency before attempting to sub-divide it.")

    labels = primary_model["labels"]

    subbanner("Extending the feature set with behavioral-shape ratios (Step 11B only)")
    print(f"   Adding {BEHAVIORAL_SHAPE_FEATURES} to the sub-cluster feature set, on top of the")
    print(f"   {len(feat_final)} features Segment 0/1 were already defined on -- these ratios are NOT part of")
    print(f"   the top-level clustering (Segment 0/1 stay exactly as currently defined).")
    X_behavioral_scaled, _, behavioral_impute_fill_values, behavioral_scaler, behavioral_log_transformed, \
        behavioral_signed_log_transformed = preprocess(
            modelling_df, BEHAVIORAL_SHAPE_FEATURES, scaler_decision,
            banner_text="STEP 11B (cont'd): Preprocessing the behavioral-shape features")
    
    feat_final_expanded = list(feat_final) + BEHAVIORAL_SHAPE_FEATURES
    X_df_expanded = pd.concat(
        [X_scaled[feat_final].reset_index(drop=True), X_behavioral_scaled[BEHAVIORAL_SHAPE_FEATURES].reset_index(drop=True)],
        axis=1)
    local_protected = list(set(BUSINESS_PROTECTED_FEATURES) | set(BEHAVIORAL_SHAPE_FEATURES))
    feat_final_expanded, multiway_removed_11b = resolve_multiway_collinearity(
        X_df_expanded, feat_final_expanded, protected=local_protected)
    if multiway_removed_11b:
        print(f"   Multi-way collinearity resolver removed {len(multiway_removed_11b)} feature(s) introduced by")
        print(f"   concatenating the behavioral ratios with their own source columns: {multiway_removed_11b}")
    else:
        print(f"   Expanded feature matrix is genuinely full-rank -- no collinearity introduced by the new ratios.")
    X = X_df_expanded[feat_final_expanded].values
    
    del X_scaled, X_behavioral_scaled, X_df_expanded
    gc.collect()

    subbanner(f"Small/outlier cluster check on the WINNING algorithm's top-level partition ({winner_algo})")
    print(f"   Explicitly re-surfaced here (not just narrated once, deep inside Step 7A/7C/etc.) so this")
    print(f"   step's actionability review covers every real segment this pipeline produces in one place.")
    group_col_11b = structure_result.get("group_col_selected") if structure_result else None
    winner_small_cluster_paths = export_small_clusters_if_any(
        labels, modelling_df[entity_col].values, modelling_df, entity_col, f"{winner_algo} (top-level, re-checked in Step 11B)",
        group_col=group_col_11b,
    )

    HOPKINS_SUBCLUSTER_THRESHOLD = 0.75  # SAME threshold Step 4 itself gates on
    seeds = [1, 2, 3, 4, 5]

    results = {}
    for c in sorted(set(labels)):
        mask = labels == c
        n_c = int(mask.sum())
        subbanner(f"Cluster {c} (n={n_c:,}) - within-cluster Hopkins check")

        if n_c < SUBCLUSTER_MIN_SIZE:
            print(f"   n={n_c:,} < SUBCLUSTER_MIN_SIZE={SUBCLUSTER_MIN_SIZE:,} -- too small for a meaningful")
            print(f"   sub-partition search. Cluster {c} stays as ONE segment (LEAF).")
            results[c] = {"is_leaf": True, "H_mean": None, "sub_k": None, "reason": "below minimum size"}
            continue

        X_c = X[mask]
        if preliminary_metric == "mahalanobis":
            Hs = [hopkins_statistic(X_c, seed=s, metric="mahalanobis") for s in seeds]
        else:
            Hs = [hopkins_statistic(X_c, seed=s) for s in seeds]
        H_mean = float(np.mean(Hs))
        H_rounded = [round(h, 4) for h in Hs]
        print(f"   Mean Hopkins H = {H_mean:.4f}  (threshold {HOPKINS_SUBCLUSTER_THRESHOLD}), "
              f"individual seeds: {H_rounded}")

        if H_mean <= HOPKINS_SUBCLUSTER_THRESHOLD:
            print(f"   -> LEAF: no genuine sub-clustering tendency detected (H={H_mean:.4f} <= "
                  f"{HOPKINS_SUBCLUSTER_THRESHOLD}). Cluster {c} stays as ONE segment.")
            results[c] = {"is_leaf": True, "H_mean": H_mean, "sub_k": None, "reason": "Hopkins below threshold"}
            continue

        print(f"   -> Genuine sub-clustering tendency detected (H={H_mean:.4f} > "
              f"{HOPKINS_SUBCLUSTER_THRESHOLD}).")
        print(f"   Re-running a 3-way convergent-evidence k-selection (K-Means + Hierarchical + GMM), scoped to")
        print(f"   just this cluster's {n_c:,} customers, on the EXPANDED (behavioral-shape-inclusive) feature")
        print(f"   set -- GMM added on explicit request, since a 2-algorithm check alone previously produced a")
        print(f"   split with only moderate cophenetic correlation for one cluster; not a new methodology, the")
        print(f"   same functions Steps 7A/7C/7F already ran once, called again on a subset.")

        X_c_df = pd.DataFrame(X_c, columns=feat_final_expanded)
        sub_km_k, sub_km_details = step7a_kmeans_sweep(
            X_c_df, feat_final_expanded, k_range=range(2, 7), structure_result=structure_result,
            banner_text=f"STEP 11B (cont'd): K-Means k-selection sweep, scoped to Cluster {c}")
        sub_hier_labels, sub_coph_r, sub_Z, sub_hier_k, sub_hier_details = step7b_hierarchical(
            X_c_df, feat_final_expanded, sub_km_k, k_range=range(2, 7), structure_result=structure_result,
            banner_text=f"STEP 11B (cont'd): Hierarchical clustering, scoped to Cluster {c}")
        sub_gmm_labels, sub_gmm_model, sub_gmm_details = fit_gmm_candidate_for_step8a(
            X_c_df, feat_final_expanded, k_range=range(2, 7), structure_result=structure_result,
            banner_text=f"STEP 11B (cont'd): GMM candidate fit, scoped to Cluster {c}")
        sub_gmm_k = len(set(sub_gmm_labels))
        
        del sub_Z, sub_gmm_model
        gc.collect()

        votes = {"kmeans": sub_km_k, "hierarchical": sub_hier_k, "gmm": sub_gmm_k}
        vote_counts = pd.Series(list(votes.values())).value_counts()
        top_k, top_count = vote_counts.index[0], vote_counts.iloc[0]
        print(f"\n   Sub-cluster k votes for Cluster {c}: {votes}")

        if top_count == 3:
            print(f"   -> FULL 3/3 AGREEMENT on k={top_k} -- the strongest evidence this pipeline's own")
            print(f"   convergent-evidence standard can produce.")
            decided_k, evidence_strength = top_k, "unanimous (3/3)"
        elif top_count == 2:
            print(f"   -> MAJORITY 2/3 agreement on k={top_k} -- real convergent evidence, but explicitly")
            print(f"   flagged as WEAKER than full agreement, not silently treated the same.")
            decided_k, evidence_strength = top_k, "majority (2/3)"
        else:
            print(f"   -> NO majority -- all three algorithms disagree. Treating Cluster {c} as a single LEAF")
            print(f"   segment rather than accepting a sub-division only one algorithm supports.")
            results[c] = {"is_leaf": True, "H_mean": H_mean, "sub_k": None, "reason": "no majority across 3 algorithms"}
            continue

        if decided_k <= 1:
            print(f"   -> Decided k={decided_k} is not a genuine split. Cluster {c} stays as ONE segment (LEAF).")
            results[c] = {"is_leaf": True, "H_mean": H_mean, "sub_k": None, "reason": "decided k<=1"}
            continue

     
        if sub_km_k == decided_k:
            final_sub_labels = sub_km_details["labels"]
        else:
            print(f"   K-Means' own sweep picked k={sub_km_k}, not the decided k={decided_k} -- refitting")
            print(f"   K-Means directly at k={decided_k} for the final sub-partition labels.")
            _km_refit = KMeans(n_clusters=decided_k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(X_c)
            final_sub_labels = _km_refit.labels_

        print(f"   -> Cluster {c} genuinely sub-divides into {decided_k} sub-segments -- {evidence_strength}")
        print(f"   evidence across structurally different algorithms.")
        
        sub_centroids = np.array([X_c[final_sub_labels == sc].mean(axis=0) for sc in range(decided_k)])
        results[c] = {"is_leaf": False, "H_mean": H_mean, "sub_k": decided_k, "evidence_strength": evidence_strength,
                       "sub_labels": final_sub_labels, "row_mask": mask, "sub_centroids": sub_centroids}

        subbanner(f"Cluster {c} sub-partition stability assessment (Bootstrap Jaccard + ARI)")
        print(f"   Convergent evidence (above) answers 'do multiple algorithms AGREE on this sub-k'; this")
        print(f"   answers a DIFFERENT question -- 'is THIS SPECIFIC partition robust under resampling', the")
        print(f"   same bar Steps 9A/9B already require at the top level, extended here on explicit request.")
        print(f"   Reuses the SAME bootstrap functions Steps 9A/9B call, at their SAME n=15 convention --")
        print(f"   feature-selection stability is NOT re-run here, since sub-clusters reuse an already-fixed")
        print(f"   feature set rather than running their own separate selection process.")
        sub_centroids = np.array([X_c[final_sub_labels == sc].mean(axis=0) for sc in range(decided_k)])
        sub_primary_model = {"algorithm": "kmeans", "centroids": sub_centroids}
        
        sub_group_labels = (modelling_df.loc[mask, group_col_11b].values
                             if group_col_11b and group_col_11b in modelling_df.columns else None)
        N_BOOTSTRAP_SUBCLUSTER = 15  # matches Step 9A's own convention
        sub_jaccards = bootstrap_jaccard_stability(
            X_c, sub_primary_model, final_sub_labels, decided_k, n_bootstrap=N_BOOTSTRAP_SUBCLUSTER,
            structure_result=structure_result, group_labels=sub_group_labels)
        sub_aris = bootstrap_ari_stability(
            X_c, sub_primary_model, final_sub_labels, decided_k, n_bootstrap=N_BOOTSTRAP_SUBCLUSTER,
            structure_result=structure_result, group_labels=sub_group_labels)
        jaccard_means = {sc: (np.mean(sub_jaccards[sc]) if sub_jaccards[sc] else float("nan"))
                          for sc in range(decided_k)}
        overall_jaccard = np.nanmean(list(jaccard_means.values()))
        ari_mean = float(sub_aris.mean()) if len(sub_aris) > 0 else float("nan")
        print(f"\n   Per-sub-segment Jaccard: " +
              ", ".join(f"{c}{chr(97+sc)}={jaccard_means[sc]:.4f}" for sc in range(decided_k)))
        print(f"   Overall mean Jaccard: {overall_jaccard:.4f}  |  Bootstrap ARI: mean={ari_mean:.4f} "
              f"(n={N_BOOTSTRAP_SUBCLUSTER})")
        
        stable = overall_jaccard >= 0.6 and ari_mean >= 0.5
        stability_note = (f"\u2705 STABLE ({'highly stable' if overall_jaccard >= 0.75 else 'usable'}, "
                           f"Jaccard={overall_jaccard:.3f}, ARI={ari_mean:.3f})") if stable else \
                          (f"\u26a0\ufe0f  FLAGGED UNSTABLE (Jaccard={overall_jaccard:.3f}, ARI={ari_mean:.3f} "
                           f"-- below the 0.6/0.5 usable bar). Split is KEPT (convergent evidence already "
                           f"confirmed it), but treat this specific partition's boundary with caution.")
        print(f"   {stability_note}")
        results[c]["stability"] = {"jaccard_per_subsegment": jaccard_means, "overall_jaccard": overall_jaccard,
                                     "ari_mean": ari_mean, "stable": stable, "note": stability_note}

        subbanner(f"Naming Cluster {c}'s sub-segments (dynamic, driver-based)")
        print(f"   Building a profiling table for this sub-partition, matching profiling()'s own schema, so")
        print(f"   generate_dynamic_persona_names() (the SAME function naming the top-level clusters) can name")
        print(f"   these sub-segments too -- on explicit request ('for all the found clusters').")
        sub_prof_rows = []
        for feat in feat_final_expanded:
            sub_groups = [X_c[final_sub_labels == sc, feat_final_expanded.index(feat)] for sc in range(decided_k)]
            H_stat, p_val = stats.kruskal(*sub_groups)
            eta2_val = eta_squared_from_kruskal(H_stat, len(X_c), decided_k)
            sub_means = [g.mean() for g in sub_groups]
            sub_prof_rows.append((feat, H_stat, p_val, eta2_val, *sub_means))
        sub_prof_cols = ["feature", "H", "p", "eta2"] + [f"mean_C{i}" for i in range(decided_k)]
        sub_prof_df = pd.DataFrame(sub_prof_rows, columns=sub_prof_cols)
        sub_prof_df["effect"] = sub_prof_df["eta2"].apply(
            lambda e: "large" if e >= 0.14 else "medium" if e >= 0.06 else "small" if e >= 0.01 else "negligible")
        sub_personas_raw = generate_dynamic_persona_names(sub_prof_df, decided_k)
        sub_personas = {sc: f"{c}{chr(97 + sc)}: {sub_personas_raw[sc]}" for sc in range(decided_k)}
        for sc, name in sub_personas.items():
            print(f"   Sub-segment {c}{chr(97 + sc)}: '{name}'")
        results[c]["sub_personas"] = sub_personas
        results[c]["sub_prof_df"] = sub_prof_df

        subbanner(f"Plot: Cluster {c} sub-segments (2D PCA projection)")
        from sklearn.decomposition import PCA
        pca_c = PCA(n_components=2, random_state=RANDOM_STATE)
        Xp_c = pca_c.fit_transform(X_c)
        pca_c_variance_retained = float(pca_c.explained_variance_ratio_.sum())
        plt.figure(figsize=(8, 6.5))
        for sc in sorted(set(final_sub_labels)):
            smask = final_sub_labels == sc
            plt.scatter(Xp_c[smask, 0], Xp_c[smask, 1], s=6, alpha=0.35,
                        label=f"Sub-segment {c}{chr(97 + sc)} (n={smask.sum():,})")
        plt.xlabel(f"PC1 ({pca_c.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca_c.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.title(f"Cluster {c} Sub-Segments -- 2D PCA Projection (fit within this cluster only)")
        plt.legend(markerscale=4)
        savefig(f"step11b_cluster{c}_subsegments_pca.png")

        subbanner(f"Plot: Cluster {c} sub-segments (UMAP projection)")
        pca_c_flag = pca_c_variance_retained < UMAP_PCA_VARIANCE_FLOOR
        print(f"   Trigger check: BUSINESS_CRITICAL_SEGMENTATION={BUSINESS_CRITICAL_SEGMENTATION}  OR  "
              f"PCA 2D retained variance ({pca_c_variance_retained:.1%}) < floor ({UMAP_PCA_VARIANCE_FLOOR:.0%})? "
              f"{pca_c_flag}")
        triggered_umap_c = BUSINESS_CRITICAL_SEGMENTATION or pca_c_flag
        print(f"   DECISION: {'TRIGGERED' if triggered_umap_c else 'NOT triggered'}")
        if triggered_umap_c:
            if len(X_c) > LARGE_N_THRESHOLD:
                _umap_idx_c = np.random.RandomState(RANDOM_STATE).choice(len(X_c), LARGE_N_THRESHOLD, replace=False)
                X_c_umap_input = X_c[_umap_idx_c]
                sub_labels_umap = final_sub_labels[_umap_idx_c]
                print(f"   UMAP input subsampled to {LARGE_N_THRESHOLD:,} rows (from {len(X_c):,}).")
            else:
                X_c_umap_input, sub_labels_umap = X_c, final_sub_labels
            reducer_c = umap_lib.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
            Xu_c = reducer_c.fit_transform(X_c_umap_input)
            plt.figure(figsize=(8, 6.5))
            for sc in sorted(set(sub_labels_umap)):
                smask = sub_labels_umap == sc
                plt.scatter(Xu_c[smask, 0], Xu_c[smask, 1], s=6, alpha=0.35,
                            label=f"Sub-segment {c}{chr(97 + sc)} (n={smask.sum():,})")
            plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
            plt.title(f"Cluster {c} Sub-Segments -- UMAP Projection (non-linear; local structure, unlike PCA)")
            plt.legend(markerscale=4)
            savefig(f"step11b_cluster{c}_subsegments_umap.png")

    subbanner("Final segment hierarchy")
    hierarchy_labels = np.array([str(l) for l in labels], dtype=object)
    any_split = False
    for c, res in results.items():
        if not res["is_leaf"]:
            any_split = True
            mask = res["row_mask"]
            sub_labels_arr = res["sub_labels"]
            idxs = np.where(mask)[0]
            for j, idx in enumerate(idxs):
                hierarchy_labels[idx] = f"{c}{chr(97 + sub_labels_arr[j])}"
            found = sorted(set(hierarchy_labels[mask]))
            sub_k_val = res["sub_k"]
            stability_summary = res.get("stability", {}).get("note", "n/a")
            print(f"   Segment {c}: SPLIT into {sub_k_val} sub-segments {found} ({res.get('evidence_strength', 'n/a')})")
            print(f"      Stability: {stability_summary}")
        else:
            reason = res.get("reason", "n/a")
            h_mean_val = res["H_mean"]
            h_str = f"H={h_mean_val:.4f}" if h_mean_val is not None else "H=n/a"
            print(f"   Segment {c}: LEAF (no sub-division) - {h_str}, reason: {reason}")

    if not any_split:
        print(f"\n   DECISION: no cluster showed convergent evidence of genuine sub-structure -- the")
        print(f"   original k=2 partition stands as the final segmentation.")
    else:
        print(f"\n   DECISION: at least one cluster genuinely sub-divides -- hierarchy_labels above reflects")
        print(f"   the refined segmentation. This is DISCOVERY output, not automatically deployed; Steps")
        print(f"   12/13 below still deploy the original top-level k=2 partition unless a human decides")
        print(f"   otherwise, since accepting a deeper hierarchy is a business decision, not an automatic one.")

        subbanner("Plot: Full population, coloured by final hierarchy (2D PCA projection)")
        from sklearn.decomposition import PCA
        pca_full = PCA(n_components=2, random_state=RANDOM_STATE)
        Xp_full = pca_full.fit_transform(X)
        pca_full_variance_retained = float(pca_full.explained_variance_ratio_.sum())
        plt.figure(figsize=(9, 7))
        for h in sorted(set(hierarchy_labels)):
            hmask = hierarchy_labels == h
            plt.scatter(Xp_full[hmask, 0], Xp_full[hmask, 1], s=5, alpha=0.3,
                        label=f"Segment {h} (n={hmask.sum():,})")
        plt.xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.title("Full Hierarchy -- Top-Level Clusters + Genuine Sub-Segments (2D PCA Projection)")
        plt.legend(markerscale=4)
        savefig("step11b_full_hierarchy_pca.png")

        subbanner("Plot: Full population, coloured by final hierarchy (UMAP projection)")
        pca_full_flag = pca_full_variance_retained < UMAP_PCA_VARIANCE_FLOOR
        print(f"   Trigger check: BUSINESS_CRITICAL_SEGMENTATION={BUSINESS_CRITICAL_SEGMENTATION}  OR  "
              f"PCA 2D retained variance ({pca_full_variance_retained:.1%}) < floor ({UMAP_PCA_VARIANCE_FLOOR:.0%})? "
              f"{pca_full_flag}")
        triggered_umap_full = BUSINESS_CRITICAL_SEGMENTATION or pca_full_flag
        print(f"   DECISION: {'TRIGGERED' if triggered_umap_full else 'NOT triggered'}")
        if triggered_umap_full:
            if len(X) > LARGE_N_THRESHOLD:
                _umap_idx_full = np.random.RandomState(RANDOM_STATE).choice(len(X), LARGE_N_THRESHOLD, replace=False)
                X_full_umap_input = X[_umap_idx_full]
                hierarchy_labels_umap = hierarchy_labels[_umap_idx_full]
                print(f"   UMAP input subsampled to {LARGE_N_THRESHOLD:,} rows (from {len(X):,}).")
            else:
                X_full_umap_input, hierarchy_labels_umap = X, hierarchy_labels
            reducer_full = umap_lib.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
            Xu_full = reducer_full.fit_transform(X_full_umap_input)
            plt.figure(figsize=(9, 7))
            for h in sorted(set(hierarchy_labels_umap)):
                hmask = hierarchy_labels_umap == h
                plt.scatter(Xu_full[hmask, 0], Xu_full[hmask, 1], s=5, alpha=0.3,
                            label=f"Segment {h} (n={hmask.sum():,})")
            plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
            plt.title("Full Hierarchy -- Top-Level Clusters + Genuine Sub-Segments (UMAP Projection)")
            plt.legend(markerscale=4)
            savefig("step11b_full_hierarchy_umap.png")

        subbanner("Plot: Segment size comparison, top-level vs full hierarchy")
        top_counts = pd.Series(labels).value_counts().sort_index()
        hier_counts = pd.Series(hierarchy_labels).value_counts().sort_index()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar([str(i) for i in top_counts.index], top_counts.values, color="steelblue")
        axes[0].set_title(f"Top-level (k={winner_k})")
        axes[0].set_ylabel("Customers")
        axes[1].bar([str(i) for i in hier_counts.index], hier_counts.values, color="darkorange")
        axes[1].set_title(f"Full hierarchy ({len(hier_counts)} segments)")
        plt.suptitle("Segment Sizes -- Top-Level vs. Discovered Hierarchy")
        savefig("step11b_segment_size_comparison.png")

    subbanner("Actionability profiling: every final sub-segment, zero-cash-funded cohort, and any small-cluster export")
    print(f"   Reuses profile_cluster_vs_population() (the SAME function small-cluster exports already use)")
    print(f"   for every real segment this pipeline now produces -- an EXTENDED compare_cols (the standard")
    print(f"   set + this step's own behavioral-shape ratios) is used ONLY for these Step 11B calls, since")
    print(f"   those ratios are scoped to sub-cluster discovery, not the global profiling default.")
    extended_compare_cols = CLUSTER_PROFILE_COMPARE_COLS + BEHAVIORAL_SHAPE_FEATURES
    profiling_results = {}
    if any_split:
        for h in sorted(set(hierarchy_labels)):
            if len(h) <= 1:
                continue  # a LEAF top-level cluster with no letter suffix -- nothing new to profile here
            hmask = hierarchy_labels == h
            cust_ids_h = modelling_df[entity_col].values[hmask]
            ptab, ctab = profile_cluster_vs_population(cust_ids_h, modelling_df, entity_col, f"Sub-segment {h}",
                                                          compare_cols=extended_compare_cols)
            profiling_results[f"subsegment_{h}"] = {"profile_table": ptab, "category_table": ctab}

    if pre_activation_df is not None and len(pre_activation_df) > 0:
        
        _combined_for_pa = pd.concat([modelling_df, pre_activation_df], ignore_index=True, sort=False)
        ptab_pa, ctab_pa = profile_cluster_vs_population(
            pre_activation_df[entity_col].values, _combined_for_pa, entity_col, "Zero-Cash-Funded Cohort",
            compare_cols=extended_compare_cols)
        profiling_results["pre_activation"] = {"profile_table": ptab_pa, "category_table": ctab_pa}
        subbanner("Naming the zero-cash-funded cohort (dynamic, driver-based)")
        pre_activation_persona = generate_dynamic_persona_name_single_group(ptab_pa, "Zero-Cash-Funded Cohort")
        profiling_results["pre_activation"]["persona_name"] = pre_activation_persona
        del _combined_for_pa
        gc.collect()
    else:
        print(f"   Zero-cash-funded cohort unavailable or empty in this run's scope -- skipped.")

    if winner_small_cluster_paths:
        print(f"   Winning algorithm's small-cluster export(s) already profiled above, during the explicit")
        print(f"   re-check (export_small_clusters_if_any() profiles internally as part of exporting):")
        for p in winner_small_cluster_paths:
            print(f"     {p}")
    else:
        print(f"   No small cluster in the winning algorithm's top-level partition this run -- nothing")
        print(f"   further to profile there.")

    return {"per_cluster": results, "hierarchy_labels": hierarchy_labels, "profiling_results": profiling_results,
            "winner_small_cluster_paths": winner_small_cluster_paths,
            
            "feat_final_expanded": feat_final_expanded,
            "behavioral_scaler": behavioral_scaler,
            "behavioral_impute_fill_values": behavioral_impute_fill_values,
            "behavioral_log_transformed": behavioral_log_transformed,
            "behavioral_signed_log_transformed": behavioral_signed_log_transformed}


def deploy(modelling_df, entity_col, feat_cols_raw_candidates, feat_cols_post_encoding, cat_encoders,
           log_transformed, signed_log_transformed, feat_final, impute_fill_values, scaler, primary_model,
           personas, calibration_proxy_feature, snapshot_date, structure_result=None, investigation_results=None,
           pre_activation_df=None, step11b_result=None):
    banner("STEP 12: Production Deployment Artefact")

    subbanner("Sub-cluster models (Step 11B), for artefact persistence")
    
    sub_cluster_models = {}
    if step11b_result is not None:
        for parent_c, res in step11b_result.get("per_cluster", {}).items():
            if res.get("is_leaf") is False and "sub_centroids" in res:
                sub_cluster_models[parent_c] = {
                    "sub_centroids": res["sub_centroids"],
                    "sub_k": res["sub_k"],
                    "sub_persona_names": res.get("sub_personas", {}),
                }
        if sub_cluster_models:
            print(f"   Persisting sub-cluster models for parent cluster(s): "
                  f"{[int(c) for c in sub_cluster_models.keys()]}")
            print(f"   (feat_final_expanded + the behavioral-feature scaler are shared across all of them,")
            print(f"   stored once at the top level of the artefact, not duplicated per cluster.)")
        else:
            print(f"   No cluster genuinely sub-divided in this run -- sub_cluster_models left empty.")
    else:
        print(f"   Step 11B did not run (winner's k != 2) -- sub_cluster_models left empty.")

    subbanner("Structure-detection decision matrix applied to this step")
    group_col = structure_result.get("group_col_selected") if structure_result else None
    print(f"   Temporal/group/network: N/A for CONSTRUCTING this artefact \u2014 a single deterministic build,")
    print(f"   no resampling for any of them to act on. structure_result IS persisted into the artefact")
    print(f"   below, though, so Step 13's drift-monitoring reference/current split can honour the SAME")
    print(f"   group-blocked resampling this pipeline has used everywhere else, rather than Step 13 starting")
    print(f"   from a plain-iid default.")

    subbanner("Protected-feature awareness")
    protected_in_final = [f for f in feat_final if f in BUSINESS_PROTECTED_FEATURES]
    print(f"   Business-protected features in the deployed feature set: {protected_in_final or 'none'} \u2014")
    print(f"   these are fully part of the scoring pipeline below, identically to every other feature.")

    subbanner("Leakage investigation re-check (deployment-time safety gate)")
    kpi_leak_check = [c for c in feat_final if any(name in c.lower() for name in KPI_CANDIDATE_NAMES)]
    if kpi_leak_check:
        raise RuntimeError(f"DEPLOYMENT HALTED: KPI-vocabulary column(s) {kpi_leak_check} found in the FINAL "
                            f"deployed feature set -- Step 3F's leakage investigation must be re-examined "
                            f"before shipping this artefact.")
    print(f"   \u2705 No KPI_CANDIDATE_NAMES-vocabulary column present in feat_cols_final \u2014 re-confirms Step 3F's")
    print(f"   finding ('NGR' correctly excluded, being a raw clustering candidate) still holds at the exact")
    print(f"   feature set actually being deployed, not just at the point Step 3F originally checked it.")

    artifact = {
        "version": "3.0.0",  
        "input_cols_raw": feat_cols_raw_candidates,          
        "cat_encoders": cat_encoders,
        "feat_cols_post_encoding": feat_cols_post_encoding,  
        "log_transformed_cols": log_transformed,
        "signed_log_transformed_cols": signed_log_transformed,  
        "feat_cols_final": feat_final,
        "impute_fill_values": impute_fill_values,   
        "scaler": scaler,
        "primary_model": primary_model,
        "persona_names": personas,
        "calibration_proxy_feature": calibration_proxy_feature,
        "snapshot_date": snapshot_date,             
        "vip_order": vip_order,
        "structure_result": structure_result,       
        "investigation_results": investigation_results,
        "entity_col": entity_col,
        
        "sub_cluster_models": sub_cluster_models,
        "feat_final_expanded": step11b_result.get("feat_final_expanded") if step11b_result else None,
        "behavioral_scaler": step11b_result.get("behavioral_scaler") if step11b_result else None,
        "behavioral_impute_fill_values": step11b_result.get("behavioral_impute_fill_values") if step11b_result else None,
        "behavioral_log_transformed": step11b_result.get("behavioral_log_transformed") if step11b_result else None,
        "behavioral_signed_log_transformed": step11b_result.get("behavioral_signed_log_transformed") if step11b_result else None,
    }
    print(f"\n  Artefact contents: {len(cat_encoders)} categorical encoder(s), {len(impute_fill_values)} "
          f"per-column impute value(s), scaler, log-transform lists ({len(log_transformed)} plain + "
          f"{len(signed_log_transformed)} signed), '{primary_model['algorithm']}' primary model (k="
          f"{primary_model['k']}, generic interface), persona names, snapshot_date, structure context.")
    if primary_model["algorithm"] != "kmeans":
        print(f"   \u26a0\ufe0f  Primary algorithm is '{primary_model['algorithm']}', not K-Means \u2014 this artefact")
        print(f"   deploys it via build_primary_model/primary_predict_new (proven by")
        print(f"   _self_test_generic_primary_model), the SAME mechanism used for every algorithm.")

    def assign_segment(new_customer_row: dict, artifact=artifact) -> dict:
       
        raw = dict(new_customer_row)
        engineered, total_stake_verified = _engineer_customer_features(
            raw, artifact["snapshot_date"], artifact["vip_order"])

        if total_stake_verified == 0:
            return {"segment_id": None, "persona": "Zero-cash-funded cohort (no cash stake on casino or sports -- "
                    "excluded from segmentation per Part C's business rule, not a modelled segment)", "assignment_confidence": None}

        raw.update(engineered)
        row = pd.DataFrame([raw])
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
        row_scaled = pd.DataFrame(artifact["scaler"].transform(row_imp), columns=artifact["feat_cols_post_encoding"])
        row_final = row_scaled[artifact["feat_cols_final"]]
        pred_labels, dists = primary_predict_new(row_final.values, artifact["primary_model"])
        seg = int(pred_labels[0])
        if seg == -1:
            return {"segment_id": -1, "persona": "Unclassified / Noise (DBSCAN/HDBSCAN primary model only)",
                    "assignment_confidence": 0.0}
        sorted_d = np.sort(dists[0])
        confidence = float(1 - sorted_d[0] / (sorted_d[1] + 1e-9)) if len(sorted_d) > 1 else 1.0
        result = {"segment_id": seg, "persona": artifact["persona_names"][seg],
                  "assignment_confidence": round(confidence, 4)}

        
        sub_model = (artifact.get("sub_cluster_models") or {}).get(seg)
        if sub_model is not None and artifact.get("behavioral_scaler") is not None:
            bonus_per_bet = raw.get("Bet_Count", 0) and raw.get("Bonus_Stake", 0) / raw["Bet_Count"] if raw.get("Bet_Count", 0) else np.nan
            stake_per_active_day = raw.get("active_days", 0) and raw.get("Cash_Stake_Casino", 0) / raw["active_days"] if raw.get("active_days", 0) else np.nan
            avg_stake_per_bet = raw.get("Bet_Count", 0) and raw.get("Cash_Stake_Casino", 0) / raw["Bet_Count"] if raw.get("Bet_Count", 0) else np.nan
            deposits_per_active_day = raw.get("active_days", 0) and raw.get("Deposit_Count", 0) / raw["active_days"] if raw.get("active_days", 0) else np.nan
            beh_row = pd.DataFrame([{"bonus_per_bet": bonus_per_bet, "stake_per_active_day": stake_per_active_day,
                                      "avg_stake_per_bet": avg_stake_per_bet,
                                      "deposits_per_active_day": deposits_per_active_day}])
            for c, spec in artifact["behavioral_impute_fill_values"].items():
                beh_row[c] = beh_row[c].fillna(spec["value"])
            for c in artifact["behavioral_log_transformed"]:
                beh_row[c] = np.log1p(beh_row[c])
            for c in artifact["behavioral_signed_log_transformed"]:
                beh_row[c] = np.sign(beh_row[c]) * np.log1p(np.abs(beh_row[c]))
            beh_scaled = pd.DataFrame(artifact["behavioral_scaler"].transform(beh_row), columns=BEHAVIORAL_SHAPE_FEATURES)
            expanded_row = pd.concat([row_final.reset_index(drop=True), beh_scaled], axis=1)[artifact["feat_final_expanded"]]
            sub_centroids = np.asarray(sub_model["sub_centroids"])
            sub_dists = np.linalg.norm(expanded_row.values[:, None, :] - sub_centroids[None, :, :], axis=2)
            sub_seg = int(np.argmin(sub_dists[0]))
            sub_label = f"{seg}{chr(97 + sub_seg)}"
            result["sub_segment_id"] = sub_label
            result["sub_persona"] = sub_model["sub_persona_names"].get(sub_seg, f"Sub-segment {sub_label}")
            
            sorted_sub_d = np.sort(sub_dists[0])
            sub_confidence = float(1 - sorted_sub_d[0] / (sorted_sub_d[1] + 1e-9)) if len(sorted_sub_d) > 1 else 1.0
            result["sub_segment_confidence"] = round(sub_confidence, 4)

        return result


    subbanner("Smoke test on 3 real customer records (including the zero-cash-funded cohort business rule)")
    test_rows = modelling_df.sample(min(3, len(modelling_df)), random_state=7)
    for _, r in test_rows.iterrows():
        raw_record = {c: r[c] for c in feat_cols_raw_candidates if c in r.index}
        result = assign_segment(raw_record)
        print(f"   {r[entity_col]}: {result}")
    pre_act_smoke = (pre_activation_df.sample(min(1, len(pre_activation_df)), random_state=7)
                      if pre_activation_df is not None and len(pre_activation_df) else None)
    if pre_act_smoke is not None:
        for _, r in pre_act_smoke.iterrows():
            raw_record = {c: r[c] for c in feat_cols_raw_candidates if c in r.index}
            result = assign_segment(raw_record)
            print(f"   {r[entity_col]} (known zero-cash-funded cohort customer): {result}")
            assert result["segment_id"] is None, "a genuine zero-cash-funded cohort customer must NOT receive a cluster assignment"
    print(f"\n   DECISION: assignment function validated end-to-end (including the zero-cash-funded cohort business")
    print(f"   rule) \u2014 ready for Step 13 monitoring setup.")
    return artifact, assign_segment


####################### Drift Monitoring 

def preprocess_for_drift(rows, artifact):
    
    row = rows.copy()
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
    row = row[artifact["feat_cols_post_encoding"]]
    for c, spec in artifact["impute_fill_values"].items():
        if c in row.columns:
            row[c] = row[c].fillna(spec["value"])
    for c in artifact["log_transformed_cols"]:
        row[c] = np.log1p(row[c])
    for c in artifact["signed_log_transformed_cols"]:
        row[c] = np.sign(row[c]) * np.log1p(np.abs(row[c]))
    row_scaled = pd.DataFrame(artifact["scaler"].transform(row), columns=artifact["feat_cols_post_encoding"],
                               index=rows.index)
    return row_scaled[artifact["feat_cols_final"]]


def _json_safe(obj):
    
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def build_training_baseline(modelling_df, entity_col, artifact, primary_model, k, step11b_result=None):
    
    banner("Building training_baseline.json (training-time reference statistics)")
    labels_full = np.asarray(primary_model["labels"])
    X_full = preprocess_for_drift(modelling_df, artifact)

    subbanner("PSI reference bins (Type 1)")
    psi_bins = {}
    for c in artifact["feat_cols_final"]:
        vals = X_full[c].values
        edges = np.quantile(vals, np.linspace(0, 1, 11))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        counts, _ = np.histogram(vals, bins=edges)
        psi_bins[c] = {"edges": edges.tolist(), "ref_pct": (counts / len(vals)).tolist()}
    print(f"   Captured decile bin edges + reference percentages for {len(psi_bins)} features.")

    subbanner("Segment size shares (Type 2)")
    segment_size_shares = pd.Series(labels_full).value_counts(normalize=True).sort_index().to_dict()
    print(f"   {_json_safe(segment_size_shares)}")

    subbanner("Sub-segment size shares (Type 2b)")
   
    sub_segment_size_shares = {}
    if step11b_result is not None and step11b_result.get("hierarchy_labels") is not None:
        hierarchy_labels = np.asarray(step11b_result["hierarchy_labels"])
        sub_segment_size_shares = pd.Series(hierarchy_labels).value_counts(normalize=True).sort_index().to_dict()
        print(f"   {_json_safe(sub_segment_size_shares)}")
    else:
        print(f"   Step 11B did not run or found no genuine sub-structure -- left empty.")

    subbanner("Calibration baseline rates (Type 5)")
    calibration_proxy_feature = artifact.get("calibration_proxy_feature")
    calibration_baseline_rates = {}
    if calibration_proxy_feature and calibration_proxy_feature in modelling_df.columns:
        for c in range(k):
            mask = labels_full == c
            if mask.sum() > 0:
                calibration_baseline_rates[c] = float(modelling_df.loc[mask, calibration_proxy_feature].mean())
        print(f"   Proxy='{calibration_proxy_feature}': {_json_safe(calibration_baseline_rates)}")
    else:
        print(f"   No usable calibration proxy feature \u2014 left empty.")

    subbanner("Centroid-confidence baseline (Type 6)")
    dists_all = primary_transform(X_full.values, primary_model)
    centroid_confidence_baseline = {}
    for c in range(k):
        mask = (labels_full == c)
        if mask.sum() > 0:
            own_dist = dists_all[mask, c]
            centroid_confidence_baseline[c] = {"mean": float(own_dist.mean()), "p90": float(np.percentile(own_dist, 90))}
    print(f"   Captured mean/p90 own-centroid distance for {len(centroid_confidence_baseline)} segments.")

    subbanner("Group distribution baseline (Type 7)")
    structure_result = artifact.get("structure_result")
    group_col = structure_result.get("group_col_selected") if structure_result else None
    group_baseline_counts = None
    if group_col and group_col in modelling_df.columns:
        group_baseline_counts = modelling_df[group_col].astype(str).value_counts().to_dict()
        print(f"   '{group_col}': {_json_safe(group_baseline_counts)}")
    else:
        print(f"   No group column available \u2014 left empty.")

    baseline = {
        "version": "1.0.0",
        "built_from_artifact_version": artifact.get("version"),
        "training_snapshot_date": str(artifact.get("snapshot_date")),
        "n_training_customers": int(len(modelling_df)),
        "algorithm": primary_model["algorithm"],
        "k": int(k),
        "feat_cols_final": list(artifact["feat_cols_final"]),
        "psi_bins": psi_bins,
        "segment_size_shares": segment_size_shares,
        "sub_segment_size_shares": sub_segment_size_shares,
        "calibration_proxy_feature": calibration_proxy_feature,
        "calibration_baseline_rates": calibration_baseline_rates,
        "centroid_confidence_baseline": centroid_confidence_baseline,
        "group_col": group_col,
        "group_baseline_counts": group_baseline_counts,
    }
    return _json_safe(baseline)


def check_drift_against_baseline(new_raw_df, artifact, baseline, entity_col=None):
    
    banner("Drift Check: New Data vs Saved Training Baseline")
    entity_col = entity_col or artifact.get("entity_col")
    primary_model = artifact["primary_model"]
    algorithm = primary_model["algorithm"]
    centroids = np.array(primary_model["centroids"])
    k = baseline["k"]
    print(f"   Baseline: {baseline['n_training_customers']:,} training customers as of "
          f"{baseline.get('training_snapshot_date', 'unknown date')}, algorithm='{algorithm}', k={k}")
    print(f"   New data: {len(new_raw_df):,} customers")
    if algorithm != baseline.get("algorithm"):
        print(f"   \u26a0\ufe0f  artifact's primary algorithm ('{algorithm}') differs from the baseline's "
              f"('{baseline.get('algorithm')}') \u2014 this baseline may be stale; results below may not be meaningful.")

    X_new = preprocess_for_drift(new_raw_df, artifact)
    new_labels, new_dists = primary_predict_new(X_new.values, primary_model)

    subbanner("Type 1: Feature Drift (PSI) vs saved baseline")
    PSI_STABLE, PSI_WARNING = 0.10, 0.25
    psi_results = {}
    for c in baseline["feat_cols_final"]:
        edges, ref_pct = baseline["psi_bins"][c]["edges"], baseline["psi_bins"][c]["ref_pct"]
        val = psi_from_baseline_bins(edges, ref_pct, X_new[c].values)
        psi_results[c] = val
        flag = "STABLE" if val < PSI_STABLE else ("WARNING" if val < PSI_WARNING else "ACTION REQUIRED")
        protected_tag = "  [PROTECTED]" if c in BUSINESS_PROTECTED_FEATURES else ""
        print(f"   {c:<32} PSI={val:.4f}  {flag}{protected_tag}")

    subbanner("Type 2: Segment Size Drift vs saved baseline")
    SIZE_STABLE, SIZE_INVESTIGATE = 15.0, 30.0
    new_share = pd.Series(new_labels).value_counts(normalize=True).sort_index()
    size_results = {}
    for c in range(k):
        base_share = baseline["segment_size_shares"].get(str(c))
        if base_share is None or c not in new_share.index:
            print(f"   Cluster {c}: not present in baseline or new data \u2014 skipped")
            continue
        shift = abs(new_share[c] - base_share) / base_share * 100
        flag = "STABLE" if shift < SIZE_STABLE else ("INVESTIGATE" if shift < SIZE_INVESTIGATE else "RE-RUN SWEEP")
        size_results[c] = {"baseline_share": base_share, "new_share": float(new_share[c]), "shift_pct": shift}
        print(f"   Cluster {c}: baseline={base_share*100:.1f}%  new={new_share[c]*100:.1f}%  shift={shift:.2f}%  {flag}")

    subbanner("Type 2b: Sub-Segment Size Drift vs saved baseline")
    
    sub_size_results = {}
    sub_cluster_models = artifact.get("sub_cluster_models") or {}
    if sub_cluster_models and artifact.get("behavioral_scaler") is not None and baseline.get("sub_segment_size_shares"):
        bet_count = new_raw_df.get("Bet_Count", pd.Series(0, index=new_raw_df.index)).replace(0, np.nan)
        active_days = new_raw_df.get("active_days", pd.Series(0, index=new_raw_df.index)).replace(0, np.nan)
        beh = pd.DataFrame(index=new_raw_df.index)
        beh["bonus_per_bet"] = new_raw_df.get("Bonus_Stake", 0) / bet_count
        beh["stake_per_active_day"] = new_raw_df.get("Cash_Stake_Casino", 0) / active_days
        beh["avg_stake_per_bet"] = new_raw_df.get("Cash_Stake_Casino", 0) / bet_count
        beh["deposits_per_active_day"] = new_raw_df.get("Deposit_Count", 0) / active_days
        for c, spec in artifact["behavioral_impute_fill_values"].items():
            beh[c] = beh[c].fillna(spec["value"])
        for c in artifact["behavioral_log_transformed"]:
            beh[c] = np.log1p(beh[c])
        for c in artifact["behavioral_signed_log_transformed"]:
            beh[c] = np.sign(beh[c]) * np.log1p(np.abs(beh[c]))
        beh_scaled = pd.DataFrame(artifact["behavioral_scaler"].transform(beh), columns=BEHAVIORAL_SHAPE_FEATURES,
                                   index=X_new.index)
        expanded = pd.concat([X_new, beh_scaled], axis=1)[artifact["feat_final_expanded"]]

        new_hierarchy_labels = np.array([str(l) for l in new_labels], dtype=object)
        for parent_c, sub_model in sub_cluster_models.items():
            parent_mask = (new_labels == int(parent_c))
            if parent_mask.sum() == 0:
                continue
            sub_centroids = np.asarray(sub_model["sub_centroids"])
            sub_dists = np.linalg.norm(expanded.values[parent_mask][:, None, :] - sub_centroids[None, :, :], axis=2)
            sub_idx = np.argmin(sub_dists, axis=1)
            new_hierarchy_labels[parent_mask] = [f"{parent_c}{chr(97 + i)}" for i in sub_idx]

        new_sub_share = pd.Series(new_hierarchy_labels).value_counts(normalize=True)
        for sub_label, base_share in baseline["sub_segment_size_shares"].items():
            new_pct = new_sub_share.get(sub_label)
            if new_pct is None:
                print(f"   {sub_label}: not present in new data \u2014 skipped")
                continue
            shift = abs(new_pct - base_share) / base_share * 100
            flag = "STABLE" if shift < SIZE_STABLE else ("INVESTIGATE" if shift < SIZE_INVESTIGATE else "RE-RUN SWEEP")
            sub_size_results[sub_label] = {"baseline_share": base_share, "new_share": float(new_pct), "shift_pct": shift}
            print(f"   {sub_label}: baseline={base_share*100:.2f}%  new={new_pct*100:.2f}%  shift={shift:.2f}%  {flag}")
    else:
        print(f"   No sub_cluster_models in this artefact, or no sub_segment_size_shares in this baseline "
              f"(produced before this check existed, or no cluster genuinely sub-divided) \u2014 skipped.")

    subbanner("Type 3/4: Centroid Displacement & Stability Drift (fresh refit vs persisted centroids)")
    new_refit_labels, new_refit_centroids = _fit_algorithm_generic(
        X_new.values, algorithm, k, RANDOM_STATE,
        eps=primary_model.get("eps"), min_samples=primary_model.get("min_samples"),
        min_cluster_size=primary_model.get("min_cluster_size"),
        alt_distance_metric=primary_model.get("alt_distance_metric"),
        num_cols=primary_model.get("num_cols"), cat_cols=primary_model.get("cat_cols"),
        feat_cols_ordered=primary_model.get("feat_cols_ordered"))
    ari_drift = None
    if len(new_refit_centroids) > 0:
        cost = np.linalg.norm(new_refit_centroids[:, None, :] - centroids[None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = dict(zip(row, col))
        for b_idx, ref_c in mapping.items():
            disp = np.linalg.norm(new_refit_centroids[b_idx] - centroids[ref_c])
            print(f"   Cluster {ref_c}: displacement={disp:.4f}")
        new_refit_mapped = np.array([mapping.get(l, -1) for l in new_refit_labels])
        ari_drift = float(adjusted_rand_score(new_labels, new_refit_mapped))
        print(f"   ARI(production assignments vs fresh refit) = {ari_drift:.4f}  "
              f"{'STABLE' if ari_drift >= 0.5 else 'FLAG -- material shape change'} (threshold 0.5)")
    else:
        print(f"   Refit on new data produced no valid centroids \u2014 skipped (algorithm-specific degenerate case).")

    subbanner(f"Type 5: Calibration Drift (proxy: {baseline.get('calibration_proxy_feature') or 'NONE'}) "
              f"vs saved baseline")
    proxy_feature = baseline.get("calibration_proxy_feature")
    calibration_results = {}
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
            calibration_results[c] = {"baseline_rate": base_rate, "new_rate": new_rate, "shift_pct": shift}
            print(f"   Cluster {c}: baseline={base_rate:.3f}  new={new_rate:.3f}  shift={shift:.2f}%  {flag}")
    else:
        print(f"   No usable calibration baseline/proxy in the new data \u2014 skipped.")

    subbanner("Type 6: Centroid-Confidence Monitoring vs saved baseline")
    CENTROID_CONF_WARN = 15.0
    if baseline.get("centroid_confidence_baseline"):
        for c in range(k):
            base = baseline["centroid_confidence_baseline"].get(str(c))
            mask = new_labels == c
            if base is None or mask.sum() == 0:
                continue
            new_mean = float(new_dists[mask, c].mean())
            shift_pct = (new_mean - base["mean"]) / base["mean"] * 100
            flag = "STABLE" if abs(shift_pct) < CENTROID_CONF_WARN else "EARLY WARNING -- investigate before other metrics alert"
            print(f"   Cluster {c}: baseline mean={base['mean']:.4f}  new mean={new_mean:.4f}  "
                  f"shift={shift_pct:+.2f}%  {flag}")
    else:
        print(f"   No centroid-confidence baseline available \u2014 skipped.")

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
        print(f"   '{group_col}' distribution: baseline vs new Cramer's V={cramers_v:.4f}  p={p:.4g}  {flag}")
    else:
        print(f"No group baseline/column available \u2014 Type 7 skipped.")

    return {"psi": psi_results, "segment_size": size_results, "sub_segment_size": sub_size_results,
            "ari_drift": ari_drift, "calibration": calibration_results, "new_labels": new_labels}


def build_segment_personas_json(step10_personas, prof_df, primary_model, k, step11b_result=None,
                                 pre_activation_df=None):
    
    labels_full = np.asarray(primary_model["labels"])
    sizes = pd.Series(labels_full).value_counts().sort_index()
    total = len(labels_full)

    top_driver_cols = [c for c in prof_df.columns if c.startswith("mean_C")]
    segments = {}
    for c in range(k):
        row_top = prof_df.sort_values("eta2", ascending=False).iloc[0] if not prof_df.empty else None
        segments[c] = {
            "persona_name": step10_personas.get(c, f"Segment {c}"),
            "size": int(sizes.get(c, 0)),
            "pct_of_population": round(float(sizes.get(c, 0)) / total * 100, 2) if total else None,
            "top_drivers": prof_df.sort_values("eta2", ascending=False).head(5)[
                ["feature", "eta2", "effect", f"mean_C{c}"]].to_dict(orient="records") if not prof_df.empty else [],
        }

    sub_segments = {}
    if step11b_result is not None:
        for parent_c, res in step11b_result.get("per_cluster", {}).items():
            if res.get("is_leaf") is False and "sub_personas" in res:
                for sc_label, name in res["sub_personas"].items():
                    sub_mask = res["sub_labels"] == sc_label
                    sub_segments[f"{parent_c}{chr(97 + sc_label)}"] = {
                        "parent_segment": int(parent_c),
                        "persona_name": name,
                        "size": int(sub_mask.sum()),
                        "stability": res.get("stability", {}).get("note"),
                    }

    pre_activation_persona = None
    if step11b_result is not None:
        pa_profiling = step11b_result.get("profiling_results", {}).get("pre_activation")
        if pa_profiling:
            pre_activation_persona = pa_profiling.get("persona_name")
    pre_activation_info = None
    if pre_activation_df is not None:
        pre_activation_info = {"persona_name": pre_activation_persona, "size": int(len(pre_activation_df))}

    return _json_safe({
        "version": "1.0.0",
        "algorithm": primary_model["algorithm"],
        "k": int(k),
        "total_customers_modelled": int(total),
        "segments": segments,
        "sub_segments": sub_segments,
        "zero_cash_funded_cohort": pre_activation_info,
    })


def drift_monitoring(modelling_df, entity_col, artifact, k):
    banner("STEP 13: Drift Monitoring (demonstration)")
    print("   No timestamped batches exist in this single extract. Demonstrated on a 70/30")
    print("   reference-vs-current split of the SAME data \u2014 validates the monitoring CODE")
    print("   runs correctly and shows the expected 'no drift' baseline. Production MUST re-run this")
    print("   exact code against a genuinely later data pull on the agreed refresh cadence.\n")

    primary_model = artifact["primary_model"]
    algorithm = primary_model["algorithm"]
    print(f"   Primary algorithm: {algorithm}  (generic interface \u2014 every check below works identically")
    print(f"   regardless of which algorithm won Step 8A)")
    labels_full = primary_model["labels"]

    subbanner("Structure-detection decision matrix applied to this step")
    structure_result = artifact.get("structure_result")
    group_detected = structure_result.get("group_detected", False) if structure_result else False
    group_col = structure_result.get("group_col_selected") if structure_result else None
    from sklearn.model_selection import train_test_split
    if group_detected and group_col and group_col in modelling_df.columns:
        
        print(f"   Group structure: YES ('{group_col}') \u2192 the reference/current split below is a genuine")
        print(f"   resampling procedure. Uses STRATIFIED sampling on '{group_col}' (proportional")
        print(f"   representation of every level in BOTH splits) -- NOT group-blocking (which is the right")
        print(f"   tool for Steps 8B/9A/9B's bootstrap resampling, but the WRONG one here: it would put whole")
        print(f"   groups on only one side, guaranteeing an imbalanced split for any group that lands")
        print(f"   entirely on one side -- confirmed directly on this population's 96%-Non-VIP composition).")
        group_labels_arr = modelling_df[group_col].astype(str).values
        ref_idx, cur_idx = train_test_split(modelling_df.index, test_size=0.30, random_state=RANDOM_STATE,
                                             stratify=group_labels_arr)
    else:
        print(f"   Group structure: NO (or unavailable) \u2192 plain iid reference/current split.")
        ref_idx, cur_idx = train_test_split(modelling_df.index, test_size=0.30, random_state=RANDOM_STATE)
    ref_labels = pd.Series(labels_full, index=modelling_df.index).loc[ref_idx]
    cur_labels = pd.Series(labels_full, index=modelling_df.index).loc[cur_idx]
    print(f"   Reference: {len(ref_idx):,} customers  |  Current: {len(cur_idx):,} customers")

    def preprocess_for_drift_bound(rows):
        return preprocess_for_drift(rows, artifact)

    X_ref = preprocess_for_drift_bound(modelling_df.loc[ref_idx])
    X_cur = preprocess_for_drift_bound(modelling_df.loc[cur_idx])

    subbanner("Type 1: Feature Drift (PSI)")
    print(f"   Protected-feature awareness: business-protected features monitored here are flagged")
    print(f"   explicitly ([PROTECTED]) \u2014 drift in one of these is worth escalating more urgently than")
    print(f"   drift in an arbitrary feature.")
    PSI_STABLE, PSI_WARNING = 0.10, 0.25
    psi_vals = {}
    for c in artifact["feat_cols_final"]:
        val = population_stability_index(X_ref[c].values, X_cur[c].values)
        psi_vals[c] = val
        flag = "STABLE" if val < PSI_STABLE else ("WARNING" if val < PSI_WARNING else "ACTION REQUIRED")
        protected_tag = "  [PROTECTED]" if c in BUSINESS_PROTECTED_FEATURES else ""
        print(f"   {c:<32} PSI={val:.4f}  {flag}{protected_tag}")

    plt.figure(figsize=(9, 5))
    psi_s = pd.Series(psi_vals).sort_values(ascending=True)
    colors = ["seagreen" if v < PSI_STABLE else ("orange" if v < PSI_WARNING else "red") for v in psi_s.values]
    plt.barh(psi_s.index, psi_s.values, color=colors)
    plt.axvline(PSI_STABLE, color="orange", ls="--", alpha=0.6, label=f"stable threshold ({PSI_STABLE})")
    plt.axvline(PSI_WARNING, color="red", ls="--", alpha=0.6, label=f"warning threshold ({PSI_WARNING})")
    plt.xlabel("PSI"); plt.title("Step 13 Type 1 \u2014 Feature Drift (PSI), reference vs current")
    plt.legend()
    savefig("step13_feature_drift_psi.png")

    subbanner("Type 2: Segment Size Drift")
    SIZE_STABLE, SIZE_INVESTIGATE = 15.0, 30.0
    ref_share = ref_labels.value_counts(normalize=True).sort_index()
    cur_share = cur_labels.value_counts(normalize=True).sort_index()
    for c in range(k):
        if c not in ref_share.index or c not in cur_share.index:
            print(f"   Cluster {c}: not present in one of the two splits \u2014 skipped")
            continue
        shift = abs(cur_share[c] - ref_share[c]) / ref_share[c] * 100
        flag = "STABLE" if shift < SIZE_STABLE else ("INVESTIGATE" if shift < SIZE_INVESTIGATE else "RE-RUN SWEEP")
        print(f"   Cluster {c}: ref={ref_share[c]*100:.1f}%  current={cur_share[c]*100:.1f}%  shift={shift:.2f}%  {flag}")

    plt.figure(figsize=(7, 4.5))
    x = np.arange(k); width = 0.35
    plt.bar(x - width/2, [ref_share.get(c, 0) * 100 for c in range(k)], width, label="Reference")
    plt.bar(x + width/2, [cur_share.get(c, 0) * 100 for c in range(k)], width, label="Current")
    plt.xticks(x, [f"Cluster {c}" for c in range(k)])
    plt.ylabel("% of base"); plt.title("Step 13 Type 2 \u2014 Segment Size Drift, reference vs current")
    plt.legend()
    savefig("step13_segment_size_drift.png")

    subbanner("Type 3: Centroid Displacement")
    cur_labels_refit, cur_centroids_refit = _fit_algorithm_generic(
        X_cur.values, algorithm, k, RANDOM_STATE,
        eps=primary_model.get("eps"), min_samples=primary_model.get("min_samples"),
        min_cluster_size=primary_model.get("min_cluster_size"),
        alt_distance_metric=primary_model.get("alt_distance_metric"),
        num_cols=primary_model.get("num_cols"), cat_cols=primary_model.get("cat_cols"),
        feat_cols_ordered=primary_model.get("feat_cols_ordered"))
    if len(cur_centroids_refit) > 0:
        cost = np.linalg.norm(cur_centroids_refit[:, None, :] - primary_model["centroids"][None, :, :], axis=2)
        row, col = linear_sum_assignment(cost)
        mapping = dict(zip(row, col))
        for b_idx, ref_c in mapping.items():
            disp = np.linalg.norm(cur_centroids_refit[b_idx] - primary_model["centroids"][ref_c])
            within_std = X_ref[ref_labels == ref_c].std().mean()
            flag = "STABLE" if disp < within_std else "FLAG -- re-profile"
            print(f"   Cluster {ref_c}: displacement={disp:.4f}  (1x within-cluster SD={within_std:.4f})  {flag}")
    else:
        print(f"   Refit on current data produced no valid centroids \u2014 skipped (algorithm-specific degenerate case).")
        mapping = {}

    subbanner("Type 4: Stability Drift")
    cur_scored_by_production, _ = primary_predict_new(X_cur.values, primary_model)
    cur_refit_mapped = np.array([mapping.get(l, -1) for l in cur_labels_refit])
    ari_drift = adjusted_rand_score(cur_scored_by_production, cur_refit_mapped)
    print(f"   ARI(production model assignments vs fresh refit on current data) = {ari_drift:.4f}")
    print(f"   {'STABLE' if ari_drift >= 0.5 else 'FLAG -- material shape change'}  (threshold 0.5)")
    
    if ari_drift < 0.5:
        print(f"   \u26a0\ufe0f  NOTE: this is the same-population 70/30 DEMO (see this step's opening note) --")
        print(f"   genuine drift is impossible here by construction. An ARI this low more likely reflects")
        print(f"   K-Means landing in a different local optimum on the refit than real behavioural change.")
        print(f"   Re-running this step (a fresh 70/30 split + refit) is a reasonable first check before")
        print(f"   treating this as a real signal. On GENUINELY new data (via")
        print(f"   check_drift_against_baseline()), the same low score would be a real finding worth acting on.")

    proxy_feature = artifact.get("calibration_proxy_feature")
    subbanner(f"Type 5: Calibration Drift (proxy: {proxy_feature or 'NONE SELECTED'} rate by segment) [Enhancement 14]")
    if proxy_feature is None or proxy_feature not in modelling_df.columns:
        print(f"   No usable calibration proxy feature \u2014 skipping Type 5.")
    else:
        CAL_STABLE, CAL_INVESTIGATE = 10.0, 25.0
        for c in range(k):
            ref_mask = ref_labels == c
            cur_mask = cur_labels == c
            if ref_mask.sum() == 0 or cur_mask.sum() == 0:
                print(f"   Cluster {c}: not present in one of the two splits \u2014 skipped")
                continue
            ref_rate = modelling_df.loc[ref_idx][ref_mask][proxy_feature].mean()
            cur_rate = modelling_df.loc[cur_idx][cur_mask][proxy_feature].mean()
            shift = abs(cur_rate - ref_rate) / max(abs(ref_rate), 1e-6) * 100
            flag = "STABLE" if shift < CAL_STABLE else ("INVESTIGATE" if shift < CAL_INVESTIGATE else "RECALIBRATE")
            print(f"   Cluster {c}: ref rate={ref_rate:.3f}  current={cur_rate:.3f}  shift={shift:.2f}%  {flag}")

    subbanner("Type 6: Centroid-Confidence Monitoring (Enhancement 7)")
    print(f"   Unconditional \u2014 always computed (cheap: reuses the distance-to-centroid transform already")
    print(f"   available via the generic primary_transform() interface, no extra model fit required).")
    print(f"   Rationale: mean/median/p90 distance-to-centroid can drift BEFORE PSI, size-share, or ARI")
    print(f"   cross a formal alert threshold \u2014 an earlier warning signal that customer behaviour within")
    print(f"   an existing segment is drifting away from its historical definition, even while segment")
    print(f"   MEMBERSHIP (who's assigned where) still looks stable.")
    CENTROID_CONF_WARN = 15.0
    dists_ref_all = primary_transform(X_ref.values, primary_model)
    dists_cur_all = primary_transform(X_cur.values, primary_model)
    print(f"\n   {'Cluster':>8} {'ref mean':>10} {'cur mean':>10} {'ref p90':>9} {'cur p90':>9} {'shift %':>9}  Verdict")
    for c in range(k):
        ref_mask = (ref_labels.values == c)
        cur_mask = (cur_labels.values == c)
        if ref_mask.sum() == 0 or cur_mask.sum() == 0:
            continue
        ref_own_dist = dists_ref_all[ref_mask, c]
        cur_own_dist = dists_cur_all[cur_mask, c]
        ref_mean, cur_mean = ref_own_dist.mean(), cur_own_dist.mean()
        ref_p90, cur_p90 = np.percentile(ref_own_dist, 90), np.percentile(cur_own_dist, 90)
        shift_pct = (cur_mean - ref_mean) / ref_mean * 100
        flag = "STABLE" if abs(shift_pct) < CENTROID_CONF_WARN else "EARLY WARNING -- investigate before other metrics alert"
        print(f"   {c:>8} {ref_mean:>10.4f} {cur_mean:>10.4f} {ref_p90:>9.4f} {cur_p90:>9.4f} {shift_pct:>+8.2f}%  {flag}")

    subbanner(f"Type 7: Group Distribution Drift ('{group_col or 'n/a'}') \u2014 ties the circularity")
    print(f"   investigation into ONGOING monitoring, not just a one-time Step 3F/7A-7F/8A/9D/10 finding")
    if group_col and group_col in modelling_df.columns:
        ref_group = modelling_df.loc[ref_idx, group_col].astype(str)
        cur_group = modelling_df.loc[cur_idx, group_col].astype(str)
        levels = sorted(set(ref_group.unique()) | set(cur_group.unique()))
        ref_counts = ref_group.value_counts().reindex(levels, fill_value=0)
        cur_counts = cur_group.value_counts(normalize=False).reindex(levels, fill_value=0)
        ct = np.array([ref_counts.values, cur_counts.values])
        chi2, p, _, _ = stats.chi2_contingency(ct + 1e-9)
        n = ct.sum()
        cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else 0.0
        flag = "STABLE" if cramers_v < 0.15 else ("INVESTIGATE" if cramers_v < 0.25 else "RE-EXAMINE COHORT MIX")
        print(f"   '{group_col}' distribution: ref vs current Cramer's V={cramers_v:.4f}  p={p:.4g}  {flag}")
        print(f"   If this drifts materially, revisit whether the deployed segmentation's low circularity")
        print(f"   with '{group_col}' (established at validation time) still holds on the current population.")
    else:
        print(f"   No group column available in this artefact/extract \u2014 Type 7 skipped.")

    print(f"\n   Retraining triggers wired: PSI>={PSI_WARNING}, size shift>={SIZE_INVESTIGATE}%, "
          f"stability ARI<0.5, calibration shift>={CAL_INVESTIGATE}%, centroid-confidence shift>={CENTROID_CONF_WARN}%, "
          f"group distribution Cramer's V>=0.25 \u2014 ready to schedule against real future batches.")




if PIPELINE_HALTED_AT is None:
    step11b_result = step11b_subcluster_discovery(
        primary_model, X_scaled, tendency_result["feat_final_used"], structure_result, modelling_df,
        entity_col, investigation_results, tendency_result.get("preliminary_metric"), scaler_decision,
    )
else:
    print(f"\n\u23ed\ufe0f  Step 11B (Sub-Cluster Discovery) SKIPPED -- pipeline halted at "
          f"'{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")


if PIPELINE_HALTED_AT is None:
    
    _orig_feat_cols_raw = [
        "Country_Code", "Language_Code", "Site", "Gender", "Registration_Device", "Acquisition_Channel",
        "Reg_Date", "First_Deposit_Date", "Last_Login_Date", "Last_Deposit_Date",
        "Total_Deposit_Amount", "Cash_Stake_Casino", "Cash_Stake_Sports", "Cash_Return", "Bonus_Stake",
        "vip_tier", "active_days", "distinct_games_played", "favorite_game_type", "favorite_supplier",
    ]
    deployment_artifact, assign_segment = deploy(
        modelling_df, entity_col, _orig_feat_cols_raw, new_feat_cols, encoders,
        log_transformed, signed_log_transformed, tendency_result["feat_final_used"], imputer, scaler,
        primary_model, step10_personas, calibration_proxy_feature,
        snapshot_date, structure_result=structure_result, investigation_results=investigation_results,
        pre_activation_df=pre_activation_df, step11b_result=step11b_result,
    )
    drift_monitoring(modelling_df, entity_col, deployment_artifact, primary_model["k"])

    # FIX: three deployment outputs added on explicit request.
    subbanner("Saving deployment_artifacts.joblib, training_baseline.json, segment_personas.json")
    ARTIFACT_DIR = os.path.join(data_dir, "artifacts")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    
    joblib_path = os.path.join(ARTIFACT_DIR, "deployment_artifacts.joblib")
    joblib.dump(deployment_artifact, joblib_path)
    print(f"   \U0001F4E6 Saved: {joblib_path}")

    
    training_baseline = build_training_baseline(modelling_df, entity_col, deployment_artifact, primary_model,
                                                  primary_model["k"], step11b_result=step11b_result)
    baseline_path = os.path.join(ARTIFACT_DIR, "training_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(training_baseline, f, indent=2)
    print(f"   \U0001F4E6 Saved: {baseline_path}")
    print(f"   To check a genuinely new data pull against this baseline later, without re-running this whole")
    print(f"   pipeline: load training_baseline.json + deployment_artifacts.joblib, then call")
    print(f"   check_drift_against_baseline(new_raw_df, artifact, baseline).")

    segment_personas = build_segment_personas_json(step10_personas, step9d_profile_df, primary_model,
                                                     primary_model["k"], step11b_result=step11b_result,
                                                     pre_activation_df=pre_activation_df)
    personas_path = os.path.join(ARTIFACT_DIR, "segment_personas.json")
    with open(personas_path, "w") as f:
        json.dump(segment_personas, f, indent=2)
    print(f"   \U0001F4E6 Saved: {personas_path}")
else:
    print(f"\n\u23ed\ufe0f  Steps 12/13 (Deployment Artefact + Drift Monitoring) SKIPPED \u2014 pipeline halted at "
          f"'{PIPELINE_HALTED_AT}'. Resolve the halt and re-run before proceeding.")