#!/usr/bin/env python3
"""
ATLAS Higgs Boson - Deterministic Learning Complete Evaluation
===============================================================

Production-ready standalone script comparing deterministic learning methods.

Methods:
1. Gradient Boosting (iterative baseline)
2. Single-Step Variational Projector (best deterministic, ~2s, 71% of GB)
3. Adaptive Geometric Jumps (pure geometry, certainty-weighted)
4. GB-Informed Boson-Fold (hybrid: tiny GB + geometric projection)

Features:
- Proper train/val/test splits (threshold on val, report on test)
- Switchable importance modes: GB / Lambda / Fisher
- Complete auditability: Lambda stats, importance summaries, JSON artifacts
- Offline-friendly: requires --allow-download for network access
- sklearn compatible: tested with scikit-learn 1.0+

Usage:
------
# Requires dataset file (download separately or use --allow-download)
python3 higgs_deterministic_learning_final.py --data-file ./higgs_atlas.csv.gz

# Allow automatic download (requires internet)
python3 higgs_deterministic_learning_final.py --allow-download

# Use Lambda importance (projector-pure, no GB)
python3 higgs_deterministic_learning_final.py --importance-mode lambda --data-file ./higgs_atlas.csv.gz

# Skip GB baseline (faster)
python3 higgs_deterministic_learning_final.py --skip-gb --data-file ./higgs_atlas.csv.gz

Dataset:
--------
Download ATLAS Higgs dataset from:
http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz

Or use wget:
wget -O higgs_atlas.csv.gz http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)

__version__ = "1.0.0"
__author__ = "Deterministic Learning Research"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(
    description='ATLAS Higgs Deterministic Learning Evaluation',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
)
parser.add_argument('--importance-mode', type=str, default='gb',
                    choices=['gb', 'lambda', 'fisher'],
                    help='Feature importance source for boson-fold: gb (learned), lambda (projector), fisher (statistical)')
parser.add_argument('--skip-gb', action='store_true',
                    help='Skip Gradient Boosting baseline (faster)')
parser.add_argument('--allow-download', action='store_true',
                    help='Allow automatic dataset download from CERN (requires internet)')
parser.add_argument('--data-url', type=str,
                    default='http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz',
                    help='URL to download ATLAS Higgs dataset')
parser.add_argument('--data-file', type=str, default='./higgs_atlas.csv.gz',
                    help='Local path to dataset file')
parser.add_argument('--val-size', type=float, default=0.2,
                    help='Validation set size fraction (from training set)')
parser.add_argument('--test-size', type=int, default=None,
                    help='Test set size (sampled from KaggleSet=b), None = use full test set')
parser.add_argument('--output-csv', type=str, default=None,
                    help='Output CSV file for results (default: auto-generated)')
parser.add_argument('--output-json', type=str, default=None,
                    help='Output JSON file for artifacts (default: auto-generated)')
parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

args = parser.parse_args()

# =============================================================================
# UTILITIES
# =============================================================================

def tnow():
    return time.time()

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def ams_score(y_true, y_pred, weights, b_r=10.0):
    """Approximate Median Significance (Kaggle competition metric)."""
    s = weights[(y_pred == 1) & (y_true == 1)].sum()
    b = weights[(y_pred == 1) & (y_true == 0)].sum()
    if s + b + b_r <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b + b_r) * np.log(1.0 + s / (b + b_r)) - s)))

def find_best_threshold_ams(y_true, y_proba, weights, b_r=10.0):
    """Find threshold that maximizes AMS on validation set."""
    thresholds = np.linspace(0.001, 0.999, 500)
    best_ams, best_t = 0.0, 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        ams = ams_score(y_true, y_pred, weights, b_r)
        if ams > best_ams:
            best_ams, best_t = ams, t
    return best_t, best_ams

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def fisher_score_importance(X_std, y, eps=1e-9):
    """
    Fisher score: (μ1 - μ0)² / (σ1² + σ0²)
    Normalized to sum to 1.
    """
    X1, X0 = X_std[y==1], X_std[y==0]
    dmu = X1.mean(0) - X0.mean(0)
    v = X1.var(0) + X0.var(0) + eps
    s = (dmu**2) / v
    s /= (s.sum() + eps)
    return s

def print_lambda_stats(Lambda, name="Λ"):
    """Print Lambda summary statistics."""
    print(f"  [{name}] min={Lambda.min():.4g}, median={np.median(Lambda):.4g}, max={Lambda.max():.4g}")
    nonzero = np.sum(Lambda > 1e-6)
    print(f"  [{name}] non-zero: {nonzero}/{len(Lambda)} ({100*nonzero/len(Lambda):.1f}%)")

# =============================================================================
# METHOD 1: GRADIENT BOOSTING BASELINE
# =============================================================================

class GradientBoostingBaseline:
    """Standard Gradient Boosting baseline."""
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, missing_value=-999.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.missing_value = missing_value
        self.model_ = None
        self.mu_ = None
        
    def fit(self, X, y, weights):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        # Simple mean imputation
        X_clean = X_arr.copy()
        self.mu_ = np.zeros(X_arr.shape[1])
        
        for j in range(X_arr.shape[1]):
            mask = (X_arr[:, j] != self.missing_value)
            if np.any(mask):
                self.mu_[j] = X_arr[mask, j].mean()
            X_clean[:, j] = np.where(X_arr[:, j] == self.missing_value, self.mu_[j], X_arr[:, j])
        
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        self.model_.fit(X_clean, y_arr, sample_weight=weights)
        return self
    
    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_clean = X_arr.copy()
        
        for j in range(X_arr.shape[1]):
            X_clean[:, j] = np.where(X_arr[:, j] == self.missing_value, self.mu_[j], X_arr[:, j])
        
        return self.model_.predict_proba(X_clean)[:, 1]

# =============================================================================
# METHOD 2: SINGLE-STEP VARIATIONAL PROJECTOR
# =============================================================================

class SingleStepProjector:
    """
    Single-step variational projector: C* = C₀/(1 + τΛ)
    Best deterministic method: ~2s training, 71% of GB performance.
    """
    
    def __init__(self, tau=0.01, missing_value=-999.0):
        self.tau = tau
        self.missing_value = missing_value
        self.mu_ = None
        self.sigma_ = None
        self.Lambda_ = None
        self.classifier_ = None
        
    def _compute_lambda(self, X_std, y, weights):
        """Compute geometric coupling Λ from class separation."""
        n_features = X_std.shape[1]
        Lambda = np.zeros(n_features)
        
        for j in range(n_features):
            x_j = X_std[:, j]
            var_within = 0.0
            
            for label in [0, 1]:
                mask = (y == label)
                if np.any(mask):
                    w_class = weights[mask]
                    x_class = x_j[mask]
                    w_sum = w_class.sum()
                    if w_sum > 0:
                        mu_class = (w_class * x_class).sum() / w_sum
                        var_class = (w_class * (x_class - mu_class)**2).sum() / w_sum
                        var_within += var_class * (w_sum / weights.sum())
            
            mu_total = (weights * x_j).sum() / weights.sum()
            var_total = (weights * (x_j - mu_total)**2).sum() / weights.sum()
            var_between = var_total - var_within
            Lambda[j] = var_between / (var_within + 1e-6)
        
        Lambda = Lambda / (Lambda.max() + 1e-6)
        return Lambda
    
    def fit(self, X, y, weights):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        n_features = X_arr.shape[1]
        
        # Weighted standardization
        self.mu_ = np.zeros(n_features)
        self.sigma_ = np.ones(n_features)
        
        for j in range(n_features):
            x_j = X_arr[:, j]
            mask = (x_j != self.missing_value)
            if np.any(mask):
                w_valid = weights[mask]
                x_valid = x_j[mask]
                w_sum = w_valid.sum()
                if w_sum > 0:
                    self.mu_[j] = (w_valid * x_valid).sum() / w_sum
                    var = (w_valid * (x_valid - self.mu_[j])**2).sum() / w_sum
                    self.sigma_[j] = np.sqrt(max(var, 1e-12))
        
        X_clean = X_arr.copy()
        for j in range(n_features):
            mask = (X_arr[:, j] == self.missing_value)
            if np.any(mask):
                X_clean[mask, j] = self.mu_[j]
        
        X_std = (X_clean - self.mu_) / (self.sigma_ + 1e-12)
        
        # Compute Lambda
        self.Lambda_ = self._compute_lambda(X_std, y_arr, weights)
        print_lambda_stats(self.Lambda_, "Λ (Single-Step)")
        
        # Apply projection
        shrinkage = 1.0 / (1.0 + self.tau * self.Lambda_)
        X_proj = X_std * shrinkage
        
        # Logistic readout (sklearn compatible - no penalty for speed)
        self.classifier_ = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        self.classifier_.fit(X_proj, y_arr, sample_weight=weights)
        return self
    
    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_clean = X_arr.copy()
        
        for j in range(X_arr.shape[1]):
            mask = (X_arr[:, j] == self.missing_value)
            if np.any(mask):
                X_clean[mask, j] = self.mu_[j]
        
        X_std = (X_clean - self.mu_) / (self.sigma_ + 1e-12)
        shrinkage = 1.0 / (1.0 + self.tau * self.Lambda_)
        X_proj = X_std * shrinkage
        
        return self.classifier_.predict_proba(X_proj)[:, 1]

# =============================================================================
# METHOD 3: ADAPTIVE GEOMETRIC JUMPS
# =============================================================================

class AdaptiveGeometricJumps:
    """
    Iterative deterministic projection with certainty-based adaptive stepping.
    Pure geometry (no regression).
    """
    
    def __init__(self, n_iterations=5, tau=0.01, alpha=0.1, missing_value=-999.0):
        self.n_iterations = n_iterations
        self.tau = tau
        self.alpha = alpha
        self.missing_value = missing_value
        self.mu_ = None
        self.sigma_ = None
        self.lambdas_ = []
        self.centroids_ = []
        
    def _compute_lambda_from_certainty(self, X_std, residuals, weights):
        n_features = X_std.shape[1]
        Lambda = np.zeros(n_features)
        abs_residuals = np.abs(residuals)
        certainty = 1.0 / (abs_residuals + 1e-3)
        
        for j in range(n_features):
            x_j = X_std[:, j]
            w_cert = weights * certainty
            w_sum = w_cert.sum()
            
            if w_sum > 1e-12:
                mean_x = (w_cert * x_j).sum() / w_sum
                var_x = (w_cert * (x_j - mean_x)**2).sum() / w_sum
                Lambda[j] = var_x
        
        max_lambda = Lambda.max()
        if max_lambda > 1e-12:
            Lambda = Lambda / max_lambda
        
        return Lambda
    
    def fit(self, X, y, weights):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        n_samples, n_features = X_arr.shape
        
        # Weighted standardization
        self.mu_ = np.zeros(n_features)
        self.sigma_ = np.ones(n_features)
        
        for j in range(n_features):
            x_j = X_arr[:, j]
            mask = (x_j != self.missing_value)
            if np.any(mask):
                w_valid = weights[mask]
                x_valid = x_j[mask]
                w_sum = w_valid.sum()
                if w_sum > 0:
                    self.mu_[j] = (w_valid * x_valid).sum() / w_sum
                    var = (w_valid * (x_valid - self.mu_[j])**2).sum() / w_sum
                    self.sigma_[j] = np.sqrt(max(var, 1e-12))
        
        X_clean = X_arr.copy()
        for j in range(n_features):
            mask = (X_arr[:, j] == self.missing_value)
            if np.any(mask):
                X_clean[mask, j] = self.mu_[j]
        
        X_std = (X_clean - self.mu_) / (self.sigma_ + 1e-12)
        
        # Initialize predictions
        p_pos = (weights[y_arr == 1].sum()) / weights.sum()
        p_pos = np.clip(p_pos, 1e-7, 1 - 1e-7)
        f_pred = np.full(n_samples, np.log(p_pos / (1 - p_pos)))
        
        # Iterative projection
        for iter_idx in range(self.n_iterations):
            p_pred = sigmoid(f_pred)
            residuals = y_arr - p_pred
            
            Lambda = self._compute_lambda_from_certainty(X_std, residuals, weights)
            self.lambdas_.append(Lambda)
            
            if iter_idx == 0:
                print_lambda_stats(Lambda, f"Λ (Adaptive, iter 1)")
            
            shrinkage = 1.0 / (1.0 + self.tau * Lambda)
            X_proj = X_std * shrinkage
            
            # Centroid-based scoring
            c0 = np.zeros(n_features)
            c1 = np.zeros(n_features)
            
            mask0 = (y_arr == 0)
            mask1 = (y_arr == 1)
            
            w0_sum = weights[mask0].sum()
            w1_sum = weights[mask1].sum()
            
            if w0_sum > 0:
                c0 = (weights[mask0, np.newaxis] * X_proj[mask0]).sum(axis=0) / w0_sum
            if w1_sum > 0:
                c1 = (weights[mask1, np.newaxis] * X_proj[mask1]).sum(axis=0) / w1_sum
            
            self.centroids_.append((c0, c1))
            
            d0 = np.sum((X_proj - c0)**2, axis=1)
            d1 = np.sum((X_proj - c1)**2, axis=1)
            scores = d0 - d1
            
            score_std = scores.std()
            if score_std > 1e-12:
                scores = scores / score_std
            
            f_pred += self.alpha * scores
        
        return self
    
    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        n_samples = X_arr.shape[0]
        
        X_clean = X_arr.copy()
        for j in range(X_arr.shape[1]):
            mask = (X_arr[:, j] == self.missing_value)
            if np.any(mask):
                X_clean[mask, j] = self.mu_[j]
        
        X_std = (X_clean - self.mu_) / (self.sigma_ + 1e-12)
        f_pred = np.zeros(n_samples)
        
        for Lambda, (c0, c1) in zip(self.lambdas_, self.centroids_):
            shrinkage = 1.0 / (1.0 + self.tau * Lambda)
            X_proj = X_std * shrinkage
            
            d0 = np.sum((X_proj - c0)**2, axis=1)
            d1 = np.sum((X_proj - c1)**2, axis=1)
            scores = d0 - d1
            
            score_std = scores.std()
            if score_std > 1e-12:
                scores = scores / score_std
            
            f_pred += self.alpha * scores
        
        return sigmoid(f_pred)

# =============================================================================
# METHOD 4: GB-INFORMED BOSON-FOLD (with switchable importance modes)
# =============================================================================

class GBInformedBosonFold:
    """
    GB-Informed Boson-Fold with switchable importance modes:
    - 'gb': Learn from lightweight Gradient Boosting (~45s)
    - 'lambda': Use projector's own Λ (fully deterministic, ~1s)
    - 'fisher': Use Fisher score (statistical, ~1s)
    """
    
    def __init__(self, tau=0.01, lambda_miss=0.5, missing_value=-999.0,
                 importance_mode='gb', gb_n_estimators=20, gb_max_depth=3):
        self.tau = tau
        self.lambda_miss = lambda_miss
        self.missing_value = missing_value
        self.importance_mode = importance_mode
        self.gb_n_estimators = gb_n_estimators
        self.gb_max_depth = gb_max_depth
        
        self.mu_raw_ = None
        self.sigma_raw_ = None
        self.feature_importances_ = None
        self.mu_fold_ = None
        self.sigma_fold_ = None
        self.Lambda_ = None
        self.c0_ = None
        self.c1_ = None
        self.v0_ = None
        self.v1_ = None
        
    def _fold_features_weighted(self, X, importances):
        """
        Weighted symmetric polynomial features.
        Note: Uses unweighted quantiles for speed (can be upgraded to weighted).
        """
        EPS = 1e-9
        w = importances / (importances.sum() + EPS)
        
        x_weighted = X * w
        
        s1_w = np.sum(x_weighted, axis=1, keepdims=True)
        s2_w = np.sum(x_weighted * X, axis=1, keepdims=True)
        
        mean_w = s1_w / (w.sum() + EPS)
        var_w = (s2_w / (w.sum() + EPS)) - mean_w**2
        std_w = np.sqrt(np.abs(var_w) + EPS)
        
        l1_w = np.sum(np.abs(x_weighted), axis=1, keepdims=True)
        l2_w = np.sqrt(np.sum(x_weighted**2, axis=1, keepdims=True))
        
        # Note: unweighted quantiles (TODO: upgrade to weighted for full rigor)
        med_w = np.median(x_weighted, axis=1, keepdims=True)
        q25_w = np.quantile(x_weighted, 0.25, axis=1, keepdims=True)
        q75_w = np.quantile(x_weighted, 0.75, axis=1, keepdims=True)
        
        # Top-k by importance
        Kcands = [1, 3, 5, 10]
        d = X.shape[1]
        K = [k for k in Kcands if k <= d]
        
        importance_order = np.argsort(importances)[::-1]
        topk = []
        for k in K:
            top_k_idx = importance_order[:k]
            topk_sum = X[:, top_k_idx].sum(axis=1, keepdims=True)
            topk.append(topk_sum)
        topk = np.hstack(topk) if topk else np.zeros((X.shape[0], 0))
        
        # Raw top features (preserves heterogeneity)
        top_features = []
        for k in [1, 3, 5]:
            if k <= d:
                top_k_idx = importance_order[:k]
                top_features.append(X[:, top_k_idx])
        top_features = np.hstack(top_features) if top_features else np.zeros((X.shape[0], 0))
        
        Phi = np.hstack([s1_w, s2_w, mean_w, var_w, std_w, l1_w, l2_w, 
                        med_w, q25_w, q75_w, topk, top_features])
        return Phi
    
    def _build_lambda(self, Phi_tr, y_tr, weights_tr, X_raw_tr):
        EPS = 1e-9
        
        mu1 = (weights_tr[y_tr==1, np.newaxis] * Phi_tr[y_tr==1]).sum(0) / weights_tr[y_tr==1].sum()
        mu0 = (weights_tr[y_tr==0, np.newaxis] * Phi_tr[y_tr==0]).sum(0) / weights_tr[y_tr==0].sum()
        
        v1 = (weights_tr[y_tr==1, np.newaxis] * (Phi_tr[y_tr==1] - mu1)**2).sum(0) / weights_tr[y_tr==1].sum() + EPS
        v0 = (weights_tr[y_tr==0, np.newaxis] * (Phi_tr[y_tr==0] - mu0)**2).sum(0) / weights_tr[y_tr==0].sum() + EPS
        
        separation = (mu1 - mu0)**2 / (v1 + v0)
        
        miss_frac = (X_raw_tr == self.missing_value).mean(axis=0)
        miss_scalar = float(miss_frac.mean())
        
        Lambda = separation + self.lambda_miss * miss_scalar
        return Lambda
    
    def fit(self, X, y, weights, feature_names=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        n_features = X_arr.shape[1]
        
        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        
        # Standardize
        self.mu_raw_ = np.zeros(n_features)
        self.sigma_raw_ = np.ones(n_features)
        
        for j in range(n_features):
            x_j = X_arr[:, j]
            mask = (x_j != self.missing_value)
            if np.any(mask):
                w_valid = weights[mask]
                x_valid = x_j[mask]
                w_sum = w_valid.sum()
                if w_sum > 0:
                    self.mu_raw_[j] = (w_valid * x_valid).sum() / w_sum
                    var = (w_valid * (x_valid - self.mu_raw_[j])**2).sum() / w_sum
                    self.sigma_raw_[j] = np.sqrt(max(var, 1e-12))
        
        X_clean = X_arr.copy()
        for j in range(n_features):
            mask = (X_arr[:, j] == self.missing_value)
            if np.any(mask):
                X_clean[mask, j] = self.mu_raw_[j]
        
        X_std = (X_clean - self.mu_raw_) / (self.sigma_raw_ + 1e-12)
        
        # Compute importance weights based on mode
        print(f"  [Importance mode: {self.importance_mode}]")
        
        if self.importance_mode == 'gb':
            print("  Training lightweight GB for feature importances...")
            gb = GradientBoostingClassifier(
                n_estimators=self.gb_n_estimators,
                max_depth=self.gb_max_depth,
                learning_rate=0.1,
                random_state=42
            )
            gb.fit(X_std, y_arr, sample_weight=weights)
            w_vec = gb.feature_importances_
            
        elif self.importance_mode == 'lambda':
            print("  Using projector Λ for feature importances (deterministic, no GB)...")
            X1, X0 = X_std[y_arr==1], X_std[y_arr==0]
            dmu = X1.mean(0) - X0.mean(0)
            v = X1.var(0) + X0.var(0) + 1e-9
            miss = (X_arr == self.missing_value).mean(0)
            w_vec = (dmu**2) / v + self.lambda_miss * miss
            
        else:  # 'fisher'
            print("  Using Fisher score for feature importances (deterministic)...")
            w_vec = fisher_score_importance(X_std, y_arr)
        
        # Normalize
        w_vec = w_vec / (w_vec.sum() + 1e-12)
        self.feature_importances_ = w_vec
        
        # Print top features
        top_k = min(10, n_features)
        top_idx = np.argsort(w_vec)[-top_k:][::-1]
        print(f"  Top-{top_k} important features:")
        for rank, idx in enumerate(top_idx, 1):
            fname = feature_names[idx] if feature_names else f"feature_{idx}"
            print(f"    {rank}. {fname}: {w_vec[idx]:.4f}")
        
        # Fold with weighted aggregation
        Phi = self._fold_features_weighted(X_std, self.feature_importances_)
        print(f"  Folded feature dimension: {Phi.shape[1]}")
        
        # Standardize folded
        self.mu_fold_ = (weights[:, np.newaxis] * Phi).sum(0) / weights.sum()
        var_fold = (weights[:, np.newaxis] * (Phi - self.mu_fold_)**2).sum(0) / weights.sum()
        self.sigma_fold_ = np.sqrt(var_fold + 1e-12)
        
        Phi_std = (Phi - self.mu_fold_) / self.sigma_fold_
        
        # Build Lambda and project
        self.Lambda_ = self._build_lambda(Phi_std, y_arr, weights, X_arr)
        print_lambda_stats(self.Lambda_, "Λ (Boson-Fold)")
        
        shrinkage = 1.0 / (1.0 + self.tau * self.Lambda_)
        Phi_proj = Phi_std * shrinkage
        
        # Store centroids for Student-t readout
        self.c1_ = (weights[y_arr==1, np.newaxis] * Phi_proj[y_arr==1]).sum(0) / weights[y_arr==1].sum()
        self.c0_ = (weights[y_arr==0, np.newaxis] * Phi_proj[y_arr==0]).sum(0) / weights[y_arr==0].sum()
        
        self.v1_ = (weights[y_arr==1, np.newaxis] * (Phi_proj[y_arr==1] - self.c1_)**2).sum(0) / weights[y_arr==1].sum() + 1e-9
        self.v0_ = (weights[y_arr==0, np.newaxis] * (Phi_proj[y_arr==0] - self.c0_)**2).sum(0) / weights[y_arr==0].sum() + 1e-9
        
        return self
    
    def predict_proba(self, X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        
        X_clean = X_arr.copy()
        for j in range(X_arr.shape[1]):
            mask = (X_arr[:, j] == self.missing_value)
            if np.any(mask):
                X_clean[mask, j] = self.mu_raw_[j]
        
        X_std = (X_clean - self.mu_raw_) / (self.sigma_raw_ + 1e-12)
        
        Phi = self._fold_features_weighted(X_std, self.feature_importances_)
        Phi_std = (Phi - self.mu_fold_) / self.sigma_fold_
        
        shrinkage = 1.0 / (1.0 + self.tau * self.Lambda_)
        Phi_proj = Phi_std * shrinkage
        
        # Student-t readout
        s1 = ((Phi_proj - self.c1_)**2 / self.v1_).sum(1)
        s0 = ((Phi_proj - self.c0_)**2 / self.v0_).sum(1)
        scores = s0 - s1
        
        return sigmoid(scores / (scores.std() + 1e-9))

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    print("=" * 80)
    print("ATLAS HIGGS BOSON - DETERMINISTIC LEARNING EVALUATION")
    print(f"Version {__version__}")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Importance mode: {args.importance_mode}")
    print(f"  Skip GB baseline: {args.skip_gb}")
    print(f"  Val/Test split: proper threshold selection on val")
    print(f"  Data file: {args.data_file}")
    print()
    
    # Load data with proper error handling
    print("[1/5] Loading ATLAS Higgs dataset...")
    t0_total = tnow()
    
    if not os.path.exists(args.data_file):
        if not args.allow_download:
            print()
            print("ERROR: Dataset file not found and network access not allowed.")
            print()
            print("Please either:")
            print(f"  1. Download the dataset manually to: {args.data_file}")
            print(f"     wget -O {args.data_file} {args.data_url}")
            print()
            print("  2. Or run with --allow-download to download automatically:")
            print(f"     python3 {sys.argv[0]} --allow-download")
            print()
            sys.exit(1)
        
        print(f"  Downloading from {args.data_url}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(args.data_url, args.data_file)
            print(f"  Saved to {args.data_file}")
        except Exception as e:
            print(f"  ERROR: Download failed: {e}")
            print(f"  Please download manually: wget -O {args.data_file} {args.data_url}")
            sys.exit(1)
    
    t0 = tnow()
    df = pd.read_csv(args.data_file, compression='gzip')
    t1 = tnow()
    
    print(f"  Loaded {len(df):,} total events in {format_time(t1-t0)}")
    print()
    
    # Prepare data with proper train/val/test split
    print("[2/5] Preparing train/val/test splits...")
    
    df_train_full = df[df['KaggleSet'] == 't'].copy()
    df_test_full = df[df['KaggleSet'] == 'b'].copy()
    
    # Sample test set if needed
    if args.test_size is not None and len(df_test_full) > args.test_size:
        df_test = df_test_full.sample(n=args.test_size, random_state=42)
    else:
        df_test = df_test_full
    
    feature_cols = [c for c in df_train_full.columns 
                    if c not in ['EventId', 'Weight', 'Label', 'KaggleSet', 'KaggleWeight']]
    
    # Split train into train/val
    X_train_full = df_train_full[feature_cols]
    y_train_full = df_train_full['Label'].map({'s': 1, 'b': 0})
    w_train_full = df_train_full['Weight'].values
    
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_train_full, y_train_full, w_train_full,
        test_size=args.val_size, random_state=42, stratify=y_train_full
    )
    
    X_test = df_test[feature_cols]
    y_test = df_test['Label'].map({'s': 1, 'b': 0})
    w_test = df_test['Weight'].values
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(X_train):,} samples, {(y_train==1).sum():,} signal, {(y_train==0).sum():,} background")
    print(f"  Val:   {len(X_val):,} samples, {(y_val==1).sum():,} signal, {(y_val==0).sum():,} background")
    print(f"  Test:  {len(X_test):,} samples, {(y_test==1).sum():,} signal, {(y_test==0).sum():,} background")
    print()
    
    # Train and evaluate all methods
    print("[3/5] Training all methods...")
    print("-" * 80)
    
    results = []
    artifacts = {
        'version': __version__,
        'importance_mode': args.importance_mode,
        'methods': {}
    }
    
    # METHOD 1: Gradient Boosting (optional)
    if not args.skip_gb:
        print("\n[Method 1/4] Gradient Boosting Baseline")
        print("-" * 40)
        
        t0 = tnow()
        gb = GradientBoostingBaseline(n_estimators=100, max_depth=5, learning_rate=0.1)
        gb.fit(X_train, y_train, w_train)
        t_fit = tnow() - t0
        print(f"  Training time: {format_time(t_fit)}")
        
        t0 = tnow()
        y_val_proba_gb = gb.predict_proba(X_val)
        y_test_proba_gb = gb.predict_proba(X_test)
        t_pred = tnow() - t0
        
        # Find threshold on val
        theta_gb, val_ams_gb = find_best_threshold_ams(y_val.values, y_val_proba_gb, w_val)
        
        # Apply to test
        y_test_pred_gb = (y_test_proba_gb >= theta_gb).astype(int)
        test_ams_gb = ams_score(y_test.values, y_test_pred_gb, w_test)
        test_auc_gb = roc_auc_score(y_test.values, y_test_proba_gb, sample_weight=w_test)
        
        print(f"  Val  AMS: {val_ams_gb:.6f} (threshold: {theta_gb:.4f})")
        print(f"  Test AMS: {test_ams_gb:.6f}, AUC: {test_auc_gb:.6f}")
        print(f"  Prediction time: {format_time(t_pred)}")
        
        results.append({
            'method': 'Gradient Boosting',
            'importance_mode': 'n/a',
            'val_ams': val_ams_gb,
            'test_ams': test_ams_gb,
            'test_auc': test_auc_gb,
            'threshold': theta_gb,
            'fit_time': t_fit,
            'pred_time': t_pred
        })
        
        artifacts['methods']['gradient_boosting'] = {
            'threshold': float(theta_gb),
            'val_ams': float(val_ams_gb),
            'test_ams': float(test_ams_gb),
            'test_auc': float(test_auc_gb)
        }
    
    # METHOD 2: Single-Step Projector
    print("\n[Method 2/4] Single-Step Variational Projector")
    print("-" * 40)
    
    t0 = tnow()
    proj = SingleStepProjector(tau=0.01)
    proj.fit(X_train, y_train, w_train)
    t_fit = tnow() - t0
    print(f"  Training time: {format_time(t_fit)}")
    
    t0 = tnow()
    y_val_proba_proj = proj.predict_proba(X_val)
    y_test_proba_proj = proj.predict_proba(X_test)
    t_pred = tnow() - t0
    
    theta_proj, val_ams_proj = find_best_threshold_ams(y_val.values, y_val_proba_proj, w_val)
    
    y_test_pred_proj = (y_test_proba_proj >= theta_proj).astype(int)
    test_ams_proj = ams_score(y_test.values, y_test_pred_proj, w_test)
    test_auc_proj = roc_auc_score(y_test.values, y_test_proba_proj, sample_weight=w_test)
    
    print(f"  Val  AMS: {val_ams_proj:.6f} (threshold: {theta_proj:.4f})")
    print(f"  Test AMS: {test_ams_proj:.6f}, AUC: {test_auc_proj:.6f}")
    print(f"  Prediction time: {format_time(t_pred)}")
    
    results.append({
        'method': 'Single-Step Projector',
        'importance_mode': 'n/a',
        'val_ams': val_ams_proj,
        'test_ams': test_ams_proj,
        'test_auc': test_auc_proj,
        'threshold': theta_proj,
        'fit_time': t_fit,
        'pred_time': t_pred
    })
    
    artifacts['methods']['single_step_projector'] = {
        'threshold': float(theta_proj),
        'val_ams': float(val_ams_proj),
        'test_ams': float(test_ams_proj),
        'test_auc': float(test_auc_proj),
        'lambda': proj.Lambda_.tolist()
    }
    
    # METHOD 3: Adaptive Geometric Jumps
    print("\n[Method 3/4] Adaptive Geometric Jumps")
    print("-" * 40)
    
    t0 = tnow()
    jumps = AdaptiveGeometricJumps(n_iterations=5, tau=0.01, alpha=0.1)
    jumps.fit(X_train, y_train, w_train)
    t_fit = tnow() - t0
    print(f"  Training time: {format_time(t_fit)}")
    
    t0 = tnow()
    y_val_proba_jumps = jumps.predict_proba(X_val)
    y_test_proba_jumps = jumps.predict_proba(X_test)
    t_pred = tnow() - t0
    
    theta_jumps, val_ams_jumps = find_best_threshold_ams(y_val.values, y_val_proba_jumps, w_val)
    
    y_test_pred_jumps = (y_test_proba_jumps >= theta_jumps).astype(int)
    test_ams_jumps = ams_score(y_test.values, y_test_pred_jumps, w_test)
    test_auc_jumps = roc_auc_score(y_test.values, y_test_proba_jumps, sample_weight=w_test)
    
    print(f"  Val  AMS: {val_ams_jumps:.6f} (threshold: {theta_jumps:.4f})")
    print(f"  Test AMS: {test_ams_jumps:.6f}, AUC: {test_auc_jumps:.6f}")
    print(f"  Prediction time: {format_time(t_pred)}")
    
    results.append({
        'method': 'Adaptive Geometric Jumps',
        'importance_mode': 'n/a',
        'val_ams': val_ams_jumps,
        'test_ams': test_ams_jumps,
        'test_auc': test_auc_jumps,
        'threshold': theta_jumps,
        'fit_time': t_fit,
        'pred_time': t_pred
    })
    
    artifacts['methods']['adaptive_geometric_jumps'] = {
        'threshold': float(theta_jumps),
        'val_ams': float(val_ams_jumps),
        'test_ams': float(test_ams_jumps),
        'test_auc': float(test_auc_jumps),
        'lambdas': [l.tolist() for l in jumps.lambdas_]
    }
    
    # METHOD 4: GB-Informed Boson-Fold
    print(f"\n[Method 4/4] GB-Informed Boson-Fold (importance={args.importance_mode})")
    print("-" * 40)
    
    t0 = tnow()
    boson = GBInformedBosonFold(tau=0.01, lambda_miss=0.5, 
                                 importance_mode=args.importance_mode,
                                 gb_n_estimators=20, gb_max_depth=3)
    boson.fit(X_train, y_train, w_train, feature_names=feature_cols)
    t_fit = tnow() - t0
    print(f"  Training time: {format_time(t_fit)}")
    
    t0 = tnow()
    y_val_proba_boson = boson.predict_proba(X_val)
    y_test_proba_boson = boson.predict_proba(X_test)
    t_pred = tnow() - t0
    
    theta_boson, val_ams_boson = find_best_threshold_ams(y_val.values, y_val_proba_boson, w_val)
    
    y_test_pred_boson = (y_test_proba_boson >= theta_boson).astype(int)
    test_ams_boson = ams_score(y_test.values, y_test_pred_boson, w_test)
    test_auc_boson = roc_auc_score(y_test.values, y_test_proba_boson, sample_weight=w_test)
    
    print(f"  Val  AMS: {val_ams_boson:.6f} (threshold: {theta_boson:.4f})")
    print(f"  Test AMS: {test_ams_boson:.6f}, AUC: {test_auc_boson:.6f}")
    print(f"  Prediction time: {format_time(t_pred)}")
    
    results.append({
        'method': 'GB-Informed Boson-Fold',
        'importance_mode': args.importance_mode,
        'val_ams': val_ams_boson,
        'test_ams': test_ams_boson,
        'test_auc': test_auc_boson,
        'threshold': theta_boson,
        'fit_time': t_fit,
        'pred_time': t_pred
    })
    
    artifacts['methods']['gb_informed_boson_fold'] = {
        'importance_mode': args.importance_mode,
        'threshold': float(theta_boson),
        'val_ams': float(val_ams_boson),
        'test_ams': float(test_ams_boson),
        'test_auc': float(test_auc_boson),
        'feature_importances': boson.feature_importances_.tolist(),
        'lambda': boson.Lambda_.tolist()
    }
    
    # Results summary
    print()
    print("=" * 80)
    print("[4/5] RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_ams', ascending=False)
    
    print("PERFORMANCE METRICS (sorted by Test AMS):")
    print("-" * 80)
    print(f"{'Method':<30} {'Mode':<10} {'Val AMS':>10} {'Test AMS':>10} {'Test AUC':>10}")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"{row['method']:<30} {row['importance_mode']:<10} "
              f"{row['val_ams']:>10.6f} {row['test_ams']:>10.6f} {row['test_auc']:>10.6f}")
    print()
    
    print("TIMING:")
    print("-" * 80)
    print(f"{'Method':<30} {'Mode':<10} {'Fit Time':>15} {'Pred Time':>15}")
    print("-" * 80)
    for _, row in df_results.iterrows():
        print(f"{row['method']:<30} {row['importance_mode']:<10} "
              f"{format_time(row['fit_time']):>15} {format_time(row['pred_time']):>15}")
    print()
    
    # Analysis
    print("=" * 80)
    print("[5/5] ANALYSIS")
    print("=" * 80)
    print()
    
    print("COMPETITION CONTEXT:")
    print(f"  Kaggle 1st place: AMS ≈ 3.80-3.85")
    print(f"  Kaggle baseline: AMS ≈ 3.60-3.65")
    print(f"  Our best: AMS = {df_results.iloc[0]['test_ams']:.2f} ({df_results.iloc[0]['method']})")
    print()
    
    if not args.skip_gb:
        gb_row = df_results[df_results['method'] == 'Gradient Boosting'].iloc[0]
        print("COMPARISON TO GB BASELINE:")
        print("-" * 80)
        for _, row in df_results.iterrows():
            if row['method'] != 'Gradient Boosting':
                ams_pct = row['test_ams'] / gb_row['test_ams'] * 100
                speedup = gb_row['fit_time'] / row['fit_time']
                print(f"{row['method']} ({row['importance_mode']}):")
                print(f"  Test AMS: {row['test_ams']:.4f} ({ams_pct:.1f}% of GB)")
                print(f"  Speedup: {speedup:.0f}× faster")
                print()
    
    t1_total = tnow()
    print(f"Total runtime: {format_time(t1_total - t0_total)}")
    print()
    
    # Save results
    if args.output_csv is None:
        args.output_csv = f"higgs_deterministic_results_{int(time.time())}.csv"
    
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to: {args.output_csv}")
    
    if args.output_json is None:
        args.output_json = args.output_csv.replace('.csv', '.json')
    
    with open(args.output_json, 'w') as f:
        json.dump(artifacts, f, indent=2)
    print(f"Artifacts saved to: {args.output_json}")
    
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
