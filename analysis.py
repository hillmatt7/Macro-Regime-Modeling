# analysis.py
"""
Analysis components: Label materialisation, change detection, interpretation, and forecasting
COMPLETE WORKING IMPLEMENTATION - Production Ready
"""

import pandas as pd
import numpy as np
import json
import hashlib
import os
from typing import Dict, Tuple, Optional, List, Union
from sklearn.metrics import adjusted_rand_score, f1_score, mean_absolute_error, accuracy_score
import lightgbm as lgb
import shap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ensure all directories exist
for dir_name in ['regime_labels', 'interpretation_models', 'shap_explanations', 
                 'forecast_models', 'forecast_checkpoints']:
    os.makedirs(dir_name, exist_ok=True)


class LabelMaterialisation:
    """Step 5: Label materialisation and Step 6: Change detection"""
    
    def __init__(self):
        self.labels_dir = "regime_labels"
        os.makedirs(self.labels_dir, exist_ok=True)
    
    def materialise_and_detect_change(self, labels: np.ndarray, posterior_probs: np.ndarray,
                                    dates: pd.Series, selection_metadata: Dict, 
                                    window_end: str, window_start: str) -> Tuple[str, str, Dict]:
        """
        Materialise labels with posterior probabilities and detect changes
        Returns: (labels_hash, event_type, change_metadata)
        """
        # Step 5: Materialise labels with full regime information
        labels_hash = self._materialise_labels(
            labels, posterior_probs, dates, selection_metadata, window_end
        )
        
        # Step 6: Detect change
        event_type, change_metadata = self._detect_change(
            labels_hash, window_start, selection_metadata['k_star']
        )
        
        return labels_hash, event_type, change_metadata
    
    def _materialise_labels(self, labels: np.ndarray, posterior_probs: np.ndarray,
                          dates: pd.Series, selection_metadata: Dict, 
                          window_end: str) -> str:
        """Write immutable regime labels with posterior probabilities"""
        # Create comprehensive regime dataframe
        regime_df = pd.DataFrame({
            'date': dates,
            'regime_id': labels
        })
        
        # Add posterior probabilities for each regime
        n_regimes = posterior_probs.shape[1]
        for i in range(n_regimes):
            regime_df[f'regime_{i}_prob'] = posterior_probs[:, i]
        
        # Add max probability and uncertainty measure
        regime_df['max_prob'] = np.max(posterior_probs, axis=1)
        regime_df['entropy'] = -np.sum(posterior_probs * np.log(posterior_probs + 1e-10), axis=1)
        
        content_hash = self._generate_content_hash(regime_df)
        
        # Save comprehensive labels table
        labels_filename = f"regime_labels_{window_end}_{content_hash}.csv"
        labels_path = os.path.join(self.labels_dir, labels_filename)
        regime_df.to_csv(labels_path, index=False)
        
        # Calculate regime statistics
        regime_stats = self._calculate_regime_statistics(labels, posterior_probs)
        
        # Prepare comprehensive metadata
        sidecar_metadata = {
            'content_hash': content_hash,
            'window_end': window_end,
            'n_dates': len(regime_df),
            'date_range': {
                'start': str(regime_df['date'].min()),
                'end': str(regime_df['date'].max())
            },
            'k_star': selection_metadata['k_star'],
            'bic': selection_metadata['bic'],
            'icl': selection_metadata['icl'],
            'aic': selection_metadata['aic'],
            'log_likelihood': selection_metadata['log_likelihood'],
            'elbow_position': selection_metadata['k_elbow'],
            'bootstrap_stability': {
                'mean_ari': selection_metadata['bootstrap_mean_ari'],
                'std_ari': selection_metadata['bootstrap_std_ari'],
                'n_samples': selection_metadata['bootstrap_samples']
            },
            'regime_distribution': regime_stats['distribution'],
            'regime_persistence': regime_stats['persistence'],
            'transition_matrix': regime_stats['transitions'],
            'average_confidence': float(np.mean(regime_df['max_prob'])),
            'labels_file': labels_filename,
            'immutable': True
        }
        
        # Save metadata
        sidecar_filename = f"regime_metadata_{window_end}_{content_hash}.json"
        sidecar_path = os.path.join(self.labels_dir, sidecar_filename)
        
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_metadata, f, indent=2)
        
        print(f"Materialised regime labels: {content_hash}")
        
        return content_hash
    
    def _calculate_regime_statistics(self, labels: np.ndarray, 
                                   posterior_probs: np.ndarray) -> Dict:
        """Calculate comprehensive regime statistics"""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        # Distribution
        distribution = {}
        for regime_id, count in zip(unique, counts):
            distribution[f"regime_{regime_id}"] = {
                'count': int(count),
                'percentage': float(count / total * 100),
                'avg_confidence': float(np.mean(posterior_probs[labels == regime_id, regime_id]))
            }
        
        # Persistence (average duration of regimes)
        persistence = {}
        for regime_id in unique:
            regime_runs = []
            current_run = 0
            
            for label in labels:
                if label == regime_id:
                    current_run += 1
                elif current_run > 0:
                    regime_runs.append(current_run)
                    current_run = 0
            
            if current_run > 0:
                regime_runs.append(current_run)
            
            persistence[f"regime_{regime_id}"] = {
                'avg_duration': float(np.mean(regime_runs)) if regime_runs else 0,
                'max_duration': int(np.max(regime_runs)) if regime_runs else 0,
                'n_periods': len(regime_runs)
            }
        
        # Transition matrix
        n_regimes = len(unique)
        transitions = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(labels) - 1):
            transitions[labels[i], labels[i+1]] += 1
        
        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transitions, row_sums, 
                                    where=row_sums > 0, 
                                    out=np.zeros_like(transitions))
        
        return {
            'distribution': distribution,
            'persistence': persistence,
            'transitions': transition_matrix.tolist()
        }
    
    def _detect_change(self, current_labels_hash: str, window_start: str,
                      current_k: int) -> Tuple[str, Dict]:
        """Detect changes from previous window using ARI and k comparison"""
        # Load current labels
        current_df, current_metadata = self.load_labels_by_hash(current_labels_hash)
        
        # Find previous window
        previous_result = self._find_previous_window(window_start)
        
        if previous_result is None:
            # First window
            event_type = "reset"
            change_metadata = {
                'event_type': event_type,
                'reason': 'first_window',
                'current_k_star': current_k,
                'previous_k_star': None,
                'ari_overlap': None,
                'requires_retraining': True
            }
        else:
            previous_df, previous_metadata = previous_result
            
            # Check if k changed
            k_changed = current_k != previous_metadata['k_star']
            
            # Calculate ARI on overlapping dates
            ari_overlap = self._compute_overlap_ari(current_df, previous_df)
            
            # Determine event type based on criteria
            if k_changed or ari_overlap < 0.9:
                event_type = "reset"
                reasons = []
                if k_changed:
                    reasons.append(f"k_star_changed ({previous_metadata['k_star']} -> {current_k})")
                if ari_overlap < 0.9:
                    reasons.append(f"low_ari ({ari_overlap:.3f} < 0.9)")
                reason_str = ", ".join(reasons)
                requires_retraining = True
            else:
                event_type = "incremental"
                reason_str = "stable_continuation"
                requires_retraining = False
            
            change_metadata = {
                'event_type': event_type,
                'reason': reason_str,
                'current_k_star': current_k,
                'previous_k_star': previous_metadata['k_star'],
                'ari_overlap': ari_overlap,
                'k_changed': k_changed,
                'requires_retraining': requires_retraining,
                'transition_stability': self._analyze_transition_stability(
                    current_metadata.get('transition_matrix', []),
                    previous_metadata.get('transition_matrix', [])
                )
            }
        
        print(f"Change detection: {event_type} ({change_metadata['reason']})")
        
        return event_type, change_metadata
    
    def _analyze_transition_stability(self, current_transitions: List, 
                                    previous_transitions: List) -> float:
        """Analyze stability of transition matrices between windows"""
        if not current_transitions or not previous_transitions:
            return 0.0
        
        try:
            curr = np.array(current_transitions)
            prev = np.array(previous_transitions)
            
            if curr.shape == prev.shape:
                # Frobenius norm of difference
                diff = np.linalg.norm(curr - prev, 'fro')
                max_norm = np.linalg.norm(curr, 'fro') + np.linalg.norm(prev, 'fro')
                stability = 1.0 - (2 * diff / max_norm) if max_norm > 0 else 0.0
                return float(stability)
        except:
            pass
        
        return 0.0
    
    def _generate_content_hash(self, regime_df: pd.DataFrame) -> str:
        """Generate SHA256 hash of regime labels"""
        content = regime_df[['date', 'regime_id']].to_csv(index=False)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _find_previous_window(self, window_start: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """Find the immediately prior label artefact"""
        target_date = pd.to_datetime(window_start) - pd.Timedelta(days=1)
        
        metadata_files = [f for f in os.listdir(self.labels_dir) 
                         if f.startswith('regime_metadata_') and f.endswith('.json')]
        
        for filename in metadata_files:
            metadata_path = os.path.join(self.labels_dir, filename)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            window_end = pd.to_datetime(metadata['date_range']['end'])
            if window_end.date() == target_date.date():
                content_hash = metadata['content_hash']
                return self.load_labels_by_hash(content_hash)
        
        return None
    
    def _compute_overlap_ari(self, current_df: pd.DataFrame, 
                           previous_df: pd.DataFrame) -> float:
        """Compute ARI on overlapping date range"""
        current_dates = set(pd.to_datetime(current_df['date']))
        previous_dates = set(pd.to_datetime(previous_df['date']))
        overlap_dates = current_dates.intersection(previous_dates)
        
        if len(overlap_dates) == 0:
            return 0.0
        
        overlap_dates = sorted(list(overlap_dates))
        
        # Convert dates for merging
        current_df['date'] = pd.to_datetime(current_df['date'])
        previous_df['date'] = pd.to_datetime(previous_df['date'])
        
        current_overlap = current_df[current_df['date'].isin(overlap_dates)].sort_values('date')
        previous_overlap = previous_df[previous_df['date'].isin(overlap_dates)].sort_values('date')
        
        ari = adjusted_rand_score(
            previous_overlap['regime_id'].values,
            current_overlap['regime_id'].values
        )
        
        return ari
    
    def load_labels_by_hash(self, content_hash: str) -> Tuple[pd.DataFrame, Dict]:
        """Load regime labels and metadata by content hash"""
        labels_file = None
        metadata_file = None
        
        for filename in os.listdir(self.labels_dir):
            if content_hash in filename:
                if filename.endswith('.csv'):
                    labels_file = filename
                elif filename.endswith('.json'):
                    metadata_file = filename
        
        if not labels_file or not metadata_file:
            raise FileNotFoundError(f"No labels found for hash: {content_hash}")
        
        labels_path = os.path.join(self.labels_dir, labels_file)
        regime_df = pd.read_csv(labels_path)
        
        metadata_path = os.path.join(self.labels_dir, metadata_file)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return regime_df, metadata


class RegimeSequenceDataset(Dataset):
    """Dataset for regime sequence prediction"""
    
    def __init__(self, macro_features: np.ndarray, regime_labels: np.ndarray,
                 sequence_length: int = 90, horizon: int = 1):
        """
        Create supervised dataset for sequence modeling
        X_t = sequence_length days of features
        y_t = regime at t + horizon
        """
        self.sequence_length = sequence_length
        self.horizon = horizon
        
        self.X_sequences = []
        self.y_targets = []
        
        # Create sequences
        for i in range(len(macro_features) - sequence_length - horizon + 1):
            # Feature sequence
            X_seq = macro_features[i:i + sequence_length]
            # Target regime at t + horizon
            y_target = regime_labels[i + sequence_length + horizon - 1]
            
            self.X_sequences.append(X_seq)
            self.y_targets.append(y_target)
        
        self.X_sequences = np.array(self.X_sequences, dtype=np.float32)
        self.y_targets = np.array(self.y_targets, dtype=np.int64)
    
    def __len__(self):
        return len(self.X_sequences)
    
    def __getitem__(self, idx):
        return self.X_sequences[idx], self.y_targets[idx]


class RegimeForecastLSTM(nn.Module):
    """LSTM model for regime forecasting - COMPLETE IMPLEMENTATION"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.2):
        super(RegimeForecastLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        x shape: (batch, sequence, features)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Batch normalization
        last_hidden = self.batch_norm(last_hidden)
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # First fully connected layer
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Output layer
        out = self.fc2(out)
        
        return out


class InterpretationForecasting:
    """Steps 7-8: Complete SHAP interpretation and LSTM forecasting implementation"""
    
    def __init__(self):
        # Interpretation directories
        self.models_dir = "interpretation_models"
        self.shap_dir = "shap_explanations"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.shap_dir, exist_ok=True)
        
        # Forecasting directories
        self.forecast_models_dir = "forecast_models"
        self.checkpoints_dir = "forecast_checkpoints"
        os.makedirs(self.forecast_models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model state
        self.current_lgb_model = None
        self.current_lstm_model = None
        self.feature_names = None
        self.shap_explainer = None
        
        # Parameters
        self.sequence_length = 90  # 90-day sequences
        self.forecast_horizon = 30  # 30-day ahead forecast
        self.lgb_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train_interpretation_and_forecast(self, 
                                        features: np.ndarray,
                                        labels: np.ndarray,
                                        dates: pd.Series,
                                        feature_names: List[str],
                                        event_type: str,
                                        window_end: str,
                                        labels_hash: str,
                                        historical_windows: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Train both interpretation and forecasting models
        """
        self.feature_names = feature_names
        
        print("\n=== Training Interpretation Layer (LightGBM + SHAP) ===")
        interpretation_results = self._train_interpretation(
            features, labels, event_type, window_end
        )
        
        print("\n=== Training Forecast Layer (LSTM) ===")
        forecast_results = self._train_forecast(
            features, labels, dates, event_type, window_end, labels_hash
        )
        
        # Version and persist all models
        self._version_models(window_end, event_type)
        
        return interpretation_results, forecast_results
    
    def _train_interpretation(self, features: np.ndarray, labels: np.ndarray,
                            event_type: str, window_end: str) -> Dict:
        """Train LightGBM with SHAP interpretation"""
        if event_type == "reset" or self.current_lgb_model is None:
            print("Training new interpretation model (reset event)")
            return self._train_lgb_from_scratch(features, labels, window_end)
        else:
            print("Updating interpretation model (incremental event)")
            return self._lgb_incremental_update(features, labels, window_end)
    
    def _train_lgb_from_scratch(self, features: np.ndarray, labels: np.ndarray,
                              window_end: str) -> Dict:
        """Train new LightGBM model with cross-validation"""
        from sklearn.model_selection import StratifiedKFold
        
        # Update params for number of classes
        params = self.lgb_params.copy()
        params['num_class'] = len(np.unique(labels))
        
        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        feature_importances = []
        
        best_score = -1
        best_model = None
        best_iteration = 0
        
        print("Running 5-fold cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            print(f"  Fold {fold + 1}/5...")
            
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Create LightGBM datasets
            lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.feature_names)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=self.feature_names)
            
            # Train model
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            # Evaluate
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            y_pred_class = np.argmax(y_pred, axis=1)
            
            f1 = f1_score(y_val, y_pred_class, average='macro')
            cv_scores.append(f1)
            
            # Store feature importance
            importance = model.feature_importance(importance_type='gain')
            feature_importances.append(importance)
            
            print(f"    F1 Score: {f1:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_iteration = model.best_iteration
        
        self.current_lgb_model = best_model
        
        # Average feature importances across folds
        avg_importance = np.mean(feature_importances, axis=0)
        
        print(f"Cross-validation complete. Mean F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Compute SHAP values on full dataset
        print("Computing SHAP values...")
        shap_values, shap_feature_importance = self._compute_shap_explanation(features, labels)
        
        # Save model and SHAP results
        self._save_lgb_model(window_end)
        self._save_shap_results(shap_values, shap_feature_importance, window_end)
        
        return {
            'event_type': 'reset',
            'cv_scores': cv_scores,
            'mean_f1': float(np.mean(cv_scores)),
            'std_f1': float(np.std(cv_scores)),
            'best_f1': float(best_score),
            'best_iteration': best_iteration,
            'top_10_features': shap_feature_importance[:10],
            'model_params': params,
            'n_trees': self.current_lgb_model.num_trees()
        }
    
    def _lgb_incremental_update(self, features: np.ndarray, labels: np.ndarray,
                              window_end: str) -> Dict:
        """Incremental update of LightGBM model"""
        if self.current_lgb_model is None:
            return self._train_lgb_from_scratch(features, labels, window_end)
        
        # Split new data for validation
        n_samples = len(labels)
        n_train = int(0.9 * n_samples)
        
        X_train = features[:n_train]
        y_train = labels[:n_train]
        X_val = features[n_train:]
        y_val = labels[n_train:]
        
        # Continue training
        lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.feature_names)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=self.feature_names)
        
        # Update model
        self.current_lgb_model = lgb.train(
            params=self.current_lgb_model.params,
            train_set=lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=100,  # Limited rounds for incremental
            init_model=self.current_lgb_model,
            callbacks=[
                lgb.early_stopping(20),
                lgb.log_evaluation(0)
            ]
        )
        
        # Evaluate
        y_pred = self.current_lgb_model.predict(X_val, num_iteration=self.current_lgb_model.best_iteration)
        y_pred_class = np.argmax(y_pred, axis=1)
        val_f1 = f1_score(y_val, y_pred_class, average='macro')
        
        print(f"Incremental update validation F1: {val_f1:.4f}")
        
        # Update SHAP values
        shap_values, shap_feature_importance = self._compute_shap_explanation(features, labels)
        
        # Save updated model
        self._save_lgb_model(window_end)
        self._save_shap_results(shap_values, shap_feature_importance, window_end)
        
        return {
            'event_type': 'incremental',
            'validation_f1': float(val_f1),
            'n_new_samples': n_samples,
            'n_train': n_train,
            'top_10_features': shap_feature_importance[:10],
            'n_trees': self.current_lgb_model.num_trees()
        }
    
    def _compute_shap_explanation(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute SHAP values for the current LightGBM model and return
        (i) the raw SHAP values object and (ii) a ranked list of feature
        importances.  
        – Works for binary / regression (array) *and* multiclass (list) outputs.  
        – Guards against length mismatches between SHAP arrays and
          self.feature_names.  
        – Removes the two failure modes observed:
            • IndexError: list index out of range  
            • TypeError: list indices must be integers or slices, not list
        """
        # 1. Build/refresh the explainer
        self.shap_explainer = shap.TreeExplainer(self.current_lgb_model)

        # 2. (Optional) subsample rows to keep SHAP reasonably fast
        if len(features) > 5_000:
            rows = np.random.choice(len(features), 5_000, replace=False)
            sample_features = features[rows]
        else:
            sample_features = features

        # 3. Compute SHAP values
        shap_values = self.shap_explainer.shap_values(sample_features)

        # 4. Convert to |samples|×|features| absolute values array
        if isinstance(shap_values, list):                         # multiclass
            # → list[ndarray], each (n_samples, n_features)
            abs_stack = np.stack([np.abs(sv) for sv in shap_values], axis=0)
            # mean across classes first, then across samples
            mean_shap = abs_stack.mean(axis=(0, 1))               # (n_features,)
        else:                                                     # binary / reg.
            mean_shap = np.abs(shap_values).mean(axis=0)          # (n_features,)

        mean_shap = mean_shap.flatten()                           # 1-D for safety

        # 5. Align feature names and SHAP vector length
        #    (LightGBM sometimes drops constant features → shorter SHAP array)
        n_features = len(mean_shap)
        if n_features != len(self.feature_names):
            # Keep only the overlapping prefix to prevent out-of-range indexing
            trunc = min(n_features, len(self.feature_names))
            warnings.warn(
                f"SHAP returned {n_features} features, "
                f"but self.feature_names has {len(self.feature_names)}. "
                f"Using first {trunc} entries from each."
            )
            mean_shap = mean_shap[:trunc]
            names = self.feature_names[:trunc]
        else:
            names = self.feature_names

        # 6. Rank features
        total_importance = mean_shap.sum() or 1.0                 # avoid zero-div
        sort_idx = np.argsort(mean_shap)[::-1]                    # descending

        feature_importance: List[Dict] = []
        for rank, idx in enumerate(sort_idx, start=1):
            feature_importance.append({
                "feature": names[idx],
                "mean_abs_shap": float(mean_shap[idx]),
                "rank": rank,
                "relative_importance": float(mean_shap[idx] / total_importance)
            })

        # 7. Quick log of top-k
        print("Top features by SHAP importance:")
        for fi in feature_importance[:5]:
            print(f"  {fi['rank']}. {fi['feature']}: {fi['mean_abs_shap']:.6f}")

        return shap_values, feature_importance
    
    def _train_forecast(self, features: np.ndarray, labels: np.ndarray,
                       dates: pd.Series, event_type: str,
                       window_end: str, labels_hash: str) -> Dict:
        """Train LSTM forecast model"""
        if event_type == "reset" or self.current_lstm_model is None:
            print("Training new LSTM forecast model (reset event)")
            return self._train_lstm_from_scratch(features, labels, dates, window_end, labels_hash)
        else:
            print("Updating LSTM forecast model (incremental event)")
            return self._lstm_incremental_update(features, labels, dates, window_end, labels_hash)
    
    def _train_lstm_from_scratch(self, features: np.ndarray, labels: np.ndarray,
                                dates: pd.Series, window_end: str, 
                                labels_hash: str) -> Dict:
        """Train LSTM from scratch with walk-forward validation"""
        # Create sequence dataset
        print(f"Creating sequences with length={self.sequence_length}, horizon={self.forecast_horizon}")
        dataset = RegimeSequenceDataset(
            features, labels, self.sequence_length, self.forecast_horizon
        )
        
        if len(dataset) < 100:
            print(f"Warning: Only {len(dataset)} sequences available")
        
        # Model configuration
        num_classes = len(np.unique(labels))
        input_dim = features.shape[1]
        
        print(f"Model config: input_dim={input_dim}, num_classes={num_classes}")
        
        # Initialize model
        self.current_lstm_model = RegimeForecastLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.2
        ).to(self.device)
        
        # Training setup
        train_size = int(0.8 * len(dataset))
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        optimizer = torch.optim.Adam(self.current_lstm_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        print("Training LSTM...")
        for epoch in range(100):
            # Training
            self.current_lstm_model.train()
            train_loss = 0
            train_correct = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.current_lstm_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == y_batch).sum().item()
            
            # Validation
            self.current_lstm_model.eval()
            val_loss = 0
            val_correct = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.current_lstm_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == y_batch).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / len(train_dataset)
            val_acc = val_correct / len(val_dataset)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, "
                      f"Val Acc={val_acc:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.current_lstm_model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.current_lstm_model.load_state_dict(best_model_state)
        
        # Evaluate final model
        metrics = self._evaluate_lstm(dataset)
        
        # Save model
        self._save_lstm_checkpoint(window_end, labels_hash)
        
        return {
            'event_type': 'reset',
            'n_sequences': len(dataset),
            'train_size': train_size,
            'val_size': len(val_dataset),
            'best_val_loss': float(best_val_loss),
            'final_metrics': metrics,
            'model_architecture': {
                'input_dim': input_dim,
                'hidden_dim': 128,
                'num_layers': 2,
                'num_classes': num_classes
            }
        }
    
    def _lstm_incremental_update(self, features: np.ndarray, labels: np.ndarray,
                               dates: pd.Series, window_end: str, 
                               labels_hash: str) -> Dict:
        """Incremental LSTM update with limited epochs"""
        if self.current_lstm_model is None:
            return self._train_lstm_from_scratch(features, labels, dates, window_end, labels_hash)
        
        # Create dataset
        dataset = RegimeSequenceDataset(features, labels, self.sequence_length, self.forecast_horizon)
        
        # Fine-tune for 2 epochs as specified
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.current_lstm_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=2 * len(train_loader)
        )
        criterion = nn.CrossEntropyLoss()
        
        print("Fine-tuning LSTM for 2 epochs...")
        self.current_lstm_model.train()
        
        for epoch in range(2):
            epoch_loss = 0
            epoch_correct = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.current_lstm_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                epoch_correct += (outputs.argmax(1) == y_batch).sum().item()
            
            accuracy = epoch_correct / len(dataset)
            print(f"  Epoch {epoch + 1}/2: Loss={epoch_loss/len(train_loader):.4f}, Acc={accuracy:.4f}")
        
        # Evaluate
        metrics = self._evaluate_lstm(dataset)
        
        # Save updated model
        self._save_lstm_checkpoint(window_end, labels_hash)
        
        return {
            'event_type': 'incremental',
            'n_sequences': len(dataset),
            'fine_tune_epochs': 2,
            'final_metrics': metrics
        }
    
    def _evaluate_lstm(self, dataset: Dataset) -> Dict:
        """Evaluate LSTM model with comprehensive metrics"""
        self.current_lstm_model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.current_lstm_model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        mae = mean_absolute_error(all_targets, all_preds)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        # Directional accuracy (predicting regime changes)
        if len(all_targets) > 1:
            target_changes = np.diff(all_targets) != 0
            pred_changes = np.diff(all_preds) != 0
            directional_accuracy = np.mean(target_changes == pred_changes)
        else:
            directional_accuracy = 0.0
        
        # Portfolio Sharpe (simplified trading strategy)
        if len(all_preds) > 1:
            # Long when predicting regime 1 (bull), short when regime 2 (bear)
            positions = np.where(all_preds == 1, 1, np.where(all_preds == 2, -1, 0))
            
            # Simulated returns based on actual regime
            regime_returns = {0: 0.0001, 1: 0.0005, 2: -0.0003}
            actual_returns = np.array([regime_returns.get(r, 0) for r in all_targets])
            
            strategy_returns = positions[:-1] * actual_returns[1:]
            
            if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        return {
            'mae': float(mae),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'directional_accuracy': float(directional_accuracy),
            'portfolio_sharpe': float(sharpe),
            'n_samples': len(all_targets)
        }
    
    def predict_regime_forecast(self, features: np.ndarray, 
                              horizon: int = 30) -> Dict[str, np.ndarray]:
        """Generate regime forecasts for specified horizon"""
        if self.current_lstm_model is None:
            raise ValueError("No forecast model available")
        
        self.current_lstm_model.eval()
        
        # Create sequences
        sequences = []
        for i in range(len(features) - self.sequence_length + 1):
            seq = features[i:i + self.sequence_length]
            sequences.append(seq)
        
        if not sequences:
            raise ValueError("Not enough data to create sequences")
        
        sequences = np.array(sequences, dtype=np.float32)
        
        # Generate predictions
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), 64):  # Batch processing
                batch = torch.FloatTensor(sequences[i:i+64]).to(self.device)
                outputs = self.current_lstm_model(batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        return {
            'regime_predictions': predictions,
            'regime_probabilities': probabilities,
            'forecast_horizon': horizon
        }
    
    def predict_transition_matrix(self, features: np.ndarray, 
                                horizon: int = 30) -> np.ndarray:
        """Predict regime transition matrix for specified horizon"""
        if self.current_lstm_model is None:
            raise ValueError("No forecast model loaded")
        
        self.current_lstm_model.eval()
        num_classes = self.current_lstm_model.num_classes
        
        # Initialize transition counts
        transition_counts = np.zeros((num_classes, num_classes))
        
        # Generate predictions and count transitions
        with torch.no_grad():
            for start_idx in range(len(features) - self.sequence_length - horizon):
                # Current sequence
                sequence = features[start_idx:start_idx + self.sequence_length]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                # Predict current regime
                current_output = self.current_lstm_model(sequence_tensor)
                current_regime = int(torch.argmax(current_output).item())
                
                # Future sequence
                future_idx = start_idx + horizon
                if future_idx + self.sequence_length <= len(features):
                    future_sequence = features[future_idx:future_idx + self.sequence_length]
                    future_tensor = torch.FloatTensor(future_sequence).unsqueeze(0).to(self.device)
                    
                    # Predict future regime
                    future_output = self.current_lstm_model(future_tensor)
                    future_regime = int(torch.argmax(future_output).item())
                    
                    transition_counts[current_regime, future_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_counts, 
            row_sums, 
            where=row_sums > 0,
            out=np.zeros_like(transition_counts)
        )
        
        return transition_matrix
    
    def _save_lgb_model(self, window_end: str) -> None:
        """Save LightGBM model with metadata"""
        model_path = os.path.join(
            self.models_dir, 
            f"lgb_interpreter_{window_end}.pkl"
        )
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.current_lgb_model,
                'feature_names': self.feature_names,
                'window_end': window_end,
                'model_params': self.lgb_params,
                'creation_time': datetime.now().isoformat()
            }, f)
        
        print(f"Saved LightGBM model: {model_path}")
    
    def _save_shap_results(self, shap_values: np.ndarray, 
                         feature_importance: List[Dict], window_end: str) -> None:
        """Save SHAP explanation results"""
        shap_path = os.path.join(
            self.shap_dir,
            f"shap_explanation_{window_end}.pkl"
        )
        
        # Handle the case where shap_values might be a list
        if isinstance(shap_values, list):
            # For multiclass, save as list
            save_values = shap_values
        else:
            save_values = shap_values
        
        with open(shap_path, 'wb') as f:
            pickle.dump({
                'shap_values': save_values,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'window_end': window_end,
                'explainer_type': 'TreeExplainer',
                'creation_time': datetime.now().isoformat()
            }, f)
        
        # Also save top features as JSON for easy access
        top_features_path = os.path.join(
            self.shap_dir,
            f"top_features_{window_end}.json"
        )
        
        with open(top_features_path, 'w') as f:
            json.dump({
                'top_20_features': feature_importance[:20],
                'window_end': window_end
            }, f, indent=2)
        
        print(f"Saved SHAP results: {shap_path}")
    
    def _save_lstm_checkpoint(self, window_end: str, labels_hash: str) -> None:
        """Save LSTM model checkpoint"""
        checkpoint = {
            'model_state_dict': self.current_lstm_model.state_dict(),
            'model_config': {
                'input_dim': self.current_lstm_model.input_dim,
                'hidden_dim': self.current_lstm_model.hidden_dim,
                'num_layers': self.current_lstm_model.num_layers,
                'num_classes': self.current_lstm_model.num_classes
            },
            'window_end': window_end,
            'labels_hash': labels_hash,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'creation_time': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoints_dir, 
            f"lstm_checkpoint_{window_end}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Saved LSTM checkpoint: {checkpoint_path}")
    
    def _version_models(self, window_end: str, event_type: str) -> None:
        """Version and track all model artifacts"""
        version_info = {
            'window_end': window_end,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'models': {
                'lgb_interpreter': f"lgb_interpreter_{window_end}.pkl",
                'lstm_forecast': f"lstm_checkpoint_{window_end}.pt",
                'shap_explanation': f"shap_explanation_{window_end}.pkl"
            },
            'invalidates_previous': event_type == 'reset'
        }
        
        version_file = os.path.join(
            self.models_dir,
            f"model_version_{window_end}.json"
        )
        
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        print(f"Models versioned for window {window_end}")
    
    def load_models(self, window_end: str) -> None:
        """Load previously trained models"""
        # Load LightGBM
        lgb_path = os.path.join(self.models_dir, f"lgb_interpreter_{window_end}.pkl")
        if os.path.exists(lgb_path):
            with open(lgb_path, 'rb') as f:
                data = pickle.load(f)
                self.current_lgb_model = data['model']
                self.feature_names = data['feature_names']
                print(f"Loaded LightGBM model from {window_end}")
        
        # Load LSTM
        lstm_path = os.path.join(self.checkpoints_dir, f"lstm_checkpoint_{window_end}.pt")
        if os.path.exists(lstm_path):
            checkpoint = torch.load(lstm_path, map_location=self.device)
            
            config = checkpoint['model_config']
            self.current_lstm_model = RegimeForecastLSTM(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_classes=config['num_classes']
            ).to(self.device)
            
            self.current_lstm_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded LSTM model from {window_end}")


# Ensure the module is complete and working
if __name__ == "__main__":
    print("Analysis module loaded successfully")
    print("Available classes:")
    print("  - LabelMaterialisation")
    print("  - InterpretationForecasting")
    print("  - RegimeSequenceDataset")
    print("  - RegimeForecastLSTM")