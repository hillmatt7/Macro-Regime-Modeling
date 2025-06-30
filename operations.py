# operations.py
"""
Operations components: Signal publishing, monitoring, scheduling, and reproducibility
Consolidates Steps 9-12 of the pipeline
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import hashlib
import random
import torch
from dotenv import load_dotenv
import schedule
import requests
import pandas as pd


@dataclass
class RegimeSignal:
    """Compact payload for regime signal"""
    latest_regime_id: int
    posterior_probabilities: List[float]
    transition_matrix: List[List[float]]
    top_10_shap_features: List[Dict]
    model_version_ids: Dict[str, str]
    timestamp: str
    window_end: str


class SignalPublishingMonitoring:
    """Steps 9-10: Signal publication and monitoring"""
    
    def __init__(self, message_bus_connection=None, pagerduty_key: Optional[str] = None):
        # Signal publishing
        self.message_bus = message_bus_connection
        self.topic = "macro_regime_signals"
        self.max_retries = 5
        self.base_backoff = 1.0
        
        # Monitoring
        self.pagerduty_key = pagerduty_key
        self.metrics_dir = "monitoring_metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.ari_threshold = 0.80
        self.f1_threshold = 0.55
        self.sharpe_threshold = 0.0
        self.consecutive_failures = 2
        
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)
    
    def publish_and_monitor(self, 
                          latest_regime: int,
                          posterior_probs: np.ndarray,
                          transition_matrix: np.ndarray,
                          top_features: List[Dict],
                          model_versions: Dict[str, str],
                          window_end: str,
                          metrics: Dict) -> Dict:
        """
        Check health rules and publish signal if allowed
        Returns: Dict with publication status and health check results
        """
        # Step 10: Check health rules
        health_breaches = self._check_health_rules(
            metrics['bootstrap_ari'],
            metrics['interpretation_f1'],
            metrics['forecast_sharpe'],
            window_end
        )
        
        publication_allowed = self._check_publication_allowed()
        publication_success = False
        
        # Step 9: Publish signal if allowed
        if publication_allowed:
            signal = RegimeSignal(
                latest_regime_id=latest_regime,
                posterior_probabilities=posterior_probs.tolist(),
                transition_matrix=transition_matrix.tolist(),
                top_10_shap_features=top_features[:10],
                model_version_ids=model_versions,
                timestamp=datetime.utcnow().isoformat(),
                window_end=window_end
            )
            
            publication_success = self._publish_signal(signal)
        else:
            self.logger.warning("Signal publication blocked due to health rule breach")
        
        return {
            'health_breaches': health_breaches,
            'publication_allowed': publication_allowed,
            'publication_success': publication_success
        }
    
    def _check_health_rules(self, bootstrap_ari: float, interpretation_f1: float,
                          forecast_sharpe: float, window_end: str) -> Dict[str, bool]:
        """Check three health rules"""
        current_metrics = {
            'window_end': window_end,
            'timestamp': datetime.utcnow().isoformat(),
            'bootstrap_ari': bootstrap_ari,
            'interpretation_f1': interpretation_f1,
            'forecast_sharpe': forecast_sharpe
        }
        
        self.metrics_history.append(current_metrics)
        self._save_metrics(current_metrics, window_end)
        
        breaches = {}
        
        # Rule (i): Bootstrap ARI
        if bootstrap_ari < self.ari_threshold:
            breaches['bootstrap_ari_low'] = True
            self._create_incident(
                f"Bootstrap ARI dropped to {bootstrap_ari:.3f} < {self.ari_threshold}",
                severity='high'
            )
        else:
            breaches['bootstrap_ari_low'] = False
        
        # Rule (ii): Interpretation F1
        if interpretation_f1 < self.f1_threshold:
            breaches['interpretation_f1_low'] = True
            self._create_incident(
                f"Interpretation F1 dropped to {interpretation_f1:.3f} < {self.f1_threshold}",
                severity='high'
            )
        else:
            breaches['interpretation_f1_low'] = False
        
        # Rule (iii): Forecast Sharpe (consecutive check)
        sharpe_breach = self._check_consecutive_sharpe_breach(forecast_sharpe)
        breaches['forecast_sharpe_negative'] = sharpe_breach
        
        if sharpe_breach:
            self._create_incident(
                f"Forecast Sharpe negative for {self.consecutive_failures} consecutive windows",
                severity='critical'
            )
        
        # Any breach triggers publication freeze
        if any(breaches.values()):
            self._freeze_publication(breaches, window_end)
        
        return breaches
    
    def _publish_signal(self, signal: RegimeSignal) -> bool:
        """Publish signal with retry logic"""
        payload = json.dumps(asdict(signal))
        
        for attempt in range(self.max_retries):
            try:
                message_id = f"regime_signal_{signal.window_end}_{datetime.utcnow().strftime('%Y%m%d')}"
                
                if self.message_bus:
                    result = self.message_bus.publish(
                        topic=self.topic,
                        message=payload,
                        message_id=message_id,
                        deduplication=True
                    )
                    
                    if self.message_bus.confirm_commit(result):
                        self.logger.info(f"Successfully published regime signal for {signal.window_end}")
                        return True
                else:
                    self.logger.info(f"No message bus configured, signal prepared: {message_id}")
                    return True
                    
            except Exception as e:
                backoff_time = self.base_backoff * (2 ** attempt)
                self.logger.warning(
                    f"Publication attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {backoff_time}s..."
                )
                
                if attempt >= 2:
                    self._alert_sre(f"Regime signal publication failing repeatedly: {e}")
                
                time.sleep(backoff_time)
        
        self._alert_sre(
            f"CRITICAL: Failed to publish regime signal after {self.max_retries} attempts"
        )
        return False
    
    def _check_consecutive_sharpe_breach(self, current_sharpe: float) -> bool:
        """Check if Sharpe has been negative for consecutive windows"""
        if current_sharpe >= self.sharpe_threshold:
            return False
        
        consecutive_count = 1
        
        for i in range(len(self.metrics_history) - 2, -1, -1):
            past_metrics = self.metrics_history[i]
            if past_metrics['forecast_sharpe'] < self.sharpe_threshold:
                consecutive_count += 1
            else:
                break
        
        return consecutive_count >= self.consecutive_failures
    
    def _create_incident(self, description: str, severity: str = 'high') -> None:
        """Create PagerDuty incident"""
        if not self.pagerduty_key:
            self.logger.error(f"No PagerDuty key configured. Incident: {description}")
            return
        
        try:
            incident_data = {
                "routing_key": self.pagerduty_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"Macro Regime Model Alert: {description}",
                    "source": "macro_regime_monitoring",
                    "severity": severity,
                    "timestamp": datetime.utcnow().isoformat(),
                    "custom_details": {
                        "description": description,
                        "component": "regime_modelling_pipeline",
                        "metrics_history": self.metrics_history[-5:]
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=incident_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 202:
                self.logger.info(f"PagerDuty incident created: {description}")
            else:
                self.logger.error(f"Failed to create PagerDuty incident: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error creating PagerDuty incident: {e}")
    
    def _freeze_publication(self, breaches: Dict[str, bool], window_end: str) -> None:
        """Freeze publication until human re-approval"""
        freeze_file = os.path.join(self.metrics_dir, "PUBLICATION_FROZEN.json")
        
        freeze_data = {
            "frozen": True,
            "window_end": window_end,
            "timestamp": datetime.utcnow().isoformat(),
            "breaches": breaches,
            "message": "Publication frozen due to health rule breach. Human approval required."
        }
        
        with open(freeze_file, 'w') as f:
            json.dump(freeze_data, f, indent=2)
        
        self.logger.critical("PUBLICATION FROZEN - Human intervention required")
    
    def _check_publication_allowed(self) -> bool:
        """Check if publication is currently allowed"""
        freeze_file = os.path.join(self.metrics_dir, "PUBLICATION_FROZEN.json")
        
        if os.path.exists(freeze_file):
            with open(freeze_file, 'r') as f:
                freeze_data = json.load(f)
            
            if freeze_data.get('frozen', False):
                self.logger.warning("Publication blocked - system is frozen")
                return False
        
        return True
    
    def _alert_sre(self, message: str) -> None:
        """Alert SRE team via webhook"""
        webhook_url = os.getenv('SRE_WEBHOOK_URL')
        
        if not webhook_url:
            self.logger.error(f"No webhook URL configured. Alert: {message}")
            return
        
        try:
            alert_payload = {
                "service": "macro_regime_modelling",
                "severity": "high",
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "component": "signal_publisher"
            }
            
            response = requests.post(webhook_url, json=alert_payload, timeout=5)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to send SRE alert: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending SRE alert: {e}")
    
    def _save_metrics(self, metrics: Dict, window_end: str) -> None:
        """Save metrics for historical tracking"""
        filename = f"metrics_{window_end}.json"
        filepath = os.path.join(self.metrics_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def approve_publication(self, approver: str, reason: str) -> None:
        """Human approval to unfreeze publication"""
        freeze_file = os.path.join(self.metrics_dir, "PUBLICATION_FROZEN.json")
        
        if os.path.exists(freeze_file):
            with open(freeze_file, 'r') as f:
                freeze_data = json.load(f)
            
            freeze_data['frozen'] = False
            freeze_data['approved_by'] = approver
            freeze_data['approval_reason'] = reason
            freeze_data['approval_timestamp'] = datetime.utcnow().isoformat()
            
            with open(freeze_file, 'w') as f:
                json.dump(freeze_data, f, indent=2)
            
            self.logger.info(f"Publication approved by {approver}: {reason}")


class SchedulingReproducibility:
    """Steps 11-12: Scheduling and reproducibility guardrails"""
    
    def __init__(self, pipeline_executor=None):
        # Scheduling
        self.pipeline_executor = pipeline_executor
        self.window_length_years = 10
        self.advance_months = 6
        self.schedule_months = [1, 7]  # January and July
        self.state_file = "scheduler_state.json"
        
        # Reproducibility
        load_dotenv()
        self.numpy_seed = int(os.getenv('NUMPY_SEED', '42'))
        self.random_seed = int(os.getenv('RANDOM_SEED', '42'))
        self.torch_seed = int(os.getenv('TORCH_SEED', '42'))
        
        self.manifest_dir = "execution_manifests"
        os.makedirs(self.manifest_dir, exist_ok=True)
        
        self._set_global_seeds()
        self.logger = logging.getLogger(__name__)
    
    def _set_global_seeds(self) -> None:
        """Set all random seeds for reproducibility"""
        np.random.seed(self.numpy_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.torch_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.torch_seed)
            torch.cuda.manual_seed_all(self.torch_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def calculate_window_dates(self, reference_date: Optional[datetime] = None) -> Tuple[str, str]:
        """Calculate window start and end dates based on schedule"""
        if reference_date is None:
            reference_date = datetime.now()
        
        # Load state
        state = self._load_state()
        
        if state is None:
            # First run
            window_end = reference_date.date()
            window_start = window_end - relativedelta(years=self.window_length_years)
        else:
            # Advance windows
            prev_start = datetime.fromisoformat(state['window_start']).date()
            prev_end = datetime.fromisoformat(state['window_end']).date()
            
            window_start = prev_start + relativedelta(months=self.advance_months)
            window_end = prev_end + relativedelta(months=self.advance_months)
        
        return window_start.isoformat(), window_end.isoformat()
    
    def emit_manifest(self, step_name: str, inputs: Dict[str, Any],
                     outputs: Dict[str, Any], window_end: str) -> str:
        """Emit execution manifest for reproducibility"""
        input_hash = self._calculate_hash(inputs)
        output_hash = self._calculate_hash(outputs)
        
        manifest = {
            'step_name': step_name,
            'window_end': window_end,
            'execution_timestamp': datetime.utcnow().isoformat(),
            'inputs': {
                'hash': input_hash,
                'details': self._serialize_for_manifest(inputs)
            },
            'outputs': {
                'hash': output_hash,
                'details': self._serialize_for_manifest(outputs)
            },
            'environment': {
                'numpy_seed': self.numpy_seed,
                'random_seed': self.random_seed,
                'torch_seed': self.torch_seed,
                'numpy_version': np.__version__,
                'torch_version': torch.__version__
            }
        }
        
        filename = f"manifest_{step_name}_{window_end}_{output_hash[:8]}.json"
        filepath = os.path.join(self.manifest_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return filepath
    
    def save_execution_state(self, window_start: str, window_end: str, result: Dict) -> None:
        """Save scheduler state after execution"""
        state = {
            'window_start': window_start,
            'window_end': window_end,
            'last_execution': datetime.now().isoformat(),
            'execution_result': result,
            'window_length_years': self.window_length_years,
            'advance_months': self.advance_months
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def setup_schedule(self) -> None:
        """Setup twice-yearly schedule"""
        next_runs = self._calculate_next_runs()
        
        for run_date in next_runs:
            schedule_time = f"{run_date.hour:02d}:{run_date.minute:02d}"
            
            if run_date.month == 1:
                schedule.every().january.at(schedule_time).do(self._execute_scheduled)
            else:
                schedule.every().july.at(schedule_time).do(self._execute_scheduled)
        
        self.logger.info(f"Scheduled pipeline runs for: {[str(d) for d in next_runs]}")
    
    def run_scheduler(self) -> None:
        """Run the scheduler continuously"""
        self.setup_schedule()
        self.logger.info("Scheduler started. Waiting for next execution window...")
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    
    def _execute_scheduled(self) -> None:
        """Execute pipeline on schedule"""
        window_start, window_end = self.calculate_window_dates()
        
        self.logger.info(f"Executing scheduled pipeline for window: {window_start} to {window_end}")
        
        if self.pipeline_executor:
            result = self.pipeline_executor(
                window_start=window_start,
                window_end=window_end
            )
            self.save_execution_state(window_start, window_end, result)
    
    def _calculate_next_runs(self) -> List[datetime]:
        """Calculate next scheduled run dates"""
        now = datetime.now()
        next_runs = []
        
        for month in self.schedule_months:
            year = now.year
            if month < now.month or (month == now.month and self._get_second_weekend(year, month) < now):
                year += 1
            
            run_date = self._get_second_weekend(year, month)
            next_runs.append(run_date)
        
        return sorted(next_runs)
    
    def _get_second_weekend(self, year: int, month: int) -> datetime:
        """Get the second Saturday of a given month"""
        first_day = datetime(year, month, 1)
        days_until_saturday = (5 - first_day.weekday()) % 7
        first_saturday = first_day + timedelta(days=days_until_saturday)
        second_saturday = first_saturday + timedelta(days=7)
        
        return second_saturday.replace(hour=2, minute=0, second=0)
    
    def _load_state(self) -> Optional[Dict]:
        """Load scheduler state from file"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA256 hash of data structure"""
        if isinstance(data, dict):
            stable_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            stable_str = json.dumps(list(data))
        elif isinstance(data, np.ndarray):
            stable_str = np.array2string(data, separator=',')
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            stable_str = data.to_csv(index=False)
        else:
            stable_str = str(data)
        
        return hashlib.sha256(stable_str.encode()).hexdigest()
    
    def _serialize_for_manifest(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, np.ndarray):
            return {
                'type': 'numpy_array',
                'shape': data.shape,
                'dtype': str(data.dtype),
                'sample': data.flat[:5].tolist() if data.size > 0 else []
            }
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return {
                'type': type(data).__name__,
                'shape': data.shape,
                'columns': list(data.columns) if hasattr(data, 'columns') else None
            }
        elif isinstance(data, dict):
            return {k: self._serialize_for_manifest(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_for_manifest(item) for item in data[:5]]
        else:
            return str(data)
    
    def create_synthetic_test_data(self, n_samples: int = 1000,
                                 n_features: int = 10,
                                 n_regimes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic Gaussian data for testing"""
        rng = np.random.RandomState(self.numpy_seed)
        
        regime_centers = rng.randn(n_regimes, n_features) * 3
        
        features = []
        labels = []
        
        samples_per_regime = n_samples // n_regimes
        
        for regime in range(n_regimes):
            regime_data = rng.randn(samples_per_regime, n_features)
            regime_data += regime_centers[regime]
            
            features.append(regime_data)
            labels.extend([regime] * samples_per_regime)
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        shuffle_idx = rng.permutation(len(labels))
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]
        
        return features, labels