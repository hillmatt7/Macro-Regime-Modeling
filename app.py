# app.py
"""
Streamlit app for Macro Regime Modelling Pipeline
Complete implementation with proper data handling and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, date
import json
import time
from typing import Dict, Tuple, Optional, List
import logging
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from core import DataIngestionPreprocessing, GMMFittingSelection
from analysis import LabelMaterialisation, InterpretationForecasting
from operations import SignalPublishingMonitoring, SchedulingReproducibility

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page config
st.set_page_config(
    page_title="Macro Regime Modelling",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


class MacroRegimePipelineApp:
    """Main application class with complete implementation"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_sidebar()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'pipeline_results' not in st.session_state:
            st.session_state.pipeline_results = None
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'processed_tensor' not in st.session_state:
            st.session_state.processed_tensor = None
        if 'regime_labels' not in st.session_state:
            st.session_state.regime_labels = None
        if 'posterior_probs' not in st.session_state:
            st.session_state.posterior_probs = None
        if 'shap_values' not in st.session_state:
            st.session_state.shap_values = None
        if 'feature_names' not in st.session_state:
            st.session_state.feature_names = None
    
    def setup_sidebar(self):
        """Setup sidebar with configuration options"""
        st.sidebar.title("⚙️ Pipeline Configuration")
        
        # Data source selection
        st.sidebar.header("📊 Data Source")
        
        # Primary data source (always FRED + market data)
        st.sidebar.info("The pipeline always fetches:\n• FRED macro data\n• Market OHLCV (SPY, ES, NQ, etc.)")
        
        # Optional user CSV
        self.use_user_csv = st.sidebar.checkbox("Add custom CSV data")
        self.csv_file = None
        if self.use_user_csv:
            self.csv_file = st.sidebar.file_uploader(
                "Upload additional CSV",
                type=['csv'],
                help="CSV will be merged with FRED and market data"
            )
        
        # Demo mode
        self.use_demo_data = st.sidebar.checkbox(
            "Use demo data",
            help="Use simulated data if FRED/market data unavailable"
        )
        
        # Date range selection
        st.sidebar.header("📅 Date Range")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            self.window_start = st.date_input(
                "Start date",
                value=date(2014, 1, 1),
                min_value=date(2000, 1, 1),
                max_value=date.today()
            )
        
        with col2:
            self.window_end = st.date_input(
                "End date",
                value=date(2024, 1, 1),
                min_value=self.window_start,
                max_value=date.today()
            )
        
        # K-selection parameters
        st.sidebar.header("🎯 Regime Detection")
        self.k_min = st.sidebar.slider("Minimum k", 1, 10, 1)
        self.k_max = st.sidebar.slider("Maximum k", self.k_min, 20, 10)
        
        self.force_k = st.sidebar.checkbox("Force specific k")
        if self.force_k:
            self.forced_k_value = st.sidebar.slider(
                "Select k value",
                self.k_min,
                self.k_max,
                (self.k_min + self.k_max) // 2
            )
        else:
            self.forced_k_value = None
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            self.skip_monitoring = st.checkbox("Skip health monitoring")
            self.dry_run = st.checkbox("Dry run (no signal publication)")
            self.random_seed = st.number_input("Random seed", value=42)
            self.bootstrap_samples = st.number_input("Bootstrap samples", value=500, min_value=100)
            self.ari_threshold = st.number_input("ARI threshold", value=0.85, min_value=0.5, max_value=1.0)
        
        # FRED API key
        with st.sidebar.expander("🔑 API Configuration"):
            fred_key = st.text_input("FRED API Key (optional)", type="password")
            if fred_key:
                os.environ['FRED_API_KEY'] = fred_key
        
        # Run button
        st.sidebar.divider()
        self.run_pipeline = st.sidebar.button(
            "🚀 Run Pipeline",
            type="primary",
            use_container_width=True
        )
    
    def run_pipeline_execution(self):
        """Execute the full pipeline with complete implementation"""
        progress_bar = st.progress(0)
        status_container = st.container()
        
        try:
            # Initialize components
            with status_container:
                st.info("🔄 Initializing pipeline components...")
            
            # Step 1-2: Data ingestion and preprocessing
            data_processor = DataIngestionPreprocessing()
            
            with status_container:
                st.info("📥 Loading FRED macro data and market OHLCV...")
            
            # Prepare CSV path if uploaded
            csv_path = None
            if self.use_user_csv and self.csv_file:
                csv_path = f"temp_upload_{datetime.now().timestamp()}.csv"
                with open(csv_path, 'wb') as f:
                    f.write(self.csv_file.getbuffer())
            
            # Ingest and preprocess data
            raw_data, processed_tensor = data_processor.ingest_and_preprocess(
                self.window_start.isoformat(),
                self.window_end.isoformat(),
                user_csv_path=csv_path,
                use_demo_data=self.use_demo_data
            )
            
            st.session_state.raw_data = raw_data
            st.session_state.processed_tensor = processed_tensor
            st.session_state.feature_names = data_processor.feature_names
            
            progress_bar.progress(20)
            
            with status_container:
                st.success(f"✓ Loaded {len(raw_data)} days of data with {len(data_processor.feature_names)} features")
            
            # Step 3-4: GMM fitting and selection
            gmm_selector = GMMFittingSelection(
                k_min=self.k_min,
                k_max=self.k_max,
                random_seed=self.random_seed
            )
            gmm_selector.bootstrap_samples = self.bootstrap_samples
            gmm_selector.ari_threshold = self.ari_threshold
            
            with status_container:
                st.info(f"🔍 Fitting GMMs for k={self.k_min} to k={self.k_max}...")
            
            k_star, selection_metadata, candidates, posterior_probs = gmm_selector.fit_and_select(
                processed_tensor,
                self.window_end.isoformat(),
                force_k=self.forced_k_value
            )
            
            st.session_state.posterior_probs = posterior_probs
            
            progress_bar.progress(50)
            
            with status_container:
                st.success(f"✓ Selected k={k_star} with ICL={selection_metadata['icl']:.2f}, "
                          f"Bootstrap ARI={selection_metadata['bootstrap_mean_ari']:.3f}")
            
            # Step 5-6: Label materialisation and change detection
            label_manager = LabelMaterialisation()
            
            with status_container:
                st.info("🏷️ Materialising regime labels...")
            
            selected_labels = candidates[k_star].labels
            dates = raw_data['date'] if 'date' in raw_data.columns else pd.date_range(
                start=self.window_start,
                end=self.window_end,
                periods=len(selected_labels)
            )
            
            labels_hash, event_type, change_metadata = label_manager.materialise_and_detect_change(
                selected_labels,
                posterior_probs,
                dates,
                selection_metadata,
                self.window_end.isoformat(),
                self.window_start.isoformat()
            )
            
            st.session_state.regime_labels = pd.DataFrame({
                'date': dates,
                'regime': selected_labels
            })
            
            progress_bar.progress(60)
            
            with status_container:
                st.info(f"📊 Event type: {event_type} - {change_metadata['reason']}")
            
            # Step 7-8: Interpretation and forecasting
            interpreter = InterpretationForecasting()
            
            with status_container:
                st.info("🧠 Training interpretation (LightGBM + SHAP) and forecast (LSTM) models...")
            
            interpretation_results, forecast_results = interpreter.train_interpretation_and_forecast(
                processed_tensor,
                selected_labels,
                dates,
                data_processor.feature_names,
                event_type,
                self.window_end.isoformat(),
                labels_hash
            )
            
            progress_bar.progress(80)
            
            # Generate forecasts
            with status_container:
                st.info("🔮 Generating regime forecasts...")
            
            regime_forecast = interpreter.predict_regime_forecast(processed_tensor)
            transition_matrix = interpreter.predict_transition_matrix(processed_tensor)
            
            # Step 9-10: Signal publication and monitoring
            publisher = SignalPublishingMonitoring(
                message_bus_connection=None,
                pagerduty_key=None if self.dry_run else os.getenv('PAGERDUTY_KEY')
            )
            
            with status_container:
                st.info("📡 Publishing signals and checking health...")
            
            # Prepare metrics
            metrics = {
                'bootstrap_ari': selection_metadata['bootstrap_mean_ari'],
                'interpretation_f1': interpretation_results.get('mean_f1', 
                                                              interpretation_results.get('validation_f1', 0.7)),
                'forecast_sharpe': forecast_results['final_metrics']['portfolio_sharpe']
            }
            
            # Model versions
            model_versions = {
                'gmm': f"gmm_k{k_star}_{self.window_end}",
                'interpretation': f"lgb_{self.window_end}",
                'forecast': f"lstm_{self.window_end}"
            }
            
            if not self.skip_monitoring:
                pub_results = publisher.publish_and_monitor(
                    int(selected_labels[-1]),
                    posterior_probs[-1],
                    transition_matrix,
                    interpretation_results.get('top_10_features', []),
                    model_versions,
                    self.window_end.isoformat(),
                    metrics
                )
            else:
                pub_results = {
                    'health_breaches': {},
                    'publication_allowed': True,
                    'publication_success': True
                }
            
            progress_bar.progress(100)
            
            # Compile comprehensive results
            results = {
                'execution_time': datetime.now().isoformat(),
                'window': {
                    'start': self.window_start.isoformat(),
                    'end': self.window_end.isoformat(),
                    'n_days': len(raw_data),
                    'n_features': len(data_processor.feature_names)
                },
                'regime_detection': {
                    'k_star': k_star,
                    'k_range': f"{self.k_min}-{self.k_max}",
                    'forced': self.forced_k_value is not None,
                    'event_type': event_type,
                    'change_reason': change_metadata['reason'],
                    'latest_regime': int(selected_labels[-1]),
                    'regime_confidence': float(np.max(posterior_probs[-1])),
                    'selection_metadata': selection_metadata
                },
                'model_performance': metrics,
                'health_status': pub_results,
                'interpretation': interpretation_results,
                'forecast': forecast_results,
                'regime_forecast': regime_forecast,
                'transition_matrix': transition_matrix.tolist(),
                'candidates': {k: {
                    'bic': c.bic,
                    'icl': c.icl,
                    'aic': c.aic,
                    'log_likelihood': c.log_likelihood,
                    'converged': c.converged,
                    'n_iter': c.n_iter
                } for k, c in candidates.items()},
                'feature_list': data_processor.feature_names
            }
            
            st.session_state.pipeline_results = results
            
            with status_container:
                st.success("✅ Pipeline execution completed successfully!")
            
            # Clean up temp files
            if csv_path and os.path.exists(csv_path):
                os.remove(csv_path)
            
            return True
            
        except Exception as e:
            with status_container:
                st.error(f"❌ Pipeline execution failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
            return False
    
    def display_results(self):
        """Display comprehensive pipeline results"""
        if st.session_state.pipeline_results is None:
            st.info("👈 Configure pipeline settings and click 'Run Pipeline' to begin")
            return
        
        results = st.session_state.pipeline_results
        
        # Results header
        st.title("📊 Macro Regime Analysis Results")
        
        # Key metrics summary
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Regimes (k*)",
                results['regime_detection']['k_star'],
                f"ICL: {results['regime_detection']['selection_metadata']['icl']:.1f}"
            )
        
        with col2:
            current_regime = results['regime_detection']['latest_regime']
            confidence = results['regime_detection']['regime_confidence']
            st.metric(
                "Current Regime",
                current_regime,
                f"{confidence:.1%} confidence"
            )
        
        with col3:
            ari = results['model_performance']['bootstrap_ari']
            st.metric(
                "Bootstrap ARI",
                f"{ari:.3f}",
                "✓ Stable" if ari >= 0.85 else "⚠️ Unstable"
            )
        
        with col4:
            f1 = results['model_performance']['interpretation_f1']
            st.metric(
                "Interpretation F1",
                f"{f1:.3f}",
                "✓" if f1 >= 0.55 else "⚠️"
            )
        
        with col5:
            sharpe = results['model_performance']['forecast_sharpe']
            st.metric(
                "Forecast Sharpe",
                f"{sharpe:.2f}",
                "✓" if sharpe > 0 else "⚠️"
            )
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Regime Timeline",
            "🎯 Model Selection",
            "🧠 Feature Importance",
            "🔮 Regime Forecasts",
            "📊 Data Explorer",
            "💾 Export Results"
        ])
        
        with tab1:
            self.display_regime_timeline()
        
        with tab2:
            self.display_model_selection()
        
        with tab3:
            self.display_feature_importance()
        
        with tab4:
            self.display_regime_forecasts()
        
        with tab5:
            self.display_data_explorer()
        
        with tab6:
            self.display_export_options()
    
    def display_regime_timeline(self):
        """Display comprehensive regime timeline"""
        st.header("📈 Regime Timeline Analysis")
        
        if st.session_state.regime_labels is None or st.session_state.raw_data is None:
            st.warning("No regime data available")
            return
        
        df = st.session_state.regime_labels.copy()
        raw_df = st.session_state.raw_data.copy()
        results = st.session_state.pipeline_results
        
        # Merge regime labels with raw data
        df = df.merge(raw_df, on='date', how='left')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Market Features with Regime Overlay",
                "Regime Assignment",
                "Regime Posterior Probabilities"
            )
        )
        
        # Color scheme for regimes
        n_regimes = df['regime'].nunique()
        colors = px.colors.qualitative.Set3[:n_regimes]
        regime_colors = {r: colors[i] for i, r in enumerate(sorted(df['regime'].unique()))}
        regime_names = {0: "Normal", 1: "Bull", 2: "Bear"} if n_regimes == 3 else {i: f"Regime {i}" for i in range(n_regimes)}
        
        # Build feature selection safely
        feature_options = [col for col in df.columns if col not in ["date", "regime"]]
        preferred_defaults = ["spy_close", "vix_close", "fred_unemployment"]
        safe_defaults = [d for d in preferred_defaults if d in feature_options]
        
        if not safe_defaults and feature_options:
            safe_defaults = [feature_options[0]]
    
        feature_selector = st.multiselect(
            "Select features to display",
            feature_options,
            default=safe_defaults,
        )
        
        normalize = st.checkbox("Normalize features", value=True)
        
        # Add regime shading WITHOUT annotations (to avoid the timestamp bug)
        for regime in sorted(df['regime'].unique()):
            regime_mask = df['regime'] == regime
            regime_periods = []
            start_idx = None
            
            for idx, is_regime in enumerate(regime_mask):
                if is_regime and start_idx is None:
                    start_idx = idx
                elif not is_regime and start_idx is not None:
                    regime_periods.append((start_idx, idx-1))
                    start_idx = None
            
            if start_idx is not None:
                regime_periods.append((start_idx, len(df)-1))
            
            for start, end in regime_periods:
                if start >= len(df) or end >= len(df):
                    continue
                    
                x0 = df.iloc[start]["date"]
                x1 = df.iloc[end]["date"]
    
                # Add rectangle WITHOUT annotation to avoid timestamp arithmetic bug
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=regime_colors[regime],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
        
        # Plot selected features
        for feature in feature_selector:
            if feature in df.columns:
                y_values = df[feature].values
                if normalize and np.std(y_values) > 0:
                    y_values = (y_values - np.mean(y_values)) / np.std(y_values)
                
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=y_values,
                        name=feature,
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Regime timeline
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['regime'],
                mode='lines+markers',
                line=dict(color='black', width=2),
                marker=dict(size=4),
                name='Regime',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add regime colors to timeline
        for regime in sorted(df['regime'].unique()):
            regime_df = df[df['regime'] == regime]
            fig.add_trace(
                go.Scatter(
                    x=regime_df['date'],
                    y=regime_df['regime'],
                    mode='markers',
                    marker=dict(
                        color=regime_colors[regime],
                        size=8,
                        symbol='circle'
                    ),
                    name=regime_names.get(regime, f"Regime {regime}"),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # Add regime shading to subplot 2 as well
        for regime in sorted(df['regime'].unique()):
            regime_mask = df['regime'] == regime
            regime_periods = []
            start_idx = None
            
            for idx, is_regime in enumerate(regime_mask):
                if is_regime and start_idx is None:
                    start_idx = idx
                elif not is_regime and start_idx is not None:
                    regime_periods.append((start_idx, idx-1))
                    start_idx = None
            
            if start_idx is not None:
                regime_periods.append((start_idx, len(df)-1))
            
            for start, end in regime_periods:
                if start >= len(df) or end >= len(df):
                    continue
                    
                x0 = df.iloc[start]["date"]
                x1 = df.iloc[end]["date"]
    
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=regime_colors[regime],
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    row=2, col=1
                )
        
        # Plot 3: Posterior probabilities
        if st.session_state.posterior_probs is not None:
            posteriors = st.session_state.posterior_probs
            for regime in range(posteriors.shape[1]):
                fig.add_trace(
                    go.Scatter(
                        x=df['date'][:len(posteriors)],
                        y=posteriors[:, regime],
                        name=f"P({regime_names.get(regime, f'R{regime}')})",
                        mode='lines',
                        stackgroup='one',
                        line=dict(color=regime_colors.get(regime, 'gray'))
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Normalized Value" if normalize else "Value", row=1, col=1)
        fig.update_yaxes(title_text="Regime", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=3, col=1, range=[0, 1])
        
        fig.update_layout(
            height=900,
            hovermode='x unified',
            title={
                'text': f"Regime Analysis: {results['regime_detection']['k_star']} Regimes Detected",
                'font': {'size': 20}
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add regime labels as text annotations manually (safer approach)
        st.subheader("🏷️ Regime Labels")
        regime_info = []
        for regime in sorted(df['regime'].unique()):
            regime_df = df[df['regime'] == regime]
            count = len(regime_df)
            percentage = count / len(df) * 100
            regime_info.append({
                'Regime': regime_names.get(regime, f"Regime {regime}"),
                'Days': count,
                'Percentage': f"{percentage:.1f}%",
                'Color': regime_colors[regime]
            })
        
        # Display regime info as a table
        st.dataframe(pd.DataFrame(regime_info), use_container_width=True)
        
        # Regime statistics
        self.display_regime_statistics(df)
    
    def display_regime_statistics(self, df: pd.DataFrame):
        """Display detailed regime statistics"""
        st.subheader("📊 Regime Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regime distribution
            regime_counts = df['regime'].value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Regime {r}" for r in regime_counts.index],
                    y=regime_counts.values,
                    text=[f"{c}<br>{c/len(df)*100:.1f}%" for c in regime_counts.values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Regime Distribution",
                xaxis_title="Regime",
                yaxis_title="Days",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average duration
            durations = []
            current_regime = None
            current_duration = 0
            
            for regime in df['regime']:
                if regime != current_regime:
                    if current_regime is not None:
                        durations.append({
                            'regime': current_regime,
                            'duration': current_duration
                        })
                    current_regime = regime
                    current_duration = 1
                else:
                    current_duration += 1
            
            if current_regime is not None:
                durations.append({
                    'regime': current_regime,
                    'duration': current_duration
                })
            
            duration_df = pd.DataFrame(durations)
            avg_durations = duration_df.groupby('regime')['duration'].agg(['mean', 'max', 'count'])
            
            st.dataframe(
                avg_durations.round(1).rename(columns={
                    'mean': 'Avg Duration (days)',
                    'max': 'Max Duration (days)',
                    'count': 'Number of Periods'
                }),
                use_container_width=True
            )
    
    def display_model_selection(self):
        """Display comprehensive model selection analysis"""
        st.header("🎯 Model Selection Analysis")
        
        results = st.session_state.pipeline_results
        candidates = results['candidates']
        
        # Create model comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Log-Likelihood", "BIC", "ICL", "AIC")
        )
        
        k_values = sorted(candidates.keys())
        metrics = {
            'log_likelihood': [candidates[k]['log_likelihood'] for k in k_values],
            'bic': [candidates[k]['bic'] for k in k_values],
            'icl': [candidates[k]['icl'] for k in k_values],
            'aic': [candidates[k]['aic'] for k in k_values]
        }
        
        # Plot each metric
        for idx, (metric, values) in enumerate(metrics.items()):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=values,
                    mode='lines+markers',
                    name=metric.upper(),
                    marker=dict(size=10),
                    line=dict(width=2)
                ),
                row=row, col=col
            )
            
            # Highlight selected k
            k_star = results['regime_detection']['k_star']
            k_star_value = values[k_values.index(k_star)]
            
            fig.add_trace(
                go.Scatter(
                    x=[k_star],
                    y=[k_star_value],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=f'Selected k={k_star}',
                    showlegend=idx == 0
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Number of Regimes (k)")
        fig.update_layout(height=700, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Selection details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Selection Method**: {results['regime_detection']['selection_metadata']['selection_criteria']}
            
            **Selected k**: {results['regime_detection']['k_star']}
            
            **Event Type**: {results['regime_detection']['event_type']}
            """)
        
        with col2:
            st.info(f"""
            **Bootstrap Validation**:
            - Mean ARI: {results['regime_detection']['selection_metadata']['bootstrap_mean_ari']:.3f}
            - Std ARI: {results['regime_detection']['selection_metadata']['bootstrap_std_ari']:.3f}
            - Samples: {results['regime_detection']['selection_metadata']['bootstrap_samples']}
            """)
        
        with col3:
            # Convergence info
            convergence_data = []
            for k, cand in candidates.items():
                convergence_data.append({
                    'k': k,
                    'Converged': '✓' if cand['converged'] else '✗',
                    'Iterations': cand['n_iter']
                })
            
            st.dataframe(
                pd.DataFrame(convergence_data),
                use_container_width=True
            )
    
    def display_feature_importance(self):
        """Display SHAP-based feature importance"""
        st.header("🧠 Feature Importance Analysis (SHAP)")
        
        results = st.session_state.pipeline_results
        
        if 'top_10_features' not in results['interpretation']:
            st.warning("No feature importance data available")
            return
        
        top_features = results['interpretation']['top_10_features']
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        features = [f['feature'] for f in top_features]
        importances = [f['mean_abs_shap'] for f in top_features]
        
        fig.add_trace(
            go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color='lightblue',
                text=[f"{imp:.3f}" for imp in importances],
                textposition='outside'
            )
        )
        
        fig.update_layout(
            title={
                'text': "Top 10 Features by Mean Absolute SHAP Value",
                'font': {'size': 20}
            },
            xaxis_title="Mean |SHAP|",
            yaxis_title="Feature",
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'mean_f1' in results['interpretation']:
                st.metric(
                    "Cross-Validation F1",
                    f"{results['interpretation']['mean_f1']:.3f}",
                    f"± {results['interpretation']['std_f1']:.3f}"
                )
            else:
                st.metric(
                    "Validation F1",
                    f"{results['interpretation']['validation_f1']:.3f}",
                    "Incremental update"
                )
        
        with col2:
            st.metric(
                "LightGBM Trees",
                results['interpretation']['n_trees'],
                "Boosted trees"
            )
        
        with col3:
            st.metric(
                "Training Type",
                results['interpretation']['event_type'].title(),
                "Full retrain" if results['interpretation']['event_type'] == 'reset' else "Incremental"
            )
        
        # Feature categories breakdown
        st.subheader("📊 Feature Categories")
        
        feature_categories = {
            'Market': [],
            'FRED Macro': [],
            'Technical': [],
            'User Custom': [],
            'Other': []
        }
        
        for feat in st.session_state.feature_names:
            if any(ticker in feat for ticker in ['spy', 'es', 'nq', 'vix', 'tlt', 'gld']):
                feature_categories['Market'].append(feat)
            elif 'fred_' in feat:
                feature_categories['FRED Macro'].append(feat)
            elif any(ind in feat for ind in ['rsi', 'ma', 'bb', 'volatility']):
                feature_categories['Technical'].append(feat)
            elif 'user_' in feat:
                feature_categories['User Custom'].append(feat)
            else:
                feature_categories['Other'].append(feat)
        
        category_counts = {cat: len(feats) for cat, feats in feature_categories.items() if feats}
        
        if category_counts:
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=list(category_counts.keys()),
                    values=list(category_counts.values()),
                    hole=0.3
                )
            ])
            
            fig_pie.update_layout(
                title="Feature Distribution by Category",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def display_regime_forecasts(self):
        """Display regime forecasting results"""
        st.header("🔮 Regime Forecasting Analysis")
        
        results = st.session_state.pipeline_results
        
        # Forecast metrics
        col1, col2, col3, col4 = st.columns(4)
        
        forecast_metrics = results['forecast']['final_metrics']
        
        with col1:
            st.metric(
                "Forecast Accuracy",
                f"{forecast_metrics['accuracy']:.1%}",
                "Next regime prediction"
            )
        
        with col2:
            st.metric(
                "Directional Accuracy",
                f"{forecast_metrics['directional_accuracy']:.1%}",
                "Regime change detection"
            )
        
        with col3:
            st.metric(
                "F1 Score",
                f"{forecast_metrics['f1_score']:.3f}",
                "Multi-class F1"
            )
        
        with col4:
            st.metric(
                "Portfolio Sharpe",
                f"{forecast_metrics['portfolio_sharpe']:.2f}",
                "Trading strategy"
            )
        
        # Transition matrix heatmap
        st.subheader("📊 30-Day Regime Transition Matrix")
        
        transition_matrix = np.array(results['transition_matrix'])
        n_regimes = len(transition_matrix)
        
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=[f"To Regime {i}" for i in range(n_regimes)],
            y=[f"From Regime {i}" for i in range(n_regimes)],
            text=[[f"{val:.1%}" for val in row] for row in transition_matrix],
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Probability of Transitioning Between Regimes (30-day horizon)",
            xaxis_title="Destination Regime",
            yaxis_title="Current Regime",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime persistence analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Regime Persistence")
            
            persistence_data = []
            for i in range(n_regimes):
                self_transition = transition_matrix[i, i]
                expected_duration = 1 / (1 - self_transition) if self_transition < 1 else float('inf')
                
                persistence_data.append({
                    'Regime': i,
                    'Self-Transition': f"{self_transition:.1%}",
                    'Expected Duration': f"{expected_duration:.0f} days" if expected_duration != float('inf') else "∞"
                })
            
            st.dataframe(pd.DataFrame(persistence_data), use_container_width=True)
        
        with col2:
            st.subheader("🔄 Most Likely Transitions")
            
            transitions = []
            for i in range(n_regimes):
                for j in range(n_regimes):
                    if i != j:
                        transitions.append({
                            'from': i,
                            'to': j,
                            'prob': transition_matrix[i, j]
                        })
            
            top_transitions = sorted(transitions, key=lambda x: x['prob'], reverse=True)[:5]
            
            transition_df = pd.DataFrame([
                {
                    'Transition': f"Regime {t['from']} → {t['to']}",
                    'Probability': f"{t['prob']:.1%}"
                }
                for t in top_transitions
            ])
            
            st.dataframe(transition_df, use_container_width=True)
        
        # Forecast visualization
        if 'regime_predictions' in results['regime_forecast']:
            st.subheader("📊 Regime Forecast Visualization")
            
            predictions = results['regime_forecast']['regime_predictions']
            
            # Create forecast plot
            fig = go.Figure()
            
            # Historical regimes
            regime_df = st.session_state.regime_labels
            fig.add_trace(
                go.Scatter(
                    x=regime_df['date'],
                    y=regime_df['regime'],
                    mode='lines',
                    name='Historical Regimes',
                    line=dict(color='black', width=2)
                )
            )
            
            # Predicted regimes (last N points)
            n_predictions = min(len(predictions), 100)
            if n_predictions > 0:
                last_date = regime_df['date'].iloc[-1]
                forecast_dates = pd.date_range(
                    start=last_date,
                    periods=n_predictions + 1,
                    freq='D'
                )[1:]
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=predictions[-n_predictions:],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    )
                )
            
            fig.update_layout(
                title="Regime Sequence with Forecast",
                xaxis_title="Date",
                yaxis_title="Regime",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_data_explorer(self):
        """Interactive data exploration"""
        st.header("📊 Data Explorer")
        
        if st.session_state.raw_data is None:
            st.warning("No data available")
            return
        
        df = st.session_state.raw_data.copy()
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Observations", len(df))
        
        with col2:
            st.metric("Features", len(df.select_dtypes(include=[np.number]).columns))
        
        with col3:
            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            st.metric("Date Range", date_range)
        
        # Feature correlation matrix
        st.subheader("🔗 Feature Correlations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Select subset of features for correlation
            selected_features = st.multiselect(
                "Select features for correlation analysis",
                numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )
            
            if selected_features:
                corr_matrix = df[selected_features].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Feature Correlation Matrix",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series visualization
        st.subheader("📈 Time Series Visualization")
        
        selected_feature = st.selectbox(
            "Select feature to visualize",
            numeric_cols
        )
        
        if selected_feature:
            # Add regime overlay if available
            fig = go.Figure()
            
            # Feature line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[selected_feature],
                    mode='lines',
                    name=selected_feature,
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add regime shading if available (WITHOUT annotations to avoid timestamp bugs)
            if st.session_state.regime_labels is not None:
                regime_df = st.session_state.regime_labels
                n_regimes = regime_df['regime'].nunique()
                colors = px.colors.qualitative.Set3[:n_regimes]
                
                for regime in sorted(regime_df['regime'].unique()):
                    regime_mask = regime_df['regime'] == regime
                    regime_periods = []
                    start_idx = None
                    
                    for idx, is_regime in enumerate(regime_mask):
                        if is_regime and start_idx is None:
                            start_idx = idx
                        elif not is_regime and start_idx is not None:
                            regime_periods.append((start_idx, idx-1))
                            start_idx = None
                    
                    if start_idx is not None:
                        regime_periods.append((start_idx, len(regime_df)-1))
                    
                    for start, end in regime_periods:
                        # Sanity-check indices
                        if start >= len(regime_df) or end >= len(regime_df):
                            continue
                        
                        x0 = regime_df.iloc[start]["date"]
                        x1 = regime_df.iloc[end]["date"]
                        
                        # Add rectangle WITHOUT annotation to avoid timestamp arithmetic bug
                        fig.add_vrect(
                            x0=x0,
                            x1=x1,
                            fillcolor=colors[regime],
                            opacity=0.1,
                            layer="below",
                            line_width=0
                        )
            
            fig.update_layout(
                title=f"{selected_feature} Over Time",
                xaxis_title="Date",
                yaxis_title=selected_feature,
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add regime legend if regimes are available
            if st.session_state.regime_labels is not None:
                st.subheader("🏷️ Regime Legend")
                regime_df = st.session_state.regime_labels
                regime_info = []
                
                for regime in sorted(regime_df['regime'].unique()):
                    regime_mask = regime_df['regime'] == regime
                    count = regime_mask.sum()
                    percentage = count / len(regime_df) * 100
                    color = colors[regime] if regime < len(colors) else 'gray'
                    
                    regime_info.append({
                        'Regime': f"Regime {regime}",
                        'Days': count,
                        'Percentage': f"{percentage:.1f}%",
                        'Color': color
                    })
                
                # Display as colored table
                legend_df = pd.DataFrame(regime_info)
                st.dataframe(legend_df, use_container_width=True)
        
        # Data sample
        st.subheader("📋 Data Sample")
        
        n_rows = st.slider("Number of rows to display", 10, 100, 25)
        
        # Show data with regime labels if available
        display_df = df.head(n_rows)
        if st.session_state.regime_labels is not None:
            regime_df = st.session_state.regime_labels
            display_df = display_df.merge(
                regime_df[['date', 'regime']], 
                on='date', 
                how='left'
            )
        
        st.dataframe(display_df, use_container_width=True)
    
    def display_export_options(self):
        """Export results and data"""
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Download Data")
            
            # Export regime labels with features
            if st.session_state.raw_data is not None and st.session_state.regime_labels is not None:
                export_df = st.session_state.raw_data.merge(
                    st.session_state.regime_labels,
                    on='date',
                    how='left'
                )
                
                # Add posterior probabilities
                if st.session_state.posterior_probs is not None:
                    for i in range(st.session_state.posterior_probs.shape[1]):
                        export_df[f'regime_{i}_prob'] = st.session_state.posterior_probs[:, i]
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Dataset (CSV)",
                    data=csv,
                    file_name=f"regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Export pipeline results
            if st.session_state.pipeline_results:
                results_json = json.dumps(st.session_state.pipeline_results, indent=2)
                st.download_button(
                    label="📥 Download Pipeline Results (JSON)",
                    data=results_json,
                    file_name=f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.subheader("📊 Export Visualizations")
            
            st.info("""
            **To save any chart:**
            1. Hover over the chart
            2. Click the camera icon in the toolbar
            3. Download as PNG
            
            **To get interactive HTML:**
            1. Click the three dots menu
            2. Select "Export to HTML"
            """)
            
            # Model summary report
            if st.button("Generate Summary Report"):
                report = self._generate_summary_report()
                st.download_button(
                    label="📥 Download Summary Report (TXT)",
                    data=report,
                    file_name=f"regime_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    def _generate_summary_report(self) -> str:
        """Generate text summary report"""
        results = st.session_state.pipeline_results
        
        report = f"""
MACRO REGIME MODELLING PIPELINE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
EXECUTIVE SUMMARY
================================================================================
Window Period: {results['window']['start']} to {results['window']['end']}
Data Points: {results['window']['n_days']} days
Features: {results['window']['n_features']} variables

Optimal Regimes: {results['regime_detection']['k_star']}
Current Regime: {results['regime_detection']['latest_regime']} (Confidence: {results['regime_detection']['regime_confidence']:.1%})
Event Type: {results['regime_detection']['event_type']} - {results['regime_detection']['change_reason']}

================================================================================
MODEL PERFORMANCE
================================================================================
Bootstrap Stability (ARI): {results['model_performance']['bootstrap_ari']:.3f}
Interpretation F1 Score: {results['model_performance']['interpretation_f1']:.3f}
Forecast Sharpe Ratio: {results['model_performance']['forecast_sharpe']:.2f}

================================================================================
TOP FEATURES (SHAP)
================================================================================
"""
        if 'top_10_features' in results['interpretation']:
            for i, feat in enumerate(results['interpretation']['top_10_features'][:5]):
                report += f"{i+1}. {feat['feature']}: {feat['mean_abs_shap']:.4f}\n"
        
        report += f"""
================================================================================
REGIME TRANSITION MATRIX (30-day)
================================================================================
"""
        transition_matrix = np.array(results['transition_matrix'])
        for i in range(len(transition_matrix)):
            report += f"From Regime {i}: "
            for j in range(len(transition_matrix)):
                report += f"R{j}={transition_matrix[i,j]:.1%} "
            report += "\n"
        
        report += f"""
================================================================================
FORECAST PERFORMANCE
================================================================================
Accuracy: {results['forecast']['final_metrics']['accuracy']:.1%}
Directional Accuracy: {results['forecast']['final_metrics']['directional_accuracy']:.1%}
F1 Score: {results['forecast']['final_metrics']['f1_score']:.3f}
Portfolio Sharpe: {results['forecast']['final_metrics']['portfolio_sharpe']:.2f}

================================================================================
"""
        
        return report
    
    def run(self):
        """Main app execution"""
        st.title("🌐 Macro Regime Modelling Pipeline")
        st.markdown("### Complete implementation with GMM clustering, SHAP interpretation, and LSTM forecasting")
        
        # Check if pipeline should run
        if self.run_pipeline:
            self.run_pipeline_execution()
        
        # Display results
        self.display_results()


def main():
    """Main entry point"""
    app = MacroRegimePipelineApp()
    app.run()


if __name__ == "__main__":
    main()