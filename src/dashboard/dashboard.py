from __future__ import annotations
from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd
import sqlite3
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analytics.tracking import UsageTracker
from src.analytics.analysis import PipelineAnalyzer


class PipelineDashboard:
    """Real-time dashboard for monitoring pipeline runs"""
    
    def __init__(self, db_path: str = "usage_tracking.db"):
        self.db_path = db_path
        self.tracker = UsageTracker(db_path=db_path)
        self.analyzer = PipelineAnalyzer(db_path=db_path)
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Pipeline Monitoring Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 Pipeline Monitoring Dashboard")
        st.markdown("Real-time monitoring of token usage, costs, and pipeline performance")
        
        # Sidebar for controls
        self._render_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_main_metrics()
            self._render_usage_charts()
        
        with col2:
            self._render_quick_stats()
            self._render_alerts()
        
        # Bottom section
        self._render_detailed_analysis()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("📋 Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "All Time"],
            index=1
        )
        
        # Run selector
        runs = self._get_available_runs()
        selected_run = st.sidebar.selectbox(
            "Pipeline Run",
            ["All Runs"] + runs,
            index=0
        )
        
        # Stage filter
        stages = self._get_available_stages()
        selected_stages = st.sidebar.multiselect(
            "Stages",
            stages,
            default=stages
        )
        
        # Budget settings
        st.sidebar.header("💰 Budget Settings")
        max_budget = st.sidebar.number_input(
            "Max Budget (USD)",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Store selections in session state
        st.session_state.time_range = time_range
        st.session_state.selected_run = selected_run
        st.session_state.selected_stages = selected_stages
        st.session_state.max_budget = max_budget
    
    def _render_main_metrics(self):
        """Render main metric cards"""
        st.header("📈 Key Metrics")
        
        # Get current data
        data = self._get_current_data()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Cost",
                value=f"${data['total_cost']:.4f}",
                delta=f"${data['cost_delta']:.4f}" if data['cost_delta'] != 0 else None
            )
        
        with col2:
            st.metric(
                label="Total Tokens",
                value=f"{data['total_tokens']:,}",
                delta=f"{data['tokens_delta']:,}" if data['tokens_delta'] != 0 else None
            )
        
        with col3:
            st.metric(
                label="Examples Processed",
                value=data['total_examples'],
                delta=data['examples_delta'] if data['examples_delta'] != 0 else None
            )
        
        with col4:
            st.metric(
                label="Avg Confidence",
                value=f"{data['avg_confidence']:.3f}",
                delta=f"{data['confidence_delta']:.3f}" if data['confidence_delta'] != 0 else None
            )
    
    def _render_usage_charts(self):
        """Render usage charts"""
        st.header("📊 Usage Analysis")
        
        # Get usage data
        usage_data = self._get_usage_data()
        
        if not usage_data.empty:
            # Cost over time
            fig_cost = px.line(
                usage_data, 
                x='timestamp', 
                y='cumulative_cost',
                title="Cumulative Cost Over Time",
                labels={'cumulative_cost': 'Cost (USD)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            
            # Token usage by stage
            stage_data = self._get_stage_usage_data()
            if not stage_data.empty:
                fig_stages = px.bar(
                    stage_data,
                    x='stage_id',
                    y=['prompt_tokens', 'completion_tokens'],
                    title="Token Usage by Stage",
                    barmode='stack'
                )
                st.plotly_chart(fig_stages, use_container_width=True)
        else:
            st.info("No usage data available yet. Start a pipeline run to see metrics.")
    
    def _render_quick_stats(self):
        """Render quick statistics"""
        st.header("⚡ Quick Stats")
        
        data = self._get_current_data()
        
        # Budget progress
        budget_used = data['total_cost']
        budget_limit = st.session_state.get('max_budget', 10.0)
        budget_percentage = (budget_used / budget_limit) * 100 if budget_limit > 0 else 0
        
        st.metric(
            label="Budget Used",
            value=f"{budget_percentage:.1f}%",
            delta=f"${budget_used:.4f} / ${budget_limit:.2f}"
        )
        
        # Progress bar for budget
        st.progress(min(budget_percentage / 100, 1.0))
        
        # Efficiency metrics
        st.subheader("🎯 Efficiency")
        st.metric("Cost per Example", f"${data['cost_per_example']:.6f}")
        st.metric("Tokens per Example", f"{data['tokens_per_example']:,}")
        
        # Early exit stats
        st.subheader("🚪 Early Exits")
        st.metric("Early Exit Rate", f"{data['early_exit_rate']:.1%}")
        st.metric("Most Common Exit", data.get('most_common_exit', 'None'))
    
    def _render_alerts(self):
        """Render alerts and warnings"""
        st.header("⚠️ Alerts")
        
        data = self._get_current_data()
        budget_limit = st.session_state.get('max_budget', 10.0)
        
        alerts = []
        
        # Budget alerts
        if data['total_cost'] > budget_limit * 0.8:
            alerts.append({
                "type": "warning",
                "message": f"Budget usage is at {data['total_cost']/budget_limit*100:.1f}%"
            })
        
        if data['total_cost'] > budget_limit:
            alerts.append({
                "type": "error",
                "message": "Budget exceeded!"
            })
        
        # Performance alerts
        if data['avg_confidence'] < 0.5:
            alerts.append({
                "type": "warning",
                "message": "Low average confidence detected"
            })
        
        if data['early_exit_rate'] > 0.8:
            alerts.append({
                "type": "info",
                "message": "High early exit rate - consider adjusting thresholds"
            })
        
        # Display alerts
        for alert in alerts:
            if alert["type"] == "error":
                st.error(alert["message"])
            elif alert["type"] == "warning":
                st.warning(alert["message"])
            else:
                st.info(alert["message"])
        
        if not alerts:
            st.success("All systems operational! 🎉")
    
    def _render_detailed_analysis(self):
        """Render detailed analysis section"""
        st.header("🔍 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Stage Performance", "Cost Analysis", "Optimization"])
        
        with tab1:
            self._render_stage_performance()
        
        with tab2:
            self._render_cost_analysis()
        
        with tab3:
            self._render_optimization_recommendations()
    
    def _render_stage_performance(self):
        """Render stage performance analysis"""
        stage_data = self._get_stage_performance_data()
        
        if not stage_data.empty:
            # Performance metrics table
            st.subheader("Stage Performance Metrics")
            st.dataframe(stage_data, use_container_width=True)
            
            # Performance visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Confidence by Stage", "Cost by Stage", 
                              "Token Usage by Stage", "Early Exit Rate by Stage")
            )
            
            # Confidence
            fig.add_trace(
                go.Bar(x=stage_data['stage_id'], y=stage_data['avg_confidence'],
                      name="Confidence"),
                row=1, col=1
            )
            
            # Cost
            fig.add_trace(
                go.Bar(x=stage_data['stage_id'], y=stage_data['total_cost'],
                      name="Cost"),
                row=1, col=2
            )
            
            # Tokens
            fig.add_trace(
                go.Bar(x=stage_data['stage_id'], y=stage_data['total_tokens'],
                      name="Tokens"),
                row=2, col=1
            )
            
            # Early exit rate
            fig.add_trace(
                go.Bar(x=stage_data['stage_id'], y=stage_data['early_exit_rate'],
                      name="Early Exit Rate"),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stage performance data available.")
    
    def _render_cost_analysis(self):
        """Render cost analysis"""
        cost_data = self._get_cost_analysis_data()
        
        if not cost_data.empty:
            # Cost breakdown pie chart
            fig_pie = px.pie(
                cost_data,
                values='total_cost',
                names='stage_id',
                title="Cost Breakdown by Stage"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Cost efficiency scatter plot
            fig_scatter = px.scatter(
                cost_data,
                x='avg_confidence',
                y='total_cost',
                size='total_tokens',
                color='stage_id',
                title="Cost vs Confidence Efficiency",
                labels={'avg_confidence': 'Average Confidence', 'total_cost': 'Total Cost (USD)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No cost analysis data available.")
    
    def _render_optimization_recommendations(self):
        """Render optimization recommendations"""
        recommendations = self._get_optimization_recommendations()
        
        if recommendations:
            st.subheader("💡 Optimization Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"Recommendation {i}: {rec['type']}"):
                    st.write(f"**Stage:** {rec['stage_id']}")
                    st.write(f"**Description:** {rec['description']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Potential Savings", f"${rec['potential_savings_cost']:.4f}")
                    with col2:
                        st.metric("Token Savings", f"{rec['potential_savings_tokens']:,}")
                    with col3:
                        st.metric("Accuracy Impact", f"{rec['potential_accuracy_loss']:.3f}")
                    
                    st.progress(rec['confidence'])
                    st.caption(f"Confidence: {rec['confidence']:.1%}")
        else:
            st.info("No optimization recommendations at this time.")
    
    def _get_current_data(self) -> Dict[str, Any]:
        """Get current pipeline data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get latest data
                query = """
                    SELECT 
                        SUM(cost_usd) as total_cost,
                        SUM(total_tokens) as total_tokens,
                        COUNT(DISTINCT example_id) as total_examples,
                        AVG(1.0) as avg_confidence  -- Placeholder
                    FROM usage_log
                """
                
                if st.session_state.get('selected_run') and st.session_state['selected_run'] != "All Runs":
                    query += " WHERE run_id = ?"
                    result = conn.execute(query, [st.session_state['selected_run']]).fetchone()
                else:
                    result = conn.execute(query).fetchone()
                
                if result:
                    total_cost, total_tokens, total_examples, avg_confidence = result
                    
                    # Calculate deltas (simplified - would need time-based comparison)
                    cost_delta = 0.0
                    tokens_delta = 0
                    examples_delta = 0
                    confidence_delta = 0.0
                    
                    return {
                        'total_cost': total_cost or 0.0,
                        'total_tokens': total_tokens or 0,
                        'total_examples': total_examples or 0,
                        'avg_confidence': avg_confidence or 0.0,
                        'cost_delta': cost_delta,
                        'tokens_delta': tokens_delta,
                        'examples_delta': examples_delta,
                        'confidence_delta': confidence_delta,
                        'cost_per_example': (total_cost or 0.0) / max(total_examples or 1, 1),
                        'tokens_per_example': (total_tokens or 0) / max(total_examples or 1, 1),
                        'early_exit_rate': 0.0,  # Would need to calculate from traces
                        'most_common_exit': 'None'
                    }
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
        
        return {
            'total_cost': 0.0,
            'total_tokens': 0,
            'total_examples': 0,
            'avg_confidence': 0.0,
            'cost_delta': 0.0,
            'tokens_delta': 0,
            'examples_delta': 0,
            'confidence_delta': 0.0,
            'cost_per_example': 0.0,
            'tokens_per_example': 0,
            'early_exit_rate': 0.0,
            'most_common_exit': 'None'
        }
    
    def _get_usage_data(self) -> pd.DataFrame:
        """Get usage data for charts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        timestamp,
                        SUM(cost_usd) OVER (ORDER BY timestamp) as cumulative_cost,
                        SUM(total_tokens) OVER (ORDER BY timestamp) as cumulative_tokens
                    FROM usage_log
                    ORDER BY timestamp
                """
                
                if st.session_state.get('selected_run') and st.session_state['selected_run'] != "All Runs":
                    query = query.replace("FROM usage_log", "FROM usage_log WHERE run_id = ?")
                    df = pd.read_sql_query(query, conn, params=[st.session_state['selected_run']])
                else:
                    df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                return df
                
        except Exception as e:
            st.error(f"Error loading usage data: {e}")
        
        return pd.DataFrame()
    
    def _get_stage_usage_data(self) -> pd.DataFrame:
        """Get stage usage data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        stage_id,
                        SUM(prompt_tokens) as prompt_tokens,
                        SUM(completion_tokens) as completion_tokens
                    FROM usage_log
                    GROUP BY stage_id
                """
                
                if st.session_state.get('selected_run') and st.session_state['selected_run'] != "All Runs":
                    query += " WHERE run_id = ?"
                    df = pd.read_sql_query(query, conn, params=[st.session_state['selected_run']])
                else:
                    df = pd.read_sql_query(query, conn)
                
                return df
                
        except Exception as e:
            st.error(f"Error loading stage usage data: {e}")
        
        return pd.DataFrame()
    
    def _get_stage_performance_data(self) -> pd.DataFrame:
        """Get stage performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        stage_id,
                        COUNT(*) as total_calls,
                        SUM(cost_usd) as total_cost,
                        SUM(total_tokens) as total_tokens,
                        AVG(1.0) as avg_confidence  -- Placeholder
                    FROM usage_log
                    GROUP BY stage_id
                """
                
                if st.session_state.get('selected_run') and st.session_state['selected_run'] != "All Runs":
                    query += " WHERE run_id = ?"
                    df = pd.read_sql_query(query, conn, params=[st.session_state['selected_run']])
                else:
                    df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    df['early_exit_rate'] = 0.0  # Would need to calculate from traces
                
                return df
                
        except Exception as e:
            st.error(f"Error loading stage performance data: {e}")
        
        return pd.DataFrame()
    
    def _get_cost_analysis_data(self) -> pd.DataFrame:
        """Get cost analysis data"""
        return self._get_stage_performance_data()
    
    def _get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations"""
        try:
            # Get stage performance data to analyze
            stage_data = self._get_stage_performance_data()
            
            if stage_data.empty:
                return []
            
            recommendations = []
            
            # Analyze each stage for potential optimizations
            for _, row in stage_data.iterrows():
                stage_id = row['stage_id']
                total_cost = row['total_cost']
                total_tokens = row['total_tokens']
                avg_confidence = row['avg_confidence']
                
                # Recommendation 1: High cost, low confidence stages
                if total_cost > 0.5 and avg_confidence < 0.6:
                    recommendations.append({
                        'type': 'Consider Disabling',
                        'stage_id': stage_id,
                        'description': f'High cost (${total_cost:.4f}) with low confidence ({avg_confidence:.3f})',
                        'potential_savings_cost': total_cost * 0.8,
                        'potential_savings_tokens': int(total_tokens * 0.8),
                        'potential_accuracy_loss': 0.1,
                        'confidence': 0.7
                    })
                
                # Recommendation 2: Very expensive stages
                if total_cost > 1.0:
                    recommendations.append({
                        'type': 'Cost Optimization',
                        'stage_id': stage_id,
                        'description': f'Very high cost stage (${total_cost:.4f})',
                        'potential_savings_cost': total_cost * 0.5,
                        'potential_savings_tokens': int(total_tokens * 0.5),
                        'potential_accuracy_loss': 0.2,
                        'confidence': 0.6
                    })
                
                # Recommendation 3: High token usage stages
                if total_tokens > 10000:
                    recommendations.append({
                        'type': 'Token Optimization',
                        'stage_id': stage_id,
                        'description': f'High token usage ({total_tokens:,} tokens)',
                        'potential_savings_cost': total_cost * 0.3,
                        'potential_savings_tokens': int(total_tokens * 0.3),
                        'potential_accuracy_loss': 0.15,
                        'confidence': 0.5
                    })
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            st.error(f"Error generating optimization recommendations: {e}")
            return []
    
    def _get_available_runs(self) -> List[str]:
        """Get list of available pipeline runs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT DISTINCT run_id FROM usage_log WHERE run_id IS NOT NULL"
                result = conn.execute(query).fetchall()
                return [row[0] for row in result]
        except Exception:
            return []
    
    def _get_available_stages(self) -> List[str]:
        """Get list of available stages"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT DISTINCT stage_id FROM usage_log"
                result = conn.execute(query).fetchall()
                return [row[0] for row in result]
        except Exception:
            return []


def run_dashboard(db_path: str = "usage_tracking.db"):
    """Run the pipeline dashboard"""
    dashboard = PipelineDashboard(db_path)
    dashboard.run_dashboard()


if __name__ == "__main__":
    run_dashboard()
