from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import sqlite3
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.core.core_types import RunTrace, StageResult
from .tracking import UsageTracker


@dataclass
class StageAnalysis:
    """Analysis results for a single stage"""
    stage_id: str
    total_examples: int
    total_tokens: int
    total_cost_usd: float
    avg_confidence: float
    early_exit_rate: float
    marginal_accuracy_gain: float = 0.0
    cost_per_accuracy_point: float = 0.0
    efficiency_score: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Recommendation for pipeline optimization"""
    type: str  # "disable_stage", "adjust_threshold", "early_exit"
    stage_id: str
    description: str
    potential_savings_tokens: int
    potential_savings_cost: float
    potential_accuracy_loss: float
    confidence: float


class PipelineAnalyzer:
    """Analyzes pipeline performance and provides optimization recommendations"""
    
    def __init__(self, db_path: str = "usage_tracking.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("pipeline_analyzer")
    
    def analyze_run_traces(self, traces: List[RunTrace]) -> Dict[str, Any]:
        """Analyze a list of run traces"""
        if not traces:
            return {}
        
        # Stage-level analysis
        stage_stats = self._analyze_stages(traces)
        
        # Early exit analysis
        early_exit_stats = self._analyze_early_exits(traces)
        
        # Cost-benefit analysis
        cost_benefit = self._analyze_cost_benefit(traces)
        
        # Optimization recommendations
        recommendations = self._generate_recommendations(stage_stats, cost_benefit)
        
        return {
            "stage_analysis": stage_stats,
            "early_exit_analysis": early_exit_stats,
            "cost_benefit_analysis": cost_benefit,
            "optimization_recommendations": recommendations,
            "summary": self._generate_summary(traces, stage_stats)
        }
    
    def _analyze_stages(self, traces: List[RunTrace]) -> Dict[str, StageAnalysis]:
        """Analyze performance of each stage"""
        stage_data: Dict[str, List[Dict[str, Any]]] = {}
        
        for trace in traces:
            for i, result in enumerate(trace.stage_results):
                stage_id = result.stage_id
                if stage_id not in stage_data:
                    stage_data[stage_id] = []
                
                # Calculate tokens and cost for this stage
                stage_tokens = 0
                stage_cost = 0.0
                if isinstance(result.model_usage, dict):
                    if "prompt_tokens" in result.model_usage:
                        stage_tokens += result.model_usage.get("prompt_tokens", 0)
                        stage_tokens += result.model_usage.get("completion_tokens", 0)
                        stage_cost += result.model_usage.get("cost", 0.0)
                    else:
                        # Multiple model usage (e.g., APO stage)
                        for usage in result.model_usage.values():
                            if isinstance(usage, dict):
                                stage_tokens += usage.get("prompt_tokens", 0)
                                stage_tokens += usage.get("completion_tokens", 0)
                                stage_cost += usage.get("cost", 0.0)
                
                stage_data[stage_id].append({
                    "tokens": stage_tokens,
                    "cost": stage_cost,
                    "confidence": result.confidence,
                    "should_exit": result.should_exit,
                    "stage_index": i
                })
        
        # Calculate statistics for each stage
        stage_analysis = {}
        for stage_id, data in stage_data.items():
            if not data:
                continue
            
            total_examples = len(data)
            total_tokens = sum(d["tokens"] for d in data)
            total_cost = sum(d["cost"] for d in data)
            avg_confidence = np.mean([d["confidence"] for d in data])
            early_exit_rate = sum(1 for d in data if d["should_exit"]) / total_examples
            
            stage_analysis[stage_id] = StageAnalysis(
                stage_id=stage_id,
                total_examples=total_examples,
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                avg_confidence=avg_confidence,
                early_exit_rate=early_exit_rate
            )
        
        return stage_analysis
    
    def _analyze_early_exits(self, traces: List[RunTrace]) -> Dict[str, Any]:
        """Analyze early exit patterns"""
        exit_counts = {}
        exit_stages = {}
        
        for trace in traces:
            if trace.early_exit_at:
                exit_counts[trace.early_exit_at] = exit_counts.get(trace.early_exit_at, 0) + 1
                exit_stages[trace.early_exit_at] = exit_stages.get(trace.early_exit_at, 0) + 1
        
        total_examples = len(traces)
        total_early_exits = sum(exit_counts.values())
        
        return {
            "total_examples": total_examples,
            "total_early_exits": total_early_exits,
            "early_exit_rate": total_early_exits / total_examples if total_examples > 0 else 0,
            "exit_distribution": exit_counts,
            "exit_by_stage": exit_stages
        }
    
    def _analyze_cost_benefit(self, traces: List[RunTrace]) -> Dict[str, Any]:
        """Analyze cost-benefit trade-offs"""
        # This is a simplified analysis - in practice you'd need ground truth labels
        # to calculate actual accuracy improvements
        
        stage_benefits = {}
        for trace in traces:
            for i, result in enumerate(trace.stage_results):
                stage_id = result.stage_id
                if stage_id not in stage_benefits:
                    stage_benefits[stage_id] = []
                
                # Estimate benefit based on confidence improvement
                # This is a heuristic - real analysis would use actual accuracy metrics
                benefit = result.confidence * 0.1  # Simplified benefit estimation
                stage_benefits[stage_id].append(benefit)
        
        cost_benefit_analysis = {}
        for stage_id, benefits in stage_benefits.items():
            avg_benefit = np.mean(benefits) if benefits else 0.0
            cost_benefit_analysis[stage_id] = {
                "avg_benefit": avg_benefit,
                "benefit_variance": np.var(benefits) if benefits else 0.0
            }
        
        return cost_benefit_analysis
    
    def _generate_recommendations(self, stage_stats: Dict[str, StageAnalysis], 
                                cost_benefit: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for stage_id, stats in stage_stats.items():
            # Check for expensive but low-impact stages
            if stats.total_cost_usd > 0.1 and stats.avg_confidence < 0.6:
                recommendations.append(OptimizationRecommendation(
                    type="disable_stage",
                    stage_id=stage_id,
                    description=f"Stage {stage_id} is expensive (${stats.total_cost_usd:.4f}) but has low confidence ({stats.avg_confidence:.2f})",
                    potential_savings_tokens=stats.total_tokens,
                    potential_savings_cost=stats.total_cost_usd,
                    potential_accuracy_loss=0.05,  # Estimate
                    confidence=0.7
                ))
            
            # Check for high confidence stages that could use early exit
            if stats.avg_confidence > 0.8 and stats.early_exit_rate < 0.5:
                recommendations.append(OptimizationRecommendation(
                    type="adjust_threshold",
                    stage_id=stage_id,
                    description=f"Stage {stage_id} has high confidence ({stats.avg_confidence:.2f}) but low early exit rate ({stats.early_exit_rate:.2f})",
                    potential_savings_tokens=int(stats.total_tokens * 0.3),
                    potential_savings_cost=stats.total_cost_usd * 0.3,
                    potential_accuracy_loss=0.02,
                    confidence=0.8
                ))
        
        return recommendations
    
    def _generate_summary(self, traces: List[RunTrace], stage_stats: Dict[str, StageAnalysis]) -> Dict[str, Any]:
        """Generate overall summary statistics"""
        total_tokens = sum(trace.total_tokens for trace in traces)
        total_cost = sum(trace.total_cost for trace in traces)
        total_time = sum(trace.timing_sec for trace in traces)
        
        return {
            "total_examples": len(traces),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "total_time_sec": total_time,
            "avg_tokens_per_example": total_tokens / len(traces) if traces else 0,
            "avg_cost_per_example": total_cost / len(traces) if traces else 0,
            "avg_time_per_example": total_time / len(traces) if traces else 0,
            "stage_count": len(stage_stats)
        }
    
    def analyze_database(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze data from the tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get usage data
                query = """
                    SELECT stage_id, model_name, 
                           SUM(prompt_tokens) as total_prompt_tokens,
                           SUM(completion_tokens) as total_completion_tokens,
                           SUM(total_tokens) as total_tokens,
                           SUM(cost_usd) as total_cost_usd,
                           COUNT(*) as call_count
                    FROM usage_log
                """
                if run_id:
                    query += " WHERE run_id = ?"
                    query += " GROUP BY stage_id, model_name"
                    df = pd.read_sql_query(query, conn, params=[run_id])
                else:
                    query += " GROUP BY stage_id, model_name"
                    df = pd.read_sql_query(query, conn)
                
                # Get pipeline run data
                run_query = "SELECT * FROM pipeline_runs"
                if run_id:
                    run_query += " WHERE run_id = ?"
                    run_df = pd.read_sql_query(run_query, conn, params=[run_id])
                else:
                    run_df = pd.read_sql_query(run_query, conn)
                
                return {
                    "usage_by_stage": df.to_dict('records'),
                    "pipeline_runs": run_df.to_dict('records'),
                    "summary": {
                        "total_stages": len(df),
                        "total_cost": df['total_cost_usd'].sum() if not df.empty else 0,
                        "total_tokens": df['total_tokens'].sum() if not df.empty else 0,
                        "total_calls": df['call_count'].sum() if not df.empty else 0
                    }
                }
        except Exception as e:
            self.logger.error(f"Failed to analyze database: {e}")
            return {}


def analyze_results(trace_file: str, db_conn: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to analyze results from trace file and database"""
    analyzer = PipelineAnalyzer(db_conn or "usage_tracking.db")
    
    # Load traces from file
    traces = []
    try:
        with open(trace_file, 'r') as f:
            for line in f:
                if line.strip():
                    trace_data = json.loads(line)
                    # Convert back to RunTrace object (simplified)
                    traces.append(trace_data)
    except Exception as e:
        logging.error(f"Failed to load trace file: {e}")
        return {}
    
    # Analyze traces
    return analyzer.analyze_run_traces(traces)
