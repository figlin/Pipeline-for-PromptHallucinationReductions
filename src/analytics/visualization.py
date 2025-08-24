from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3
from datetime import datetime

from src.analytics.analysis import PipelineAnalyzer, StageAnalysis
from src.core.core_types import RunTrace


class PipelineVisualizer:
    """Generates publication-quality visualizations for pipeline analysis"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        self.style = style
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
    
    def plot_accuracy_vs_tokens(self, traces: List[RunTrace], save_path: Optional[str] = None) -> plt.Figure:
        """Plot accuracy vs token usage per stage"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        stage_data = {}
        for trace in traces:
            for result in trace.stage_results:
                stage_id = result.stage_id
                if stage_id not in stage_data:
                    stage_data[stage_id] = {"tokens": [], "confidence": []}
                
                # Calculate tokens for this stage
                stage_tokens = 0
                if isinstance(result.model_usage, dict):
                    if "prompt_tokens" in result.model_usage:
                        stage_tokens += result.model_usage.get("prompt_tokens", 0)
                        stage_tokens += result.model_usage.get("completion_tokens", 0)
                    else:
                        for usage in result.model_usage.values():
                            if isinstance(usage, dict):
                                stage_tokens += usage.get("prompt_tokens", 0)
                                stage_tokens += usage.get("completion_tokens", 0)
                
                stage_data[stage_id]["tokens"].append(stage_tokens)
                stage_data[stage_id]["confidence"].append(result.confidence)
        
        # Plot each stage
        for i, (stage_id, data) in enumerate(stage_data.items()):
            ax.scatter(data["tokens"], data["confidence"], 
                      label=stage_id, alpha=0.6, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel("Token Usage")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Accuracy vs Token Usage by Stage")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_early_exit_distribution(self, traces: List[RunTrace], save_path: Optional[str] = None) -> plt.Figure:
        """Plot early exit distribution across stages"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count exits by stage
        exit_counts = {}
        total_examples = len(traces)
        
        for trace in traces:
            if trace.early_exit_at:
                exit_counts[trace.early_exit_at] = exit_counts.get(trace.early_exit_at, 0) + 1
        
        # Bar plot of exit counts
        if exit_counts:
            stages = list(exit_counts.keys())
            counts = list(exit_counts.values())
            
            bars = ax1.bar(stages, counts, color=self.colors[:len(stages)])
            ax1.set_xlabel("Exit Stage")
            ax1.set_ylabel("Number of Examples")
            ax1.set_title("Early Exit Distribution")
            ax1.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for bar, count in zip(bars, counts):
                percentage = (count / total_examples) * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Pie chart of exit vs no exit
        no_exit = total_examples - sum(exit_counts.values())
        exit_data = [no_exit] + list(exit_counts.values())
        exit_labels = ["No Early Exit"] + list(exit_counts.keys())
        
        ax2.pie(exit_data, labels=exit_labels, autopct='%1.1f%%', 
                colors=self.colors[:len(exit_data)])
        ax2.set_title("Early Exit vs Full Pipeline")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cost_breakdown(self, stage_stats: Dict[str, StageAnalysis], save_path: Optional[str] = None) -> plt.Figure:
        """Plot cost breakdown by stage"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        stages = list(stage_stats.keys())
        costs = [stats.total_cost_usd for stats in stage_stats.values()]
        tokens = [stats.total_tokens for stats in stage_stats.values()]
        
        # Cost breakdown
        bars1 = ax1.bar(stages, costs, color=self.colors[:len(stages)])
        ax1.set_xlabel("Stage")
        ax1.set_ylabel("Cost (USD)")
        ax1.set_title("Cost Breakdown by Stage")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add cost labels
        for bar, cost in zip(bars1, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'${cost:.4f}', ha='center', va='bottom')
        
        # Token breakdown
        bars2 = ax2.bar(stages, tokens, color=self.colors[:len(stages)])
        ax2.set_xlabel("Stage")
        ax2.set_ylabel("Total Tokens")
        ax2.set_title("Token Usage by Stage")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add token labels
        for bar, token_count in zip(bars2, tokens):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{token_count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confidence_distribution(self, traces: List[RunTrace], save_path: Optional[str] = None) -> plt.Figure:
        """Plot confidence score distribution by stage"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract confidence data by stage
        stage_confidences = {}
        for trace in traces:
            for result in trace.stage_results:
                stage_id = result.stage_id
                if stage_id not in stage_confidences:
                    stage_confidences[stage_id] = []
                stage_confidences[stage_id].append(result.confidence)
        
        # Create box plot
        data = [stage_confidences[stage_id] for stage_id in stage_confidences.keys()]
        labels = list(stage_confidences.keys())
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel("Stage")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Confidence Score Distribution by Stage")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_efficiency_analysis(self, stage_stats: Dict[str, StageAnalysis], save_path: Optional[str] = None) -> plt.Figure:
        """Plot efficiency analysis (cost per confidence point)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stages = list(stage_stats.keys())
        costs = [stats.total_cost_usd for stats in stage_stats.values()]
        confidences = [stats.avg_confidence for stats in stage_stats.values()]
        
        # Calculate efficiency (cost per confidence point)
        efficiencies = [cost / max(conf, 0.01) for cost, conf in zip(costs, confidences)]
        
        # Create scatter plot
        scatter = ax.scatter(confidences, costs, s=[eff * 1000 for eff in efficiencies], 
                           c=efficiencies, cmap='viridis', alpha=0.7)
        
        # Add stage labels
        for i, stage in enumerate(stages):
            ax.annotate(stage, (confidences[i], costs[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel("Average Confidence")
        ax.set_ylabel("Total Cost (USD)")
        ax.set_title("Stage Efficiency Analysis\n(Size = Cost/Confidence, Color = Efficiency)")
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Efficiency (Cost/Confidence)")
        
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard_figure(self, traces: List[RunTrace], stage_stats: Dict[str, StageAnalysis], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Create a comprehensive dashboard figure"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Cost breakdown (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        stages = list(stage_stats.keys())
        costs = [stats.total_cost_usd for stats in stage_stats.values()]
        ax1.pie(costs, labels=stages, autopct='%1.1f%%', colors=self.colors[:len(stages)])
        ax1.set_title("Cost Breakdown")
        
        # 2. Token usage (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        tokens = [stats.total_tokens for stats in stage_stats.values()]
        ax2.bar(stages, tokens, color=self.colors[:len(stages)])
        ax2.set_title("Token Usage by Stage")
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Confidence distribution (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        confidences = [stats.avg_confidence for stats in stage_stats.values()]
        ax3.bar(stages, confidences, color=self.colors[:len(stages)])
        ax3.set_title("Average Confidence by Stage")
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        
        # 4. Early exit analysis (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        exit_rates = [stats.early_exit_rate for stats in stage_stats.values()]
        ax4.bar(stages, exit_rates, color=self.colors[:len(stages)])
        ax4.set_title("Early Exit Rate by Stage")
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        
        # 5. Cost vs Confidence scatter (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(confidences, costs, s=100, c=self.colors[:len(stages)], alpha=0.7)
        for i, stage in enumerate(stages):
            ax5.annotate(stage, (confidences[i], costs[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax5.set_xlabel("Average Confidence")
        ax5.set_ylabel("Total Cost (USD)")
        ax5.set_title("Cost vs Confidence")
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        total_cost = sum(costs)
        total_tokens = sum(tokens)
        avg_confidence = np.mean(confidences)
        total_examples = len(traces)
        
        summary_text = f"""
        Pipeline Summary
        
        Total Examples: {total_examples}
        Total Cost: ${total_cost:.4f}
        Total Tokens: {total_tokens:,}
        Avg Confidence: {avg_confidence:.3f}
        Stages: {len(stages)}
        """
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='center')
        
        # 7. Early exit distribution (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        exit_counts = {}
        for trace in traces:
            if trace.early_exit_at:
                exit_counts[trace.early_exit_at] = exit_counts.get(trace.early_exit_at, 0) + 1
        
        if exit_counts:
            exit_stages = list(exit_counts.keys())
            exit_nums = list(exit_counts.values())
            ax7.bar(exit_stages, exit_nums, color=self.colors[:len(exit_stages)])
            ax7.set_title("Early Exit Distribution")
            ax7.tick_params(axis='x', rotation=45)
        
        # 8. Timeline analysis (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        times = [trace.timing_sec for trace in traces]
        ax8.hist(times, bins=20, alpha=0.7, color=self.colors[0])
        ax8.set_xlabel("Time (seconds)")
        ax8.set_ylabel("Frequency")
        ax8.set_title("Processing Time Distribution")
        
        # 9. Efficiency analysis (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        efficiencies = [cost / max(conf, 0.01) for cost, conf in zip(costs, confidences)]
        ax9.bar(stages, efficiencies, color=self.colors[:len(stages)])
        ax9.set_title("Cost per Confidence Point")
        ax9.tick_params(axis='x', rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(self, traces: List[RunTrace], stage_stats: Dict[str, StageAnalysis], 
                          output_dir: str = "plots") -> Dict[str, str]:
        """Generate all plots and save them to the output directory"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Generate all plots
        plots = {
            "accuracy_vs_tokens": self.plot_accuracy_vs_tokens(traces),
            "early_exit_distribution": self.plot_early_exit_distribution(traces),
            "cost_breakdown": self.plot_cost_breakdown(stage_stats),
            "confidence_distribution": self.plot_confidence_distribution(traces),
            "efficiency_analysis": self.plot_efficiency_analysis(stage_stats),
            "dashboard": self.create_dashboard_figure(traces, stage_stats)
        }
        
        # Save plots
        for name, fig in plots.items():
            filename = f"{name}_{timestamp}.png"
            filepath = output_path / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files[name] = str(filepath)
        
        return saved_files


def create_visualization_report(trace_file: str, db_path: str = "usage_tracking.db", 
                              output_dir: str = "plots") -> Dict[str, str]:
    """Create a comprehensive visualization report from trace file and database"""
    # Load traces
    traces = []
    try:
        with open(trace_file, 'r') as f:
            for line in f:
                if line.strip():
                    trace_data = json.loads(line)
                    # Convert dict back to RunTrace object
                    from src.core.core_types import RunTrace, StageResult
                    from dataclasses import field
                    
                    # Convert stage_results back to StageResult objects
                    stage_results = []
                    for stage_data in trace_data.get('stage_results', []):
                        stage_result = StageResult(
                            stage_id=stage_data.get('stage_id'),
                            answer=stage_data.get('answer'),
                            confidence=stage_data.get('confidence', 0.0),
                            evidence=stage_data.get('evidence'),
                            model_usage=stage_data.get('model_usage', {}),
                            should_exit=stage_data.get('should_exit', False)
                        )
                        stage_results.append(stage_result)
                    
                    # Create RunTrace object
                    trace = RunTrace(
                        qid=trace_data.get('qid'),
                        question=trace_data.get('question'),
                        stage_results=stage_results,
                        final_answer=trace_data.get('final_answer'),
                        early_exit_at=trace_data.get('early_exit_at'),
                        total_tokens=trace_data.get('total_tokens', 0),
                        total_cost=trace_data.get('total_cost', 0.0),
                        timing_sec=trace_data.get('timing_sec', 0.0),
                        artifacts=trace_data.get('artifacts', {}),
                        y_true=trace_data.get('y_true')
                    )
                    traces.append(trace)
    except Exception as e:
        print(f"Failed to load trace file: {e}")
        return {}
    
    # Analyze traces
    analyzer = PipelineAnalyzer(db_path)
    analysis = analyzer.analyze_run_traces(traces)
    stage_stats = analysis.get("stage_analysis", {})
    
    # Generate visualizations
    visualizer = PipelineVisualizer()
    saved_files = visualizer.generate_all_plots(traces, stage_stats, output_dir)
    
    return saved_files
