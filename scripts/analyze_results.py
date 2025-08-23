import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.analytics.analysis import analyze_results
from src.analytics.visualization import create_visualization_report


def main():
    parser = argparse.ArgumentParser(description="Analyze pipeline results")
    parser.add_argument("trace_file", help="Path to trace file (JSONL format)")
    parser.add_argument("--db-path", default="usage_tracking.db", help="Database path")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory for plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Check if trace file exists
    if not Path(args.trace_file).exists():
        print(f"Error: Trace file {args.trace_file} not found")
        sys.exit(1)
    
    print(f"Analyzing results from {args.trace_file}")
    
    # Run analysis
    analysis_results = analyze_results(args.trace_file, args.db_path)
    
    if analysis_results:
        print("\n=== Analysis Results ===")
        
        # Print summary
        summary = analysis_results.get("summary", {})
        if summary:
            print(f"Total examples: {summary.get('total_examples', 0)}")
            print(f"Total tokens: {summary.get('total_tokens', 0):,}")
            print(f"Total cost: ${summary.get('total_cost_usd', 0):.6f}")
            print(f"Average time per example: {summary.get('avg_time_per_example', 0):.3f}s")
        
        # Print stage analysis
        stage_analysis = analysis_results.get("stage_analysis", {})
        if stage_analysis:
            print("\n=== Stage Analysis ===")
            for stage_id, stats in stage_analysis.items():
                print(f"\nStage {stage_id}:")
                print(f"  Total examples: {stats.total_examples}")
                print(f"  Total tokens: {stats.total_tokens:,}")
                print(f"  Total cost: ${stats.total_cost_usd:.6f}")
                print(f"  Average confidence: {stats.avg_confidence:.3f}")
                print(f"  Early exit rate: {stats.early_exit_rate:.1%}")
        
        # Print early exit analysis
        early_exit = analysis_results.get("early_exit_analysis", {})
        if early_exit:
            print(f"\n=== Early Exit Analysis ===")
            print(f"Early exit rate: {early_exit.get('early_exit_rate', 0):.1%}")
            print(f"Exit distribution: {early_exit.get('exit_distribution', {})}")
        
        # Print optimization recommendations
        recommendations = analysis_results.get("optimization_recommendations", [])
        if recommendations:
            print(f"\n=== Optimization Recommendations ===")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec.type.upper()}: {rec.stage_id}")
                print(f"   Description: {rec.description}")
                print(f"   Potential savings: {rec.potential_savings_tokens:,} tokens, ${rec.potential_savings_cost:.4f}")
                print(f"   Potential accuracy loss: {rec.potential_accuracy_loss:.3f}")
                print(f"   Confidence: {rec.confidence:.1%}")
        else:
            print("\nNo optimization recommendations at this time.")
        
        # Generate plots
        if not args.no_plots:
            print(f"\nGenerating plots in {args.output_dir}...")
            try:
                saved_plots = create_visualization_report(args.trace_file, args.db_path, args.output_dir)
                print(f"Generated {len(saved_plots)} plots:")
                for plot_name, plot_path in saved_plots.items():
                    print(f"  - {plot_name}: {plot_path}")
            except Exception as e:
                print(f"Error generating plots: {e}")
    
    else:
        print("No analysis results generated.")


if __name__ == "__main__":
    main()
