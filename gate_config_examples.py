# Example gate configurations for different datasets and use cases

# Example configurations showing different gate modes and criteria
GATE_EXAMPLES = {
    
    # SimpleQA - Exit on high exact match or F1 score
    "simpleqa_gates": {
        "high_precision": {
            "mode": "metric",
            "criteria": "em>=0.9||f1>=0.8"  # Exit if exact match >=90% OR F1 >=80%
        },
        "moderate": {
            "mode": "metric", 
            "criteria": "em>=0.7||f1>=0.6"  # Exit if exact match >=70% OR F1 >=60%
        }
    },
    
    # TruthfulQA - More complex criteria combining multiple metrics
    "truthfulqa_gates": {
        "conservative": {
            "mode": "metric",
            "criteria": "mc_accuracy>=0.8&&rouge1>=0.5"  # Both MC accuracy >=80% AND ROUGE-1 >=50%
        },
        "balanced": {
            "mode": "metric",
            "criteria": "mc_accuracy>=0.7||rouge1>=0.6||bleurt>=0.3"  # Any one of: MC >=70%, ROUGE >=60%, or BLEURT >=0.3
        },
        "aggressive": {
            "mode": "metric", 
            "criteria": "mc_accuracy>=0.6||llm_judge>=0.7"  # MC accuracy >=60% OR LLM judge >=70%
        },
        "comprehensive": {
            "mode": "metric",
            "criteria": "mc_accuracy>=0.7&&rouge1>=0.4||bleurt>=0.5||llm_judge>=0.8"  # (MC >=70% AND ROUGE >=40%) OR BLEURT >=0.5 OR LLM judge >=80%
        }
    },
    
    # Other gate modes for comparison
    "other_modes": {
        "oracle": {
            "mode": "oracle"  # Uses ground truth (perfect for offline evaluation)
        },
        "judge": {
            "mode": "judge",
            "judge": "judge",
            "threshold": 0.7  # LLM judge with 70% confidence threshold
        },
        "no_gate": {
            "mode": "none"  # No early exit - run all stages
        }
    }
}

# Usage examples for main.py config:

# Example 1: TruthfulQA with balanced criteria
TRUTHFULQA_CONFIG = {
    "stages": [
        {"type":"baseline","id":"s0","model":"target"},
        {"type":"apo","id":"s1","helper":"helper","target":"target"},
        {"type":"cove","id":"s2","model":"target"},
        {"type":"self_correct","id":"s3","model":"target"},
        {"type":"confidence_check","id":"s4","model":"target"}
    ],
    "gate": {
        "mode": "metric",
        "criteria": "mc_accuracy>=0.7||rouge1>=0.6||bleurt>=0.3"
    },
    "token_diffs": True,
    "evaluate_all_stages": True
}

# Example 2: SimpleQA with high precision
SIMPLEQA_CONFIG = {
    "stages": [
        {"type":"baseline","id":"s0","model":"target"},
        {"type":"apo","id":"s1","helper":"helper","target":"target"},
        {"type":"cove","id":"s2","model":"target"},
        {"type":"self_correct","id":"s3","model":"target"}
    ],
    "gate": {
        "mode": "metric", 
        "criteria": "em>=0.9||f1>=0.8"
    },
    "token_diffs": True
}

# Metric explanation:
"""
Available metrics for gate criteria:
- em: Exact Match (0.0-1.0) - Perfect string match after normalization
- f1: F1 Score (0.0-1.0) - Token-level overlap score  
- rouge1: ROUGE-1 (0.0-1.0) - Unigram overlap score
- bleurt: BLEURT (-2.0 to 1.0) - Neural semantic similarity score
- mc_accuracy: Multiple Choice Accuracy (0.0-1.0) - Correct answer selection
- llm_judge: LLM Judge Score (0.0-1.0) - LLM evaluation of answer quality

Operators:
- >: Greater than
- >=: Greater than or equal to  
- &&: AND (both conditions must be true)
- ||: OR (either condition can be true)

Examples:
- "em>=0.8" - Exit if exact match is 80% or higher
- "mc_accuracy>=0.7&&rouge1>=0.5" - Exit if BOTH MC accuracy >=70% AND ROUGE >=50%  
- "em>=0.9||f1>=0.7||bleurt>=0.4" - Exit if ANY of: EM >=90% OR F1 >=70% OR BLEURT >=0.4
- "mc_accuracy>=0.6&&rouge1>=0.3||llm_judge>=0.8" - Exit if (MC >=60% AND ROUGE >=30%) OR LLM judge >=80%
"""