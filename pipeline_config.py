# Pipeline Configuration
# This file contains different pipeline configurations for various datasets and use cases

# Default configuration - works for most datasets
DEFAULT_CONFIG = {
    "stages": [
        {"type": "baseline", "id": "s0", "model": "target"},
        {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
        {"type": "cove", "id": "s2", "model": "target"},
        {"type": "self_correct", "id": "s3", "model": "target"},
        {"type": "confidence_check", "id": "s4", "model": "target"}
    ],
    "gate": {"mode": "oracle"},
    "token_diffs": True,
    "debug_context": False,
    "evaluate_all_stages": True
}

# SimpleQA optimized configuration
SIMPLEQA_CONFIG = {
    "stages": [
        {"type": "baseline", "id": "s0", "model": "target"},
        {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
        {"type": "cove", "id": "s2", "model": "target"},
        {"type": "self_correct", "id": "s3", "model": "target"}
    ],
    "gate": {
        "mode": "metric",
        "criteria": "em>=0.8||f1>=0.7"  # Exit on high exact match or F1 score
    },
    "token_diffs": True,
    "debug_context": False,
    "evaluate_all_stages": True
}

# TruthfulQA optimized configuration
TRUTHFULQA_CONFIG = {
    "stages": [
        {"type": "baseline", "id": "s0", "model": "target"},
        {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
        {"type": "cove", "id": "s2", "model": "target"},
        {"type": "self_correct", "id": "s3", "model": "target"},
        {"type": "confidence_check", "id": "s4", "model": "target"}
    ],
    "gate": {
        "mode": "metric",
        "criteria": "mc_accuracy>=0.7&&rouge1>=0.4||bleurt>=0.5||llm_judge>=0.8"
    },
    "token_diffs": True,
    "debug_context": False,
    "evaluate_all_stages": True
}

# Conservative configuration - higher thresholds, more stages
CONSERVATIVE_CONFIG = {
    "stages": [
        {"type": "baseline", "id": "s0", "model": "target"},
        {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
        {"type": "cove", "id": "s2", "model": "target"},
        {"type": "self_correct", "id": "s3", "model": "target"},
        {"type": "judge", "id": "s4", "judge": "judge", "exit_on_pass": True, "threshold": 0.8},
        {"type": "confidence_check", "id": "s5", "model": "target"}
    ],
    "gate": {
        "mode": "metric",
        "criteria": "mc_accuracy>=0.8&&rouge1>=0.6||bleurt>=0.6||llm_judge>=0.9"
    },
    "token_diffs": True,
    "debug_context": False,
    "evaluate_all_stages": True
}

# Fast configuration - minimal stages, aggressive gates
FAST_CONFIG = {
    "stages": [
        {"type": "baseline", "id": "s0", "model": "target"},
        {"type": "apo", "id": "s1", "helper": "helper", "target": "target"}
    ],
    "gate": {
        "mode": "metric", 
        "criteria": "em>=0.6||f1>=0.5||mc_accuracy>=0.6"
    },
    "token_diffs": False,
    "debug_context": False,
    "evaluate_all_stages": False
}

# No early exit - run all stages for comprehensive evaluation
COMPREHENSIVE_CONFIG = {
    "stages": [
        {"type": "baseline", "id": "s0", "model": "target"},
        {"type": "apo", "id": "s1", "helper": "helper", "target": "target"},
        {"type": "cove", "id": "s2", "model": "target"},
        {"type": "self_correct", "id": "s3", "model": "target"},
        {"type": "judge", "id": "s4", "judge": "judge", "exit_on_pass": False, "threshold": 0.8},
        {"type": "confidence_check", "id": "s5", "model": "target"}
    ],
    "gate": {"mode": "none"},  # No early exit
    "token_diffs": True,
    "debug_context": False,
    "evaluate_all_stages": True
}

# Available configurations mapping
CONFIGS = {
    "default": DEFAULT_CONFIG,
    "simpleqa": SIMPLEQA_CONFIG,
    "truthfulqa": TRUTHFULQA_CONFIG,
    "conservative": CONSERVATIVE_CONFIG,
    "fast": FAST_CONFIG,
    "comprehensive": COMPREHENSIVE_CONFIG
}

def get_config(name: str = "default", dataset_schema: str = None):
    """
    Get a pipeline configuration by name or automatically select based on dataset
    
    Args:
        name: Configuration name from CONFIGS
        dataset_schema: Dataset schema (simpleqa, truthfulqa) for auto-selection
    
    Returns:
        Configuration dictionary
    """
    if name == "auto" and dataset_schema:
        if dataset_schema == "simpleqa":
            return CONFIGS["simpleqa"]
        elif dataset_schema == "truthfulqa":
            return CONFIGS["truthfulqa"]
        else:
            return CONFIGS["default"]
    
    if name in CONFIGS:
        return CONFIGS[name]
    
    print(f"ERROR: Configuration '{name}' not found!")
    print(f"Available configurations: {list(CONFIGS.keys())}")
    print("Using 'default' configuration instead.")
    print()
    return CONFIGS["default"]

def list_configs():
    """List all available configurations with descriptions"""
    descriptions = {
        "default": "Basic configuration with oracle gate - good for testing",
        "simpleqa": "Optimized for SimpleQA - exits on high EM/F1 scores",
        "truthfulqa": "Optimized for TruthfulQA - comprehensive metric criteria",
        "conservative": "High thresholds, all stages - maximum quality",
        "fast": "Minimal stages, aggressive gates - speed optimized",
        "comprehensive": "All stages, no early exit - full evaluation"
    }
    
    print("Available pipeline configurations:")
    for name, desc in descriptions.items():
        print(f"  {name:15} - {desc}")
    print()
    print("Usage in .env file:")
    print("PIPELINE_CONFIG=truthfulqa")
    print("# or")
    print("PIPELINE_CONFIG=auto  # Auto-select based on dataset")