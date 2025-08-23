from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time
import logging
from datetime import datetime
import sqlite3
import json
from pathlib import Path

from src.core.core_types import StageResult, RunTrace


@dataclass
class UsageInfo:
    """Structured usage information for a single model call"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model_name: str = ""
    timestamp: float = field(default_factory=time.time)
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
    
    @total_tokens.setter
    def total_tokens(self, value: int):
        pass  # Read-only property


@dataclass
class StageUsage:
    """Aggregated usage for a single stage"""
    stage_id: str
    usage: Dict[str, UsageInfo] = field(default_factory=dict)
    
    @property
    def total_prompt_tokens(self) -> int:
        return sum(u.prompt_tokens for u in self.usage.values())
    
    @property
    def total_completion_tokens(self) -> int:
        return sum(u.completion_tokens for u in self.usage.values())
    
    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.usage.values())
    
    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage.values())


class UsageTracker:
    """Tracks token usage and costs across pipeline runs"""
    
    def __init__(self, max_budget_usd: Optional[float] = None, db_path: Optional[str] = None):
        self.max_budget_usd = max_budget_usd
        self.total_cost_usd = 0.0
        self.total_tokens = 0
        self.stage_usage: Dict[str, StageUsage] = {}
        self.logger = logging.getLogger("usage_tracker")
        
        # Database setup
        self.db_path = db_path or "usage_tracking.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for usage tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        example_id TEXT NOT NULL,
                        stage_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        cost_usd REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        run_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_runs (
                        run_id TEXT PRIMARY KEY,
                        total_examples INTEGER NOT NULL,
                        total_cost_usd REAL NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        config TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_usage_example_id ON usage_log(example_id);
                    CREATE INDEX IF NOT EXISTS idx_usage_stage_id ON usage_log(stage_id);
                    CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp);
                """)
        except Exception as e:
            self.logger.warning(f"Failed to initialize database: {e}")
    
    def extract_usage_from_model_response(self, response: Any, model_name: str = "unknown") -> UsageInfo:
        """Extract usage information from a model response"""
        usage = UsageInfo(model_name=model_name)
        
        if hasattr(response, 'prompt_tokens'):
            usage.prompt_tokens = int(response.prompt_tokens or 0)
        if hasattr(response, 'completion_tokens'):
            usage.completion_tokens = int(response.completion_tokens or 0)
        if hasattr(response, 'cost'):
            usage.cost_usd = float(response.cost or 0.0)
        
        return usage
    
    def extract_usage_from_stage_result(self, result: StageResult) -> StageUsage:
        """Extract usage information from a stage result"""
        stage_usage = StageUsage(stage_id=result.stage_id)
        
        if isinstance(result.model_usage, dict):
            # Handle single model usage
            if "prompt_tokens" in result.model_usage:
                usage = self.extract_usage_from_model_response(result.model_usage)
                stage_usage.usage["main"] = usage
            else:
                # Handle multiple model usage (e.g., APO stage with helper + target)
                for key, usage_data in result.model_usage.items():
                    if isinstance(usage_data, dict):
                        usage = self.extract_usage_from_model_response(usage_data)
                        stage_usage.usage[key] = usage
        
        return stage_usage
    
    def check_budget(self, additional_cost: float = 0.0) -> bool:
        """Check if we're within budget limits"""
        if self.max_budget_usd is None:
            return True
        
        projected_cost = self.total_cost_usd + additional_cost
        if projected_cost > self.max_budget_usd:
            self.logger.warning(
                f"Budget exceeded: {projected_cost:.6f} USD > {self.max_budget_usd:.6f} USD"
            )
            return False
        return True
    
    def log_stage_usage(self, example_id: str, stage_result: StageResult, run_id: str = ""):
        """Log usage for a single stage"""
        stage_usage = self.extract_usage_from_stage_result(stage_result)
        
        # Update totals
        self.total_cost_usd += stage_usage.total_cost_usd
        self.total_tokens += stage_usage.total_tokens
        
        # Store stage usage
        self.stage_usage[stage_result.stage_id] = stage_usage
        
        # Log to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                for usage_key, usage_info in stage_usage.usage.items():
                    conn.execute("""
                        INSERT INTO usage_log 
                        (example_id, stage_id, model_name, prompt_tokens, completion_tokens, 
                         total_tokens, cost_usd, timestamp, run_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        example_id, stage_result.stage_id, usage_info.model_name,
                        usage_info.prompt_tokens, usage_info.completion_tokens,
                        usage_info.total_tokens, usage_info.cost_usd,
                        usage_info.timestamp, run_id
                    ))
        except Exception as e:
            self.logger.warning(f"Failed to log usage to database: {e}")
        
        # Log to console
        self.logger.info(
            f"Stage {stage_result.stage_id}: {stage_usage.total_tokens} tokens, "
            f"${stage_usage.total_cost_usd:.6f} cost"
        )
    
    def log_pipeline_run(self, run_id: str, total_examples: int, config: Dict[str, Any]):
        """Log completion of a pipeline run"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pipeline_runs 
                    (run_id, total_examples, total_cost_usd, total_tokens, 
                     start_time, end_time, config)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, total_examples, self.total_cost_usd, self.total_tokens,
                    time.time(), time.time(), json.dumps(config)
                ))
        except Exception as e:
            self.logger.warning(f"Failed to log pipeline run to database: {e}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of current usage"""
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "stage_breakdown": {
                stage_id: {
                    "total_tokens": usage.total_tokens,
                    "total_cost_usd": usage.total_cost_usd,
                    "prompt_tokens": usage.total_prompt_tokens,
                    "completion_tokens": usage.total_completion_tokens
                }
                for stage_id, usage in self.stage_usage.items()
            }
        }
    
    def reset(self):
        """Reset tracking for a new run"""
        self.total_cost_usd = 0.0
        self.total_tokens = 0
        self.stage_usage.clear()
