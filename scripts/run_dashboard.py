import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.dashboard.dashboard import run_dashboard

if __name__ == "__main__":
    db_path = os.getenv("TRACKING_DB_PATH", "usage_tracking.db")
    
    print(f"Starting pipeline dashboard with database: {db_path}")
    print("Open your browser to http://localhost:8501")
    
    run_dashboard(db_path)
