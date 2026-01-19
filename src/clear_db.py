"""Clear corrupted database"""
import shutil
from pathlib import Path

db_path = Path("data/vectordb")
if db_path.exists():
    shutil.rmtree(db_path)
    print("✅ Cleared corrupted database")