import sqlite3
import os
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()
DB_PATH = os.getenv("PHARMACY_DB", "pharmacy.db")

def list_tables() -> List[str]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [t[0] for t in cursor.fetchall()]

def describe_table(table_name: str) -> List[Tuple[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        return [(col[1], col[2]) for col in cursor.fetchall()]

def execute_query(sql: str) -> List[Tuple]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
        return result
