import sqlite3
from pathlib import Path

curr_path = Path.cwd()

class SQL_Manager:
    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)

    def create_general_model(self):
        query = """CREATE TABLE General (data_id TEXT, source TEXT,title TEXT, topic TEXT, anchor TEXT, positive TEXT, negative TEXT);"""
        self.con.execute(query)
        self.con.commit()

    def insert_general_model(self, data_id: str, source: str, title: str, topic: str, anchor: str, positive: str, negative: str):
        query = """INSERT INTO General VALUES (?, ?, ?, ?, ?, ?, ?)"""
        self.con.execute(query, (data_id, source, title, topic, anchor, positive, negative))
        #Not commit here, I'll commit while process data

if __name__ == "__main__":
    sql = SQL_Manager(curr_path / "database" / "data.db")
    sql.create_general_model()