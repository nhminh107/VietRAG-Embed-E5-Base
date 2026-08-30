import sqlite3
from pathlib import Path
from database.models import GeneralModel
curr_path = Path.cwd()

class SQL_Manager:
    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)

    def create_general_model(self):
        query = """CREATE TABLE General (data_id TEXT PRIMARY KEY , source TEXT,title TEXT, topic TEXT, anchor TEXT, positive TEXT, negative TEXT);"""
        self.con.execute(query)
        self.con.commit()

    def insert_general_model(self, data: GeneralModel):
        query = """INSERT INTO General VALUES (?, ?, ?, ?, ?, ?, ?)"""
        self.con.execute(query, (data.data_id, data.source, data.title, data.topic, data.anchor, data.positive, data.hard_negative))
        #Not commit here, I'll commit while process data

if __name__ == "__main__":
    sql = SQL_Manager(curr_path / "database" / "data.db")
    sql.create_general_model()