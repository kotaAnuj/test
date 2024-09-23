import sqlite3

class RelationalDBAgent:
    def __init__(self, db_path="database.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS data_table (id INTEGER PRIMARY KEY, info TEXT)"
            )

    def store_data(self, data):
        with self.conn:
            self.conn.execute("INSERT INTO data_table (info) VALUES (?)", (data,))
