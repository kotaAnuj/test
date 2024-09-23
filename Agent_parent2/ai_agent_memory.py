# chatbot_memory.py
import sqlite3
import json
import datetime

class ChatbotMemory:
    def __init__(self, db_path="chatbot_memory.db"):
        self.db_path = db_path
        self._setup_database()

    def _setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS conversation_history (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                role TEXT,
                                content TEXT,
                                timestamp TEXT,
                                reasoning TEXT
                              )''')
            conn.commit()

    def save_message(self, role, content, reasoning=None):
        timestamp = datetime.datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO conversation_history (role, content, timestamp, reasoning)
                              VALUES (?, ?, ?, ?)''',
                           (role, content, timestamp, json.dumps(reasoning) if reasoning else None))
            conn.commit()

    def load_conversation_history(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT role, content, reasoning FROM conversation_history ORDER BY id ASC''')
            return cursor.fetchall()
