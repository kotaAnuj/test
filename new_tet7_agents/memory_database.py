import sqlite3

class MemoryDatabase:
    def __init__(self, db_name='agent_memory.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS memory (
            agent_name TEXT PRIMARY KEY,
            context TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def update_context(self, agent_name, context):
        query = """
        INSERT INTO memory (agent_name, context)
        VALUES (?, ?)
        ON CONFLICT(agent_name) DO UPDATE SET context=excluded.context
        """
        self.conn.execute(query, (agent_name, context))
        self.conn.commit()

    def get_context(self, agent_name):
        query = "SELECT context FROM memory WHERE agent_name = ?"
        cursor = self.conn.execute(query, (agent_name,))
        result = cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()
