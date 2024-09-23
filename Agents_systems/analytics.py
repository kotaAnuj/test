# analytics.py

import sqlite3

def log_interaction(user_id, query, response, timestamp):
    conn = sqlite3.connect('agent_memory.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO interactions_log (user_id, query, response, timestamp) VALUES (?, ?, ?, ?)",
                   (user_id, query, response, timestamp))
    conn.commit()
    conn.close()

def retrieve_interaction_logs(user_id):
    conn = sqlite3.connect('agent_memory.db')
    cursor = conn.cursor()
    cursor.execute("SELECT query, response, timestamp FROM interactions_log WHERE user_id = ?", (user_id,))
    logs = cursor.fetchall()
    conn.close()
    return logs
