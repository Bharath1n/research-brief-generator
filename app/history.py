import sqlite3

conn = sqlite3.connect('history.db')
conn.execute('CREATE TABLE IF NOT EXISTS briefs (user_id TEXT, brief TEXT)')

def save_brief(user_id: str, brief: str):
    conn.execute('INSERT INTO briefs VALUES (?, ?)', (user_id, brief))
    conn.commit()

def load_history(user_id: str) -> str:
    cursor = conn.execute('SELECT brief FROM briefs WHERE user_id=?', (user_id,))
    return "\n".join(row[0] for row in cursor.fetchall())

# Optional: Close conn on shutdown
import atexit
atexit.register(conn.close)