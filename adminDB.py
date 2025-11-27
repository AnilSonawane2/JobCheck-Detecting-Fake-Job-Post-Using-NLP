import sqlite3
conn = sqlite3.connect('job_predictions.db')
conn.execute("""
CREATE TABLE IF NOT EXISTS admin (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
);
""")
 
# Insert default admin
conn.execute("INSERT INTO admin (username, password) VALUES ('admin', 'admin123')")
conn.commit()
conn.close()
 
print("Admin user created: username='admin', password='admin123'")

 