# Day 11: Database Integration with Flask (SQLite)
 
import sqlite3
 
# Connect to database (creates file if not exists)
conn = sqlite3.connect('job_predictions.db')
 
# Create table for predictions
conn.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_description TEXT,
    prediction TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
''')
conn.close()

print("✅ Database and table created successfully!")

 