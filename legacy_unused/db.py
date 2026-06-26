import sqlite3
import pandas as pd

# Connect to your database file
conn = sqlite3.connect("app.db")

# View all tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("📊 Tables in database:\n", tables)

# View user table
users = pd.read_sql_query("SELECT * FROM users;", conn)
print("\n👤 Users Table:")
print(users)

# View activity logs
activity = pd.read_sql_query("SELECT * FROM activity_log;", conn)
print("\n🧠 Activity Log Table:")
print(activity.head(10))  # show first 10 logs
