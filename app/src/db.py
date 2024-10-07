# db.py
# This script initializes the PostgreSQL database and creates necessary tables.

import psycopg2
import os
from src.constants import DATABASE_URI, DB_PASSWORD, DB_NAME, DB_USER, DB_HOST

def init_db():
    """Initialize the PostgreSQL database with necessary tables."""

    print('[DEBUG] Initializing database...')
    
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor:
            # cursor.execute("DROP TABLE IF EXISTS feedback")
            # cursor.execute("DROP TABLE IF EXISTS conversations")
            # Create the conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    time_taken INT,  
                    total_hits INT,
                    relevance_score FLOAT,   
                    topic TEXT,
                    selected_talk TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')        
            
            # Create the feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            ''')
            
            conn.commit()

def save_conversation(conversation_id, question, answer, time_taken, total_hits, relevance_score, topic, selected_talk):
    """Save a conversation to the database."""
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor: 
            cursor.execute('''
                INSERT INTO conversations (id, question, answer, time_taken, total_hits, relevance_score, topic, selected_talk) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (conversation_id, question, answer, time_taken, total_hits, relevance_score, topic, selected_talk))
            conn.commit()
            print('[DEBUG] Conversation saved to database.')

def save_feedback(conversation_id, feedback):
    """Save feedback to the database."""
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO feedback (conversation_id, feedback) VALUES (%s, %s)
            ''', (conversation_id, feedback))
            conn.commit()
            print('[DEBUG] Feedback saved to database.')

def get_recent_conversations(limit=5):
    """Retrieve recent conversations from the database."""
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM conversations ORDER BY timestamp DESC LIMIT %s', (limit,))
            rows = cursor.fetchall()
    
    # Convert rows to dictionaries for easier access
    conversations = []
    for row in rows:
        conversation = {
            'id': row[0],
            'question': row[1],
            'answer': row[2],
            'time_taken': row[3],
            'total_hits': row[4],
            'relevance_score': row[5],
            'topic': row[6],
            'timestamp': row[7]
        }
        conversations.append(conversation)
    
    return conversations

def get_feedback_stats():
    """Retrieve feedback statistics from the database."""
    with psycopg2.connect(DATABASE_URI) as conn:
        with conn.cursor() as cursor:
            # Count positive feedback
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = 1')
            thumbs_up = cursor.fetchone()[0]
            
            # Count negative feedback
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = -1')
            thumbs_down = cursor.fetchone()[0]
    
    return {
        'thumbs_up': thumbs_up,
        'thumbs_down': thumbs_down
    }

if __name__ == "__main__":
    init_db()
    print("Database initialized and tables created.")
