import streamlit as st
import pandas as pd
import time
import uuid
from src.espipeline import ElSearchRAGPipeline
from src.db import (
    init_db,
    save_conversation,
    save_feedback,
    get_recent_conversations,
    get_feedback_stats,
)

def print_log(message):
    print(message, flush=True)

def load_talks_data(file_path):
    """Load the talks data from a CSV file and return a mapping of topics to titles."""
    df = pd.read_csv(file_path)
    
    # Expand the 'topics' column into individual topics
    df['topics'] = df['topics'].apply(lambda x: eval(x))  # Assuming topics are stored as a list string
    all_topics = set()
    for topics in df['topics']:
        all_topics.update(topics)
    
    return df, sorted(all_topics)

def main():
    st.set_page_config(page_title="Ted Talks Assistant", layout="wide")
    st.title("🌟 Ted Talks Assistant 🌟")

    # Initialize the database
    init_db()

    # Load the talks data
    talks_data_path = 'data/topk_cleaned_data.csv'
    df, unique_topics = load_talks_data(talks_data_path)

    # Check if the text index has been created
    if "text_index_created" not in st.session_state:
        st.session_state.text_index_created = False
    if "count" not in st.session_state:
        st.session_state.count = 0
    if "has_asked_question" not in st.session_state:
        st.session_state.has_asked_question = False  # Track if user has asked a question

    # Sidebar for multiple topic selection
    st.sidebar.header("Select Topics")
    selected_topics = st.sidebar.multiselect("Choose topics:", unique_topics)
    
    # Display corresponding talk titles based on selected topics
    st.sidebar.subheader("Available Talks")
    if selected_topics:
        # Filter talks that have any of the selected topics
        filtered_talks = df[df['topics'].apply(lambda topics: any(topic in topics for topic in selected_topics))]
        talk_titles = filtered_talks['title'].tolist()
        
        selected_talk = st.sidebar.selectbox("Choose a talk:", talk_titles)
        
        # Show the selected title and description
        if selected_talk:
            talk_description = filtered_talks[filtered_talks['title'] == selected_talk]['description'].values[0]
            st.sidebar.subheader("Talk Description:")
            st.sidebar.write(talk_description)
    else:
        st.sidebar.write("Please select at least one topic to see available talks.")

    # Initialize the text search pipeline
    pipeline = ElSearchRAGPipeline()
    if not st.session_state.text_index_created:
        with st.spinner("Reading and indexing data..."):
            pipeline.read_data()
            pipeline.create_index()
            st.session_state.text_index_created = True
        st.success("Text index created!")

    # User input
    user_input = st.text_input("💬 Enter your question related to the talk:", "")

    if st.button("Ask"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.has_asked_question = True  # Mark that the user has asked a question
        
        with st.spinner("Processing..."):
            answer_data, time_taken, total_hits, relevance_score, topic = pipeline.get_response(user_input, selected_talk)
            st.success("Completed!")
            st.write(answer_data)

            # Save conversation to database
            save_conversation(
                st.session_state.conversation_id, 
                user_input, 
                answer_data, 
                time_taken, 
                total_hits, 
                relevance_score,
                topic,
                "Text"  # Only using text search
            )

    # Feedback section: only show if a question has been asked
    if st.session_state.has_asked_question:
        st.subheader("Feedback")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('👍'):
                st.session_state.count += 1
                save_feedback(st.session_state.conversation_id, 1)
                st.success('Thank you for the positive feedback!')
        with col2:
            if st.button('👎'):
                st.session_state.count -= 1
                save_feedback(st.session_state.conversation_id, -1)
                st.error('Sorry to hear that. We appreciate your feedback!')

    # Display recent conversations
    st.subheader("Recent Conversations")
    recent_conversations = get_recent_conversations(limit=3)
    for conv in recent_conversations:
        st.write(f"**Q:** {conv['question']}")
        st.write(f"**A:** {conv['answer']}")
        st.write("---")

    # Display feedback stats
    feedback_stats = get_feedback_stats()
    st.subheader("Feedback Statistics")
    st.write(f"👍 Thumbs up: {feedback_stats['thumbs_up']}")
    st.write(f"👎 Thumbs down: {feedback_stats['thumbs_down']}")

if __name__ == "__main__":
    print_log("Course Assistant application started")
    main()
