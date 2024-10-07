import streamlit as st
import pandas as pd
import uuid
from src.espipeline import ElSearchRAGPipeline
from src.vectorpipeline import VecSearchRAGPipeline
from src.summary import summarize
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
    df['topics'] = df['topics'].apply(lambda x: eval(x))  # Assuming topics are stored as a list string
    all_topics = set()
    for topics in df['topics']:
        all_topics.update(topics)
    return df, sorted(all_topics)

def summary_page(selected_talk):

    st.subheader("Talk Summary")
    talk_summary = summarize(selected_talk)
    st.write(talk_summary)  

def qa_page(selected_talk, pipeline):

    st.subheader("Ask a Question")
    user_input = st.text_input("💬 Enter your question related to the talk:", "")

    if st.button("Ask"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.has_asked_question = True
        
        # pipeline =  ElSearchRAGPipeline() #VecSearchRAGPipeline() # 
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
    if "indexing_in_progress" not in st.session_state:
        st.session_state.indexing_in_progress = False  # Flag for indexing status
    if "has_asked_question" not in st.session_state:
        st.session_state.has_asked_question = False  # Track if user has asked a question

    # Sidebar for multiple topic selection
    st.sidebar.header("Select Topics")
    selected_topics = st.sidebar.multiselect("Choose topics:", unique_topics)
    
    # Display corresponding talk titles based on selected topics
    st.sidebar.subheader("Available Talks")
    selected_talk = None
    talk_description = ""
    
    if selected_topics:
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

    # Select the page with nicer formatting
    st.sidebar.header("Page Navigation")
    page = st.sidebar.radio('',("Summary 📄", "Q&A ❓"))

    # Initialize the text search pipeline if not already done
    pipeline =VecSearchRAGPipeline() #  ElSearchRAGPipeline() #
    if not st.session_state.text_index_created and not st.session_state.indexing_in_progress:
        st.session_state.indexing_in_progress = True
        with st.spinner("Reading and indexing data..."):
            pipeline.read_data()
            pipeline.create_index()
            st.session_state.text_index_created = True
            st.session_state.indexing_in_progress = False
        st.success("Text index created!")

    # Render the selected page
    if page == "Summary 📄":
        if selected_talk:
            summary_page(selected_talk)
        else:
            st.warning("Please select a talk to view the summary.")

    elif page == "Q&A ❓":
        if selected_talk:
            qa_page(selected_talk, pipeline=pipeline)
            # Feedback section: only show if a question has been asked
            if st.session_state.has_asked_question:
                st.subheader("Feedback")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button('👍'):
                        save_feedback(st.session_state.conversation_id, 1)
                        st.success('Thank you for the positive feedback!')
                with col2:
                    if st.button('👎'):
                        save_feedback(st.session_state.conversation_id, -1)
                        st.error('Sorry to hear that. We appreciate your feedback!')
        else:
            st.warning("Please select a talk to ask questions.")

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
