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
    df = pd.read_csv(file_path)
    df['topics'] = df['topics'].apply(lambda x: eval(x))
    all_topics = set()
    for topics in df['topics']:
        all_topics.update(topics)
    return df, sorted(all_topics)

def summary_page(selected_talk):
    talk_summary = summarize(selected_talk)
    return talk_summary

def qa_page(selected_talk, pipeline):
    st.header("Ask a Question")
    user_input = st.text_input("💬 Enter your question related to the talk:", "")

    if st.button("Ask", key="ask_button"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.has_asked_question = True
         
        with st.spinner("Processing..."):
            answer_data, time_taken, total_hits, relevance_score, topic = pipeline.get_response(user_input, selected_talk)
            st.success("Completed!")
            st.markdown(f"**Answer:** {answer_data}")

            save_conversation(
                st.session_state.conversation_id, 
                user_input, 
                answer_data, 
                time_taken, 
                total_hits, 
                relevance_score,
                topic,
                selected_talk,
            )

def main():
    st.set_page_config(page_title="Ted Talks Assistant", layout="wide")
    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #FF4B4B;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Ted Talks Assistant</p>', unsafe_allow_html=True)

    init_db()

    talks_data_path = 'data/topk_cleaned_data.csv'
    df, unique_topics = load_talks_data(talks_data_path)

    if "text_index_created" not in st.session_state:
        st.session_state.text_index_created = False
    if "indexing_in_progress" not in st.session_state:
        st.session_state.indexing_in_progress = False
    if "has_asked_question" not in st.session_state:
        st.session_state.has_asked_question = False

    with st.sidebar:
        st.header("🎯 Select Topics")
        selected_topics = st.multiselect("Choose topics:", unique_topics)
        
        st.header("📚 Available Talks")
        selected_talk = None
        talk_description = ""
        
        if selected_topics:
            filtered_talks = df[df['topics'].apply(lambda topics: any(topic in topics for topic in selected_topics))]
            talk_titles = filtered_talks['title'].tolist()
            
            selected_talk = st.selectbox("Choose a talk:", talk_titles)
            
            if selected_talk: 
                speaker = filtered_talks[filtered_talks['title'] == selected_talk]['speaker'].values[0]
                about_speaker = filtered_talks[filtered_talks['title'] == selected_talk]['about_speakers'].values[0]
                st.subheader(f"{speaker} - {selected_talk}")
                talk_description = filtered_talks[filtered_talks['title'] == selected_talk]['description'].values[0]
                st.markdown("**Talk Description:**")
                st.info(talk_description)
        else:
            st.warning("Please select at least one topic to see available talks.")

        st.header("📑 Page Navigation")
        page = st.radio('',("Summary 📄", "Q&A ❓"))

    pipeline = VecSearchRAGPipeline()
    if not st.session_state.text_index_created and not st.session_state.indexing_in_progress:
        st.session_state.indexing_in_progress = True
        with st.spinner("Reading and indexing data..."):
            pipeline.read_data()
            pipeline.create_index()
            st.session_state.text_index_created = True
            st.session_state.indexing_in_progress = False
        st.success("Text index created!")

    if page == "Summary 📄":
        if selected_talk:
            st.markdown("---")  
            st.markdown(f'### "{selected_talk}" by {speaker}')
            st.markdown("<br>", unsafe_allow_html=True) 
            st.subheader("About the Speaker")  
            st.info(about_speaker)

            st.markdown("---")  
            st.subheader("Summary")
            talk_summary = summary_page(selected_talk)
            st.write(talk_summary)
        else:
            st.warning("Please select a talk to view the summary.")

    elif page == "Q&A ❓":
        if selected_talk:
            qa_page(selected_talk, pipeline=pipeline)
            if st.session_state.has_asked_question:
                st.subheader("Feedback")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button('👍 Helpful'):
                        save_feedback(st.session_state.conversation_id, 1)
                        st.success('Thank you for the positive feedback!')
                with col2:
                    if st.button('👎 Not Helpful'):
                        save_feedback(st.session_state.conversation_id, -1)
                        st.error('Sorry to hear that. We appreciate your feedback!')
        else:
            st.warning("Please select a talk to ask questions.")

        st.markdown("---")
        st.subheader("Recent Conversations")
        recent_conversations = get_recent_conversations(limit=3)
        for conv in recent_conversations:
            st.markdown(f"**Q:** {conv['question']}")
            st.markdown(f"**A:** {conv['answer']}")
            st.markdown("---")

        feedback_stats = get_feedback_stats()
        st.subheader("Feedback Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👍 Helpful", feedback_stats['thumbs_up'])
        with col2:
            st.metric("👎 Not Helpful", feedback_stats['thumbs_down'])

if __name__ == "__main__":
    print_log("Ted Talks Assistant application started")
    main()