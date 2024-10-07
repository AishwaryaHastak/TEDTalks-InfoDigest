import os
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint
from src.constants import HF_API_TOKEN

def summarize(selected_talk):
    # Read data from csv file
    data_file_path = os.path.join('data', 'topk_cleaned_data.csv')
    df = pd.read_csv(data_file_path)

    # Retrieve the transcript for the selected talk
    transcript_series = df['transcript'][df['title'] == selected_talk]
    speaker = df['speaker'][df['title'] == selected_talk]
    if transcript_series.empty:
        raise ValueError(f"No transcript found for talk: {selected_talk}")

    transcript = transcript_series.values[0]

    # Prepare the prompt
    prompt = f"""
    You are a TED Talks Summary Assistant. 
    Break down the talk into logical sections and summarize the concepts or ideas discussed.
    The talk is about:
    TALK TITLE: {selected_talk} by {speaker}
    TRANSCRIPT: 
    {transcript}
    """.strip()
    
    # Initialize LLM 
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=1024, 
        temperature=0.5,
        huggingfacehub_api_token=HF_API_TOKEN,
    )

    # Call the model and handle response
    try:
        llm_response = llm.invoke(prompt) 
        return llm_response
    except Exception as e:
        print(f"Error during model invocation: {e}")
        raise
