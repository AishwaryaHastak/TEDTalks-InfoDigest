# vectorpipeline.py

import os
import pandas as pd
import requests
from elasticsearch import Elasticsearch, helpers
from langchain_huggingface import HuggingFaceEndpoint
from tqdm import tqdm 
from src.constants import model_name,index_name, embedding_size, HF_API_TOKEN
# from constants import model_name,index_name, embedding_model, embedding_size, HF_API_TOKEN 
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

class VecSearchRAGPipeline:
    def __init__(self): 
        self.query = None
        self.response = None 
        self.es = Elasticsearch("http://elasticsearch:9200")
        # self.es = Elasticsearch("http://localhost:9200") 
        self.emb_model = HuggingFaceInferenceAPIEmbeddings(
                    api_key=HF_API_TOKEN, 
                    model_name="sentence-transformers/all-MiniLM-l6-v2"
                )
        self.data_dict = None

    def get_embeddings_batch(self, texts):
        """
        Generates embedding vectors for a batch of texts using the Hugging Face endpoint.
        """
        data = self.emb_model.embed_documents(texts)[:embedding_size]
        return data 

    def read_data(self, batch_size=64):
        """
        Reads data from csv file and converts it into list of dictionaries.
        Additionally, generates vector embeddings for the question and answer
        using the SentenceTransformer model and adds them to the dictionary.
        """
        
        print('\n\n [DEBUG] Reading data...')
        # Read data into dataframe 
        data_file_path = os.path.join('data', 'chunked_data.csv') 
        df = pd.read_csv(data_file_path).dropna()

        # Convert dataframe to list of dictionaries
        data_dict = df.to_dict(orient="records")

        print('\n\n [DEBUG] Generating vector embeddings...') 
        # Add answer and question vector embeddings
        vector_data_dict = []  
        for i in tqdm(range(0, len(data_dict), batch_size)):
            batch = data_dict[i:i + batch_size]
            texts = [f"{data['title']} {data['transcript']}" for data in batch]
            embeddings = self.get_embeddings_batch(texts)
            
            for j, data in enumerate(batch):
                data['title_transcript_vector'] = embeddings[j][:embedding_size]
                vector_data_dict.append(data) 
        self.data_dict = vector_data_dict
    
    def create_index(self):
        """
        Creates an Elasticsearch index with the specified settings and mappings,
        and adds data from the data_dict to the index.

        :return: None
        """
        print('\n\n[[DEBUG] Creating Index...')
        index_settings={
            "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                "title": {"type": "text"},
                "transcript": {"type": "text"}, 
                "topics": {"type": "text"}, 
                "speaker": {"type": "text"}, 
                "about_speakers": {"type": "text"}, 
                "description": {"type": "text"}, 
                "title_transcript_vector": {"type": "dense_vector", "dims": embedding_size, "index": True, "similarity": "cosine"},
                }
            }
        }
        
        # Create Index and delete if it already exists
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
        self.es.indices.create(index=index_name, body = index_settings)

        # Add Data to Index using index()
        print('\n\n[[DEBUG] Adding data to index...') 
        actions = [
            {
                "_index": index_name,
                "_id": i,
                "_source": row
            }
            for i, row in enumerate(self.data_dict)
        ]
        helpers.bulk(self.es, actions)

    def search(self, query, selected_talk, num_results):
        """
        Retrieves results from the index based on the query.

        Args: 
            query (str): The search query string.
            num_results (int): The number of top results to return.

        Returns:
            list of str: List of results matching the search criteria, ranked by relevance.
        """

        # Retrieve Search Results
        print('\n\n[[DEBUG] Retrieving Search Results...') 
        query_vector = self.emb_model.embed_query(selected_talk+query)[:embedding_size]
        
        knn_query = {
            "field": "title_transcript_vector",
            "query_vector": query_vector,
            "k": num_results,
            "num_candidates": 50
        }
        print('\n\n[[DEBUG] knn_query: ', knn_query)

        results = self.es.search(index=index_name, knn=knn_query, size = num_results)

        time_taken = results['took']
        relevance_score = results['hits']['max_score']
        total_hits = results['hits']['total']['value']
        topics = results['hits']['hits'][0]['_source']['topics'] 
        result_docs = [hit['_source'] for hit in results['hits']['hits']] 

        response = [result['transcript'] for result in result_docs]
 
        return response, time_taken, relevance_score, total_hits, topics
    
    def generate_prompt(self, query, response):
        """
        Generates a prompt for the LLM based on the query and response.

        Args:
            query (str): The search query string.
            response (str): The response from the retrieval model.

        Returns:
            str: The prompt to be sent to the LLM.
        """

        prompt_template = """
        You are a TED Talks Q&A Assistant. 
        Answer the question about the talks using the speaker's transcript of the talk below, summarizing what the speaker meant to say.
        QUESTION: {question}
        CONTEXT: 
        {response}
        Provide a clear and insightful answer.
        """.strip()

        prompt = prompt_template.format(question=query, response=response)
        return prompt
    
    def generate_response(self, prompt): 
        """
        Generates a response using the LLM based on the given prompt.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response from the LLM.
        """

        print('[DEBUG] Generating LLM response...') 
        
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=HF_API_TOKEN,
        ) 
        llm_response = llm.invoke(prompt) 
        return llm_response
    
    def get_response(self,query, selected_talk, num_results=3):
        """
        Retrieves and generates a response for a given query.

        Args:
            query (str): The search query string.
            num_results (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            str: The generated response from the LLM.
        """ 
        results, time_taken, total_hits, relevance_score, topic  = self.search(query, selected_talk, num_results)
        prompt = self.generate_prompt(query, results)
        llm_response = self.generate_response(prompt)
        return llm_response, time_taken, total_hits, relevance_score , topic