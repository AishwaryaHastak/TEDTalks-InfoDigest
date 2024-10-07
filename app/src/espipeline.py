# espipeline.py

import os
import pandas as pd 
from elasticsearch import Elasticsearch 
from tqdm import tqdm  
from langchain_huggingface import HuggingFaceEndpoint
from src.constants import model_name,index_name, HF_API_TOKEN
# import tensorflow as tf

# from constants import model_name,index_name, HF_API_TOKEN

class ElSearchRAGPipeline:
    def __init__(self): 
        self.query = None
        self.response = None 
        # self.es = Elasticsearch("http://elasticsearch:9200") 
        self.es = Elasticsearch("http://localhost:9200") 
        self.data_dict = None

    def read_data(self):
        """
        Reads data from csv file and converts it into list of dictionaries
        
        :return: None
        """
        print('[DEBUG] Reading data...')
        # Read data into dataframe 
        data_file_path = os.path.join('data', 'chunked_data.csv')
        # data_file_path = os.path.join('..','data', 'chunked_data.csv')
        df = pd.read_csv(data_file_path).dropna()

        # Convert dataframe to list of dictionaries
        self.data_dict = df.to_dict(orient="records")
        
    def create_index(self):
        """
        Creates an Elasticsearch index with the specified settings and mappings,
        and adds data from the data_dict to the index.

        :return: None
        """
        print('\n\n[[DEBUG] Creating Index...')
        mappings = {
                "properties": {
                    "transcript": {"type": "text"},
                    "topics": {"type": "text"},
                    "speaker": {"type": "text"},
                    "about_speakers": {"type": "text"},
                    "description": {"type": "text"},
                    "title": {"type": "keyword"},
                    "id": {"type": "keyword"},
            }
        }
        
        # Create Index and delete if it already exists
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
        self.es.indices.create(index=index_name, mappings=mappings)

        # Add Data to Index using index()
        print('\n\n[[DEBUG] Adding data to index...') 
        for i in tqdm(range(len(self.data_dict))):
            row = self.data_dict[i]
            # self.es.index(index=index_name, document=row) 
            self.es.index(index=index_name, id=row['id'], document=row) 

    def search(self, query, title, num_results=3):
        """
        Retrieves results from the index based on the query.

        Args:
            data_dict (list of dict): List of dictionaries containing the data to be indexed.
            query (str): The search query string.
            num_results (int): The number of top results to return.

        Returns:
            list of str: List of results matching the search criteria, ranked by relevance.
        """
        # Retrieve Search Results
        print('\n\n[[DEBUG] Retrieving Search Results...') 
        print('\n title:', title.strip())
        # title = 'Why we love, why we cheat'
        # filtered_results = self.es.search(
        results = self.es.search(
            index=index_name,
            size = num_results,
            query={ 
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": query,
                                "fields": ["transcript^3", "topics" ,"speaker","description", "about_speaker"],
                                "type": "best_fields",
                            }
                        }, 
                        "filter": {
                            "term": {
                                "title": title
                            }
                        }
                },
            }
        ) 
        
        time_taken = results['took']
        relevance_score = results['hits']['max_score']
        total_hits = results['hits']['total']['value']
        topics = results['hits']['hits'][0]['_source']['topics']

        print('\n\n[DEBUG] Retrieved results:', time_taken, relevance_score, total_hits)
        result_docs = [hit['_source'] for hit in results['hits']['hits']] 

        response = [result['transcript'] for result in result_docs]
 
        return response, time_taken, total_hits, relevance_score, topics
    
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
        print('\n\n Prompt: ', prompt)
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
        # print(llm.invoke(prompt))
        llm_response = llm.invoke(prompt)
        print(llm_response)
        return llm_response
    
    def get_response(self,query,selected_talk, num_results=3):
        """
        Retrieves and generates a response for a given query.

        Args:
            query (str): The search query string.
            num_results (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            str: The generated response from the LLM.
        """ 
        results, time_taken, total_hits, relevance_score, topic = self.search(query,selected_talk, num_results)
        prompt = self.generate_prompt(query, results)
        # llm_response = self.generate_response(query, results)
        llm_response = self.generate_response(prompt)
        return llm_response, time_taken, total_hits, relevance_score, topic