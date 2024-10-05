# vectorpipeline.py

import os
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.constants import model_name,index_name, embedding_model, embedding_size
# from constants import model_name,index_name, embedding_model, embedding_size
from sentence_transformers import SentenceTransformer

class VecSearchRAGPipeline:
    def __init__(self): 
        self.query = None
        self.response = None
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        # self.es = Elasticsearch("http://elasticsearch:9200")
        self.es = Elasticsearch("http://localhost:9200")
        self.emb_model = SentenceTransformer(embedding_model,truncate_dim=embedding_size) 
        self.data_dict = None

    def read_data(self):
        """
        Reads data from csv file and converts it into list of dictionaries.
        Additionally, generates vector embeddings for the question and answer
        using the SentenceTransformer model and adds them to the dictionary.
        """
        
        print('[DEBUG] Reading data...')
        # Read data into dataframe 
        # data_file_path = os.path.join('data', 'data.csv')
        data_file_path = os.path.join('..','data', 'chunked_data.csv')
        df = pd.read_csv(data_file_path).dropna()

        # Convert dataframe to list of dictionaries
        data_dict = df.to_dict(orient="records")

        print('[DEBUG] Generating vector embeddings...')
        # Add answer and question vector embeddings
        vector_data_dict = []
        for data in tqdm(data_dict):
            title_transcript = data['title'] + ' ' + data['transcript']
            data['title_transcript_vector'] = self.emb_model.encode(title_transcript)
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
        for i in tqdm(range(len(self.data_dict))):
            row = self.data_dict[i]
            self.es.index(index=index_name, id=i, document=row)

        # helpers.bulk(es, data_dict)

    def search(self, query, num_results):
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
        query_vector = self.emb_model.encode(query)
        knn_query = {
            "field": "question_answer_vector",
            "query_vector": query_vector,
            "k": num_results,
            "num_candidates": 5
        }

        results = self.es.search(index=index_name, knn=knn_query, size = num_results)

        time_taken = results['took']
        relevance_score = results['hits']['max_score']
        total_hits = results['hits']['total']['value']
        topics = results['hits']['hits'][0]['_source']['topics']
        title = results['hits']['hits'][0]['_source']['title']
        result_docs = [hit['_source'] for hit in results['hits']['hits']] 

        response = [result['answer'] for result in result_docs]
 
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
        You are a TED Talks Summary and Q&A Assistant. 
        Answer the question using the context below, summarizing and rephrasing the information instead of quoting it directly.
        QUESTION: {question}
        CONTEXT: 
        {response}
        Provide a clear and insightful explanation.
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
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )

        # Generate Response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            min_length=100,
            no_repeat_ngram_size=3,
            do_sample=True, 
            num_beams=4,        
            early_stopping = True # Stop once all beams are finished  
        )
        
        llm_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return llm_response
    
    def get_response(self,query, num_results=3):
        """
        Retrieves and generates a response for a given query.

        Args:
            query (str): The search query string.
            num_results (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            str: The generated response from the LLM.
        """ 
        results, time_taken, total_hits, relevance_score, topic  = self.search(query, num_results)
        prompt = self.generate_prompt(query, results)
        llm_response = self.generate_response(prompt)
        return llm_response, time_taken, total_hits, relevance_score , topic