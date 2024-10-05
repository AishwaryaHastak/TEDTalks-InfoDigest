# espipeline.py

import os
import pandas as pd 
from elasticsearch import Elasticsearch 
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
from src.constants import model_name,index_name
import tensorflow as tf

# from constants import model_name,index_name

class ElSearchRAGPipeline:
    def __init__(self): 
        self.query = None
        self.response = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # self.tokenizer.pad_token = self.tokenizer.eos_token 
        # self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # self.model = T5ForConditionalGeneration.from_pretrained(model_name) 
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
        self.data_dict = df[:100].to_dict(orient="records")
        
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
        # Considering only the first 100 rows for now
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
        filtered_results = self.es.search(
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
        # Check if any documents exist in the index
        count = self.es.count(index=index_name)
        print('count',count)
        results = self.es.search(
            index=index_name,
            size = num_results,
            query={ 
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": query,
                                "fields": ["transcript^3", "topics","description"],
                                "type": "best_fields",
                            }
                        }, 
                },
            }
        )
        print('\n\n [DEBUG]  Filtered Results: ', filtered_results['hits']['hits'])
        print('\n\n [DEBUG]  Results: ', results['hits']['hits'])
        time_taken = results['took']
        relevance_score = results['hits']['max_score']
        total_hits = results['hits']['total']['value']
        topics = results['hits']['hits'][0]['_source']['topics']

        print('\n\n[DEBUG] Retrieved results:', time_taken, relevance_score, total_hits)
        result_docs = [hit['_source'] for hit in results['hits']['hits']] 

        response = [result['transcript'] for result in result_docs]
        print('\n\n [DEBUG]  response: ', response)
 
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
        # prompt_template = """
        # You are a TED Talks Summary and Q&A Assistant. 
        # Answer the question using the context below, summarizing and rephrasing the information instead of quoting it directly.
        # QUESTION: {question}
        # CONTEXT: 
        # {response}
        # Provide a clear and insightful explanation.
        # """.strip()

        prompt_template = """
        You are a TED Talks Summary and Q&A Assistant. 
        Answer the question about the talks using the speaker's transcript of the talk below, summarizing what the speaker meant to say.
        QUESTION: {question}
        CONTEXT: 
        {response}
        Provide a clear and insightful answer.
        """.strip()

        prompt = prompt_template.format(question=query, response=response)
        print('\n\n Prompt: ', prompt)
        return prompt
    
    def generate_response(self, query, response): #prompt): 
        """
        Generates a response using the LLM based on the given prompt.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response from the LLM.
        """

        print('[DEBUG] Generating LLM response...')
        # inputs = self.tokenizer(
        #     prompt, 
        #     return_tensors="pt", 
        #     max_length=512, 
        #     truncation=True, 
        #     padding='max_length'
        # )

        # # Generate Response
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=1024,
        #     min_length=100,
        #     no_repeat_ngram_size=3,
        #     do_sample=True, 
        #     num_beams=6,        
        #     early_stopping = True # Stop once all beams are finished 
        #     # early_stopping = False # Stop once max_new_tokens is reached
        # )
        
        # llm_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        inputs = self.tokenizer(query, response, return_tensors="tf")
        outputs = self.model(**inputs)

        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        llm_response = self.tokenizer.decode(predict_answer_tokens)

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
        # prompt = self.generate_prompt(query, results)
        llm_response = self.generate_response(query, results)
        # llm_response = self.generate_response(prompt)
        return llm_response, time_taken, total_hits, relevance_score, topic