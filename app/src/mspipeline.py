# mspipeline.py

import os
import pandas as pd
# from src import minisearch
import minisearch
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from src.constants import keyword_fields, text_fields, model_name
from constants import keyword_fields, text_fields, model_name

class MiniSearchRAGPipeline:
    def __init__(self): 
        self.query = None
        self.response = None
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name) 

    def read_data(self):
        """
        Reads data from csv file and converts it into list of dictionaries
        """
        print('[DEBUG] Reading data...')
        # Read data into dataframe 
        data_file_path = os.path.join('..','data', 'chunked_data.csv')
        df = pd.read_csv(data_file_path).dropna()

        # Convert dataframe to list of dictionaries
        data_dict = df.to_dict(orient="records")
        return data_dict
    
    def search(self, data_dict, query, num_results):
        """
        Retrieves results from the index based on the query.

        Args:
            data_dict (list of dict): List of dictionaries containing the data to be indexed.
            query (str): The search query string.
            num_results (int): The number of top results to return.

        Returns:
            list of str: List of results matching the search criteria, ranked by relevance.
        """
        
        print('[DEBUG] Creating Index...')
        # Create Index
        ms = minisearch.Index(
            text_fields=text_fields,
            keyword_fields=keyword_fields,
        )

        # Retrieve Results
        print('[DEBUG] Retrieving Search Results...')
        ms.fit(data_dict)
        
        results = ms.search( query = query,
                            num_results = num_results)
        response = [result['transcript'] for result in results]
        print('\n\n[DEBUG] Retrieved results:', response)
        return response
    
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
            You're a TED Talks Assistant.
            Provide concise and complete answers to the questions based on the transcipt portions provided as context given below.
            QUESTION: {question}

            CONTEXT: 
            {response}
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
            # early_stopping = False # Stop once max_new_tokens is reached
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
        data_dict = self.read_data()
        results = self.search(data_dict, query, num_results)
        prompt = self.generate_prompt(query, results)
        llm_response = self.generate_response(prompt)
        return llm_response