# Ted Talks Explorer Application

Ted Talks are a source of profound knowledge and opinions delivered in an engaging manner by experts. This project provides answers and summaries of the most viewed Ted Talks since 2006, offering an easy-to-use interface for insights and quick summaries.🚀

The application utilizes tools like ElasticSearch, Streamlit, PostgreSQL, Grafana, and Docker.
 
![TED_assistant_Diagram](https://github.com/user-attachments/assets/c58f2016-2b95-4664-bd71-3208beffd86f)



### 🔍📝👉 To learn more about RAGs, check out this [article](https://medium.com/@aishwaryahastak/understanding-the-roots-of-rags-7b77d26c3dca).

## 📈Project Overview

This application utilizes the [TED Talks dataset](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset) provided by Miguel Corral Jr. on Kaggle. These datasets contain over 4,000 TED talks including transcripts in many languages. For this application, we are only focusing on the 100 most viewed TED talks that cover many topics such as global issues, culture, business, and technology.


## RAG Flow

The Retrieval-Augmented Generation (RAG) flow combines a knowledge base with a language model to deliver accurate responses:

- **Knowledge Base:** Contains a Ted Talks transcript dataset stored in `data.csv`.
- **Language Model:** Uses **Mistral-7B-Instruct**, an open-source model by Mistral AI on Hugging Face, for augmented response generation.

## 📊 Retrieval Evaluation

The performance of retrieval methods was assessed using `ground-truth.csv`. The following methods were evaluated:

- **ElasticSearch:** 
  - **Hit Rate: 0.65** 
  - **Mean Reciprocal Rank (MRR): 0.56** 
  - Best performing retrieval method with combined Title-Transcript vector embedding.
  
- **Minisearch:** 
  - Competitive results but not as optimal as ElasticSearch.

- **Hybrid Search:** 
  - Did not achieve the best accuracy or performance compared to ElasticSearch.

Detailed results can be found in the notebooks in the `evaluation` folder. 

## 🔍 RAG Evaluation

Using the cosine similarity metric, the RAG pipeline was evaluated against the ground truth dataset. The similarity scores indicate a moderate level of alignment, with most scores clustering around **0.6**. However, there are noticeable variations, as seen in the spread and peaks of the distribution. The plot shows a bimodal distribution, with peaks around 0.4 and 0.7, suggesting two main groupings of similarity levels.

 ![image](https://github.com/user-attachments/assets/8f9cae8d-1a69-4402-8865-c0f525d547e6)

The moderate similarity scores can be attributed to the nature of the model's responses compared to the transcripts:
- Holistic Responses: The model tends to provide holistic answers that aim to explain and interpret the speaker's intent, going beyond mere excerpts from the transcripts.
- Transcript Limitations: Transcripts are often just segments of a larger conversation, lacking context that might be captured in a more comprehensive response by the model.

## 🖥️ User Interface

Built with **Streamlit**, the UI allows users to input queries and view responses easily.

## Monitoring Feedback and Containerization

User feedback is collected via the UI's thumbs-up👍 and thumbs-down👎 buttons. This feedback is stored in a **PostgreSQL database** and helps in improving the application based on user experiences. The application is containerized using **Docker** to simplify deployment.

A dashboard was created on **Grafana** to analyze the data.

![image](https://github.com/user-attachments/assets/2b05fc4a-c267-418d-83d4-fff47aca276e)



## How to run this code

1. clone the repository to your local machine:
```bash
git clone https://github.com/AishwaryaHastak/TEDTalks-InfoDigest.git
```

2. Navigate to the Project Directory
```
cd app
```

3. Update the `.env.example` file with your environment variables. Make a copy of the file as `.env`:
```
cp .env.example .env
```
Then edit the `.env` file to include your specific database configuration and Hugging Face API token key.

4. Build and start the application using Docker Compose
```bash
docker-compose build
docker-compose up -d
```

5. Once the application is up and running, open your web browser and navigate to:
```
http://localhost:8501
```
---

## Acknowledgements

Detailed steps on how to use ElasticSearch in Python:

https://dylancastillo.co/posts/elasticseach-python.html#create-a-local-elasticsearch-cluster


Ted Talks Transcripts Dataset

https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset
