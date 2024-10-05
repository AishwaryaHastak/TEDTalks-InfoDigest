# Ted Talks Explorer Application

Ted Talks are a source of profound knowledge and opinions that experts from their fields deliver in a fun and easy to digest manner. This project is designed to provide answers and summary of the most viewed Ted Talks since 2006. The app aims to provide an easy to use interface to gain insights and quick summaries from the talks. 🚀

The application utilizes tools like ElasticSearch, Streamlit, PostgreSQL, Grafana, and Docker.

![TED_assistant_Diagram](https://github.com/user-attachments/assets/51a5ef39-3616-4867-80be-2d45f283c8e7)


### 🔍📝👉 To learn more about RAGs, check out this [article](https://medium.com/@aishwaryahastak/understanding-the-roots-of-rags-7b77d26c3dca).

## 📈Project Overview

This application utilizes the ![TED Talks dataset](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset) provided by Miguel Corral Jr. on Kaggle. These datasets contain over 4,000 TED talks including transcripts in many languages. For this application we are only focusing on the 100 most viewed TED talks that cover many topics such as global issues, culture, business, technology.


## RAG Flow

The Retrieval-Augmented Generation (RAG) flow combines a knowledge base with a language model to deliver accurate responses:

- **Knowledge Base:** Contains a Data Science Q&A dataset stored in `data.csv`.
- **Language Model:** Uses **Flan-T5**, an open-source model from Google available on Hugging Face, for augmented response generation.

## 📊 Retrieval Evaluation

The performance of retrieval methods was assessed using `ground-truth.csv`. The following methods were evaluated:

- **ElasticSearch:** 
  - **Hit Rate:** 
  - **Mean Reciprocal Rank (MRR):** 
  - Best performing retrieval method with combined Question-Answer vector embedding.
  
- **Minisearch:** 
  - Competitive results but not as optimal as ElasticSearch.

- **Hybrid Search:** 
  - Did not achieve the best accuracy or performance compared to ElasticSearch.

Detailed results can be found in the notebooks in the `evaluation` folder. 

## 🔍 RAG Evaluation

The RAG pipeline was evaluated against the ground truth dataset using the cosine similarity metric. The system achieved a cosine similarity score of **xyz**, reflecting strong alignment with the expected results. 

![image](https://github.com/user-attachments/assets/4120dc26-6a43-4a3a-b2fe-e5ec5de7cb5a)


## 🖥️ User Interface

The application features a simple and intuitive UI built with **Streamlit**. Users can easily input queries and view responses through a straightforward interface. 

![image](https://github.com/user-attachments/assets/a62fdc48-2c3a-4560-9236-18c7fc52511d)

![image](https://github.com/user-attachments/assets/35cdf80f-272c-4415-8ea6-1ddf30deb70e)


## Monitoring Feedback and Containerization

User feedback is collected via thumbs-up👍 and thumbs-down👎 buttons in the UI. This feedback is stored in a **PostgreSQL database** and helps in improving the application based on user experiences. The application is containerized using **Docker** to simplify deployment.

A dashboard was created on **Grafana** to analyze the data.

![image](https://github.com/user-attachments/assets/e4b8d943-7e45-4fce-8d81-b5bad15adb1e)

- The model performs well on questions related to Supervised Learning.
- Vector-based search is about **X times faster** than text-based search.


## How to run this code

1. clone the repository to your local machine:
```bash
git clone https://github.com/AishwaryaHastak/ted_rag.git
```

2. Navigate to the Project Directory
```
cd app
```

3. Update the `.env.example` file with your environment variables. Make a copy of the file as `.env`:
```
cp .env.example .env
```
Then edit the `.env` file to include your specific configuration.

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