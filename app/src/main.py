# from src.mspipeline import MiniSearchRAGPipeline
# from src.espipeline import ElSearchRAGPipeline
# from mspipeline import MiniSearchRAGPipeline
from espipeline import ElSearchRAGPipeline

# ms = MiniSearchRAGPipeline()
es = ElSearchRAGPipeline()
query1 = "What was the comparison between the USA and Vietnam?"
topic1 = 'global issues'
query2 = "What did the speaker mean when he mentioned 'Courage calls to courage everywhere'"
query3 = "What is Howard's notion of the Platonic dish?"
topic3 = 'business'
es.read_data()
es.create_index()
print(es.get_response(query3,topic3))