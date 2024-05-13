from llama_index import GPTVectorStoreIndex, ServiceContext, ResponseEvaluator
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredFileLoader
import os
import random

# Load data from product catalog
data_dir = "path/to/product/catalog"
loaders = [UnstructuredFileLoader(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Create a vector store index using OpenAI's GPT-3 for embeddings and FAISS as the vector database
embed_model = OpenAI(model_name="text-davinci-003", max_tokens=8192)
service_context = ServiceContext.from_defaults(embed_model=embed_model, vector_store_config={'type': 'faiss'})
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# Define ground truth questions and answers
ground_truth = {}
for doc in random.sample(documents, 100):
    question = f"What are the key features of the {doc.get_doc_id()} product?"
    answer = doc.get_text()
    ground_truth[question] = answer

# Evaluate the index
evaluator = ResponseEvaluator(ground_truth)
metrics = evaluator.evaluate(index)

print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1 Score: {metrics['f1']}")
