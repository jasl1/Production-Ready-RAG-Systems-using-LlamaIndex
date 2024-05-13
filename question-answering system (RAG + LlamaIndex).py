from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.llms import OpenAI
import os

# Load data from a directory of scientific papers
data_dir = "path/to/scientific/papers"
documents = SimpleDirectoryReader(data_dir).load_data()

# Create a vector store index using OpenAI's GPT-3 for embeddings and FAISS as the vector database
embed_model = OpenAI(model_name="text-davinci-003", max_tokens=8192)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
index = GPTVectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    vector_store_config={'type': 'faiss'}
)

# Query the index with a complex question
query = "What are the latest advancements in the field of quantum computing, and how do they compare to classical computing in terms of computational power and potential applications?"

# Use the RAG system to generate a response
response = index.query(query)
print(response)

# Optionally, you can also use the RAG system for follow-up questions
follow_up_query = "Can you provide more details on the potential applications of quantum computing in cryptography and cybersecurity?"
follow_up_response = index.query(follow_up_query, previous_query=query, previous_result=response)
print(follow_up_response)
