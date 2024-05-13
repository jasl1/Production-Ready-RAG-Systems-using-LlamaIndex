
from llama_index import GPTVectorStoreIndex, ServiceContext, LLMPredictor, StorageContext, load_index_from_storage
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader
import os

# Load data from multiple sources (text files, PDFs, and other file types)
data_dir = "path/to/knowledge/base"
loaders = [TextLoader(os.path.join(data_dir, file)) for file in os.listdir(data_dir) if file.endswith(".txt")]
loaders += [PyPDFLoader(os.path.join(data_dir, file)) for file in os.listdir(data_dir) if file.endswith(".pdf")]
loaders += [UnstructuredFileLoader(os.path.join(data_dir, file)) for file in os.listdir(data_dir) if not file.endswith((".txt", ".pdf"))]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Create a vector store index using OpenAI's GPT-3 for embeddings and Weaviate as the vector database
embed_model = OpenAI(model_name="text-davinci-003", max_tokens=8192)
service_context = ServiceContext.from_defaults(embed_model=embed_model, vector_store_config={'type': 'weaviate'})
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# Save the index to persistent storage
storage_context = StorageContext.from_defaults()
index.storage_context.persist(persist_dir="path/to/storage")

# Load the index from persistent storage
loaded_index = load_index_from_storage(storage_context, persist_dir="path/to/storage")

# Query the index with a complex question
query = "What are the key factors to consider when designing a secure and scalable cloud infrastructure?"

# Use the RAG system to generate a response
response = loaded_index.query(query)
print(response)
