import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings()
vec = embeddings.embed_query("test connection")

print("API key works. Embedding vector length:", len(vec))
