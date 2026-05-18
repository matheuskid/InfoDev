import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    """
    Central de Configurações do Sistema.
    Todas as variáveis de ambiente, nomes de modelos e parâmetros globais ficam aqui.
    """
    
    # API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("❌ ERRO: GROQ_API_KEY não encontrada no arquivo .env!")

    # BANCO DE DADOS E COLLECTIONS
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = "clean_shark"
    COLLECTION_ISSUES = "rich_issues"
    COLLECTION_COMMITS = "rich_commits"
    COLLECTION_EMAILS = "rich_emails"

    # MODELOS DE LINGUAGEM
    LLM_MODEL = "llama3-70b-8192"
    LLM_TEMPERATURE = 0.1

    # MODELOS DE EMBEDDINGS E VETORIZAÇÃO
    EMBEDDINGS_MODEL_CODE = "jinaai/jina-embeddings-v2-base-code"
    EMBEDDINGS_MODEL_TEXT = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDINGS_DEVICE = "cpu"
    EMBEDDINGS_TRUST_REMOTE_CODE = True
    EMBEDDINGS_BATCH_SIZE = 32 

    # Parâmetros de fatiamento de texto (Chunking)
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # 5. CONFIGURAÇÕES DOS AGENTES LANGGRAPH
    RETRIEVER_K = 5