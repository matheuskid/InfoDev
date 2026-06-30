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
    LLM_MODEL = "openai/gpt-oss-120b"
    LLM_TEMPERATURE = 0.0

    # MODELOS DE EMBEDDINGS E VETORIZAÇÃO
    EMBEDDINGS_MODEL_CODE = "jinaai/jina-embeddings-v2-base-code"
    EMBEDDINGS_MODEL_TEXT = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDINGS_DEVICE = "cpu"
    EMBEDDINGS_TRUST_REMOTE_CODE = True
    EMBEDDINGS_BATCH_SIZE = 32 

    # Parâmetros de fatiamento de texto (Chunking)
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300

    # 5. CONFIGURAÇÕES DOS AGENTES LANGGRAPH
    RETRIEVER_K = 7

    # 6. BUSCA HÍBRIDA (RRF)
    RRF_K = 60               # Parâmetro de suavização do Reciprocal Rank Fusion
    RETRIEVER_K_VECTOR = 10  # Candidatos da busca vetorial (antes do merge)
    RETRIEVER_K_BM25 = 10    # Candidatos da busca BM25 (antes do merge)
    RETRIEVER_K_FINAL = 7    # Docs finais após fusão RRF