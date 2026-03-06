import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env assim que o módulo é importado
load_dotenv(override=True)

class Config:
    """
    Central de Configurações do Sistema.
    Todas as variáveis de ambiente, nomes de modelos e parâmetros globais ficam aqui.
    """
    
    # =========================================================================
    # 1. AUTENTICAÇÃO E API
    # =========================================================================
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("❌ ERRO: GROQ_API_KEY não encontrada no arquivo .env!")

    # =========================================================================
    # 2. BANCO DE DADOS (MONGODB)
    # =========================================================================
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = "clean_shark"
    COLLECTION_ISSUES = "rich_issues"
    COLLECTION_COMMITS = "rich_commits"
    COLLECTION_EMAILS = "rich_emails"

    # =========================================================================
    # 3. MODELOS DE LINGUAGEM (LLM)
    # =========================================================================
    # Usando o Llama 3 70B para máximo raciocínio lógico nos agentes
    LLM_MODEL = "llama3-70b-8192"
    LLM_TEMPERATURE = 0.1  # Temperatura baixa para respostas mais determinísticas e precisas

    # =========================================================================
    # 4. MODELOS DE EMBEDDINGS E VETORIZAÇÃO
    # =========================================================================
    EMBEDDINGS_MODEL = "jinaai/jina-embeddings-v3"
    EMBEDDINGS_DEVICE = "cpu"
    EMBEDDINGS_TRUST_REMOTE_CODE = True
    EMBEDDINGS_BATCH_SIZE = 1  # Evita o erro de alinhamento de tensores no PyTorch

    # Parâmetros de fatiamento de texto (Chunking)
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # =========================================================================
    # 5. CONFIGURAÇÕES DOS AGENTES LANGGRAPH
    # =========================================================================
    # Quantidade máxima de documentos que o VectorStore deve retornar na busca
    RETRIEVER_K = 5