import os
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importa a classe Config do arquivo Config.py
from Config import Config

class VectorStoreManager:
    """
    Classe utilitária para carregar, processar e gerenciar VectorStores a partir do MongoDB.
    """
    def __init__(self, persist_directory, collection_name):
        # O diretório local de PDFs foi removido da inicialização
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDINGS_MODEL,
            model_kwargs={
                'device': Config.EMBEDDINGS_DEVICE,
                'trust_remote_code': Config.EMBEDDINGS_TRUST_REMOTE_CODE
            },
            encode_kwargs={'batch_size': Config.EMBEDDINGS_BATCH_SIZE}
        )
        
        os.makedirs(self.persist_directory, exist_ok=True)

        # Carrega o vectorstore na inicialização para evitar recarregamentos.
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def ingest_documents(self, mongo_filter=None):
        """
        Busca os documentos no MongoDB, fatia e armazena no ChromaDB.
        Executar apenas quando precisar popular ou atualizar o banco vetorial.
        
        Args:
            mongo_filter (dict): Filtro opcional para o MongoDB. Ex: {"project": "commons-cli"}
        """
        if mongo_filter is None:
            mongo_filter = {}

        print(f"🔌 Conectando ao MongoDB em {Config.MONGO_URI}...")
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.DB_NAME]
        
        raw_documents = []
        
        # 1. Extração das Issues
        print(f"🔍 Buscando issues na coleção '{Config.COLLECTION_ISSUES}'...")
        for doc in db[Config.COLLECTION_ISSUES].find(mongo_filter):
            text = doc.get("text_for_embedding", "")
            if text:
                raw_documents.append(Document(
                    page_content=text,
                    # Mantendo os metadados em inglês conforme padrão estrutural
                    metadata={"source": f"Issue_{doc.get('original_id', 'unknown')}", "type": "issue"}
                ))
                
        # 2. Extração dos Commits
        print(f"🔍 Buscando commits na coleção '{Config.COLLECTION_COMMITS}'...")
        for doc in db[Config.COLLECTION_COMMITS].find(mongo_filter):
            text = doc.get("text_for_embedding", "")
            if text:
                raw_documents.append(Document(
                    page_content=text,
                    metadata={"source": f"Commit_{doc.get('hash', 'unknown')}", "type": "commit"}
                ))

        print(f"🔍 Buscando e-mails na coleção '{Config.COLLECTION_EMAILS}'...")
        for doc in db[Config.COLLECTION_EMAILS].find(mongo_filter):
            text = doc.get("text_for_embedding", "")
            if text:
                raw_documents.append(Document(
                    page_content=text,
                    # Usando 'original_id' (ou o ID adequado do seu esquema de e-mail)
                    metadata={"source": f"Email_{doc.get('original_id', 'unknown')}", "type": "email"}
                ))

        if not raw_documents:
            print(f"⚠️ Nenhum documento encontrado no MongoDB para processar.")
            return

        # 3. Processamento (Chunking)
        print(f"✂️ Dividindo {len(raw_documents)} documentos inteiros...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(raw_documents)
        
        # 4. Vetorização e Armazenamento
        print(f"🧠 Criando embeddings para {len(chunks)} chunks na coleção Chroma '{self.collection_name}'...")
        self.vectorstore.add_documents(chunks)
        print("✅ Ingestão no VectorStore concluída com sucesso!")

    def get_vectorstore(self):
        return self.vectorstore

    def get_retriever(self, k=Config.RETRIEVER_K):
        print(f"⚙️ Retriever para a coleção '{self.collection_name}' configurado com k={k}.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})