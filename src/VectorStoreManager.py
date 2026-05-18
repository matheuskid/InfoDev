import os
from tqdm import tqdm
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config import Config

class VectorStoreManager:
    """
    Gerenciador de VectorStore otimizado.
    Cada instância gerencia uma coleção específica, com barra de progresso e processamento em lotes.
    """
    
    def __init__(self, persist_directory, collection_name, model_name):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                'device': Config.EMBEDDINGS_DEVICE,
                'trust_remote_code': Config.EMBEDDINGS_TRUST_REMOTE_CODE
            },
            encode_kwargs={'batch_size': Config.EMBEDDINGS_BATCH_SIZE}
        )
        
        os.makedirs(self.persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def ingest_documents(self, mongo_collection_name, doc_type, mongo_filter=None, batch_size=500):
        """
        Extrai dados de uma coleção específica do Mongo, fatia e salva no Chroma em lotes.
        
        Args:
            mongo_collection_name (str): Nome da coleção no MongoDB (ex: 'rich_commits')
            doc_type (str): Tipo do documento para os metadados (ex: 'commit', 'issue')
            mongo_filter (dict): Filtro para o MongoDB (ex: {"project": "commons-cli"})
            batch_size (int): Quantidade de chunks enviados para a IA por vez.
        """
        if mongo_filter is None:
            mongo_filter = {}

        print(f"\nConectando ao MongoDB ({Config.DB_NAME} -> {mongo_collection_name})...")
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.DB_NAME]
        
        raw_documents = []
        
        print(f"Buscando documentos...")
        for doc in db[mongo_collection_name].find(mongo_filter).limit(1000):  # Limite para evitar sobrecarga, pode ser ajustado
            text = doc.get("text_for_embedding", "")
            if text:
                # Usa hash para commits, original_id para os outros
                doc_id = doc.get("hash") or doc.get("original_id", "unknown")
                raw_documents.append(Document(
                    page_content=text,
                    metadata={"source": f"{doc_type.capitalize()}_{doc_id}", "type": doc_type}
                ))

        if not raw_documents:
            print(f"Nenhum documento encontrado com o filtro: {mongo_filter}")
            return

        print(f"Dividindo {len(raw_documents)} documentos...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(raw_documents)
        total_chunks = len(chunks)
        print(f"Total gerado: {total_chunks} chunks.")
        
        print(f"Iniciando vetorização (Lotes de {batch_size})...")
        
        for i in tqdm(range(0, total_chunks, batch_size), desc=f"Vetorizando '{self.collection_name}'"):
            lote = chunks[i : i + batch_size]
            self.vectorstore.add_documents(lote)
            
        print(f"Ingestão na coleção '{self.collection_name}' concluída com sucesso!")

    def get_vectorstore(self):
        return self.vectorstore

    def get_retriever(self, k=Config.RETRIEVER_K):
        print(f"Retriever para '{self.collection_name}' configurado (k={k}).")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})