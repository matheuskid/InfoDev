import os
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreManager:
    """
    Classe utilitária para carregar, processar e gerenciar VectorStores.
    """
    def __init__(self, diretorio, persist_directory, collection_name):
        self.diretorio = diretorio
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = HuggingFaceEmbeddings(
                                    model_name="jinaai/jina-embeddings-v3",
                                    model_kwargs={
                                        'device': 'cpu',
                                        'trust_remote_code': True 
                                    },
                                    encode_kwargs={'batch_size': 1})
        
        os.makedirs(self.diretorio, exist_ok=True)
        os.makedirs(self.persist_directory, exist_ok=True)

        # MELHORIA: Carrega o vectorstore na inicialização para evitar recarregamentos.
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def ingest_documents(self):
        """
        Carrega, divide e armazena os documentos. Executar apenas quando os documentos mudam.
        """
        print(f"Carregando PDFs de '{self.diretorio}'...")
        if not os.listdir(self.diretorio):
            print(f"AVISO: Nenhum arquivo encontrado em '{self.diretorio}'.")
            return

        loader = PyPDFDirectoryLoader(self.diretorio)
        docs = loader.load()
        
        if not docs:
            print(f"Nenhum documento para processar para a coleção '{self.collection_name}'.")
            return

        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
        
        print(f"Criando/atualizando embeddings para {len(chunks)} chunks da coleção '{self.collection_name}'...")
        self.vectorstore.add_documents(chunks)
        print(f"Vectorstore '{self.collection_name}' atualizado.")

    def get_vectorstore(self):
        return self.vectorstore

    def get_retriever(self, k=5):
        print(f"Retriever para a coleção '{self.collection_name}' carregado.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})