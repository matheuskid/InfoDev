import os
from tqdm import tqdm
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from Config import Config

class VectorStoreManager:
    """
    Gerenciador de VectorStore otimizado.
    Cada instância gerencia uma coleção específica, com barra de progresso e processamento em lotes.
    Suporta Busca Híbrida (Vetorial + BM25) com Reciprocal Rank Fusion (RRF).
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

        # Índice BM25 — construído sob demanda (lazy)
        self._bm25_index = None
        self._bm25_docs = None

    def _build_bm25_index(self):
        """
        Constrói o índice BM25 a partir de todos os documentos armazenados no Chroma.
        Executado uma única vez (lazy initialization).
        """
        print(f"📚 Construindo índice BM25 para '{self.collection_name}'...")
        
        # Recupera todos os documentos do Chroma em lotes (evita erro de SQLite com muitas variáveis)
        documents = []
        batch_size = 5000
        offset = 0
        
        while True:
            batch = self.vectorstore.get(limit=batch_size, offset=offset)
            batch_docs = batch.get("documents", [])
            if not batch_docs:
                break
            documents.extend(batch_docs)
            offset += len(batch_docs)
            if len(batch_docs) < batch_size:
                break
        
        if not documents:
            print(f"⚠️ Nenhum documento encontrado no Chroma para '{self.collection_name}'. BM25 ficará vazio.")
            self._bm25_docs = []
            self._bm25_index = None
            return
        
        self._bm25_docs = documents
        
        # Tokenização simples: lowercase + split por espaço
        tokenized_docs = [doc.lower().split() for doc in documents]
        self._bm25_index = BM25Okapi(tokenized_docs)
        
        print(f"✅ Índice BM25 construído: {len(documents)} chunks indexados para '{self.collection_name}'.")

    def _bm25_search(self, query, k=10):
        """
        Executa busca BM25 e retorna os top-k documentos como lista de strings.
        """
        if self._bm25_index is None:
            return []
        
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        
        # Ordena pelos maiores scores e pega os top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [self._bm25_docs[i] for i in top_indices if scores[i] > 0]

    def hybrid_search(self, query, k_vector=None, k_bm25=None, k_final=None, rrf_k=None):
        """
        Busca Híbrida: combina busca vetorial (Chroma) + busca lexical (BM25)
        e funde os resultados com Reciprocal Rank Fusion (RRF).
        
        Args:
            query: A query de busca.
            k_vector: Quantidade de candidatos da busca vetorial.
            k_bm25: Quantidade de candidatos da busca BM25.
            k_final: Quantidade final de documentos após fusão.
            rrf_k: Parâmetro de suavização do RRF.
        
        Returns:
            Lista de strings (page_content) dos documentos mais relevantes.
        """
        k_vector = k_vector or Config.RETRIEVER_K_VECTOR
        k_bm25 = k_bm25 or Config.RETRIEVER_K_BM25
        k_final = k_final or Config.RETRIEVER_K_FINAL
        rrf_k = rrf_k or Config.RRF_K

        # Lazy init do BM25
        if self._bm25_index is None and self._bm25_docs is None:
            self._build_bm25_index()

        # 1. Busca Vetorial (Chroma)
        vector_results = self.vectorstore.similarity_search(query, k=k_vector)
        vector_docs = [doc.page_content for doc in vector_results]
        
        # 2. Busca BM25
        bm25_docs = self._bm25_search(query, k=k_bm25)
        
        print(f"   🔎 Vetorial: {len(vector_docs)} docs | BM25: {len(bm25_docs)} docs")

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(vector_docs, bm25_docs, rrf_k)
        
        return fused[:k_final]

    @staticmethod
    def _reciprocal_rank_fusion(list_a, list_b, k=60):
        """
        Reciprocal Rank Fusion (RRF) — funde dois rankings em um único ranking.
        
        RRF_score(doc) = Σ 1 / (k + rank_i(doc))
        
        Args:
            list_a: Primeiro ranking (lista de strings).
            list_b: Segundo ranking (lista de strings).
            k: Parâmetro de suavização (default=60).
        
        Returns:
            Lista de strings ordenada pelo score RRF (maior → menor).
        """
        scores = {}
        
        for rank, doc in enumerate(list_a):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        
        for rank, doc in enumerate(list_b):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        
        # Ordena pelo score RRF decrescente
        sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        
        return sorted_docs

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

        # Invalida o cache do BM25 para forçar reconstrução
        self._bm25_index = None
        self._bm25_docs = None

    def get_vectorstore(self):
        return self.vectorstore

    def get_retriever(self, k=Config.RETRIEVER_K):
        print(f"Retriever para '{self.collection_name}' configurado (k={k}).")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})