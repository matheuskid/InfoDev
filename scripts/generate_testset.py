import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==============================================================================
# IMPORTAÇÕES ATUALIZADAS DO RAGAS (v0.4+)
# ==============================================================================
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.testset.evolutions import simple, reasoning, multi_context

# 1. CARREGAR VARIÁVEIS DE AMBIENTE
load_dotenv(override=True)

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ ERRO: GROQ_API_KEY não encontrada no arquivo .env!")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# ==============================================================================
# 2. CONFIGURAR OS MODELOS (GROQ + JINA)
# ==============================================================================
print("🤖 Inicializando modelos...")

# LLM da Groq
groq_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1 
)

# Embeddings Locais (Jina)
print("📥 Loading Jina Embeddings v3...")
jina_embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v3",
    model_kwargs={
        'device': 'cpu',
        'trust_remote_code': True 
    },
    # THE FIX: Forces Jina to process chunks one by one.
    # It takes milliseconds longer, but completely avoids the PyTorch mismatch crash!
    encode_kwargs={'batch_size': 1} 
)

print("⚙️ Montando o Gerador do Ragas...")

generator = TestsetGenerator.from_langchain(
    generator_llm=groq_llm,
    critic_llm=groq_llm,
    embeddings=jina_embeddings
)

print("✅ Gerador montado com sucesso!")

# ==============================================================================
# 3. CARREGAR DADOS DO MONGODB (CLEAN_SHARK)
# ==============================================================================
print(f"📦 Conectando ao MongoDB em {MONGO_URI}...")
client = MongoClient(MONGO_URI)
db = client["clean_shark"]

docs_langchain = []
target_project = "tez"

# Pegando uma amostra (Pode aumentar se quiser, mas cuidado com os limites da API gratuita da Groq)
print("🔍 Buscando amostras no banco de dados...")
for doc in db.rich_issues.find({"project": target_project}).limit(30):
    docs_langchain.append(Document(
        page_content=doc.get("text_for_embedding", ""),
        metadata={"source": f"Issue_{doc.get('original_id')}", "type": "issue"}
    ))

for doc in db.rich_commits.find({"project": target_project}).limit(30):
    docs_langchain.append(Document(
        page_content=doc.get("text_for_embedding", ""),
        metadata={"source": f"Commit_{doc.get('hash', 'unknown')}", "type": "commit"}
    ))

print(f"✅ Total de documentos carregados: {len(docs_langchain)}")

# ==============================================================================
# CHUNKING
# ==============================================================================
print("✂️ Quebrando documentos em Chunks...")

# Configura o cortador para pedaços de no máximo 1000 caracteres
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100 # Mantém 200 caracteres do pedaço anterior para não perder o contexto
)

# Substitui a lista de documentos inteiros pela lista de pedaços
docs_langchain = text_splitter.split_documents(docs_langchain)

print(f"🧩 Após o corte, temos {len(docs_langchain)} pedaços de documento prontos para o Ragas!")

# ==============================================================================
# 4. GENERATE TESTSET (NO BRAKES)
# ==============================================================================
# Restoring the complex question types
distributions = {
    simple: 0.4,
    reasoning: 0.3,
    multi_context: 0.3
}

# Set your target size here. 50 is a great baseline for testing the final system.
TARGET_QUESTIONS = 10 

print(f"🚀 Generating {TARGET_QUESTIONS} synthetic questions at maximum speed...")
testset = generator.generate_with_langchain_docs(
    documents=docs_langchain,
    test_size=TARGET_QUESTIONS,
    distributions=distributions
    # Notice: run_config is gone! We let Ragas run fully asynchronous now.
)

# ==============================================================================
# 5. EXPORT
# ==============================================================================
df = testset.to_pandas()
df.to_csv("ragas_testset_tcc.csv", index=False)

print(f"\n🎉 SUCCESS! Full dataset saved to 'ragas_testset_tcc.csv'!")
print(df[["question", "evolution_type"]].head())