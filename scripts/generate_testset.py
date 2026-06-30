import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# IMPORTAÇÕES ATUALIZADAS DO RAGAS (v0.4+)
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from ragas.testset.evolutions import simple, reasoning, multi_context

# 1. CARREGAR VARIÁVEIS DE AMBIENTE
load_dotenv(override=True)

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ ERRO: GROQ_API_KEY não encontrada no arquivo .env!")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# 2. CONFIGURAR OS MODELOS (GROQ + JINA)
print("🤖 Inicializando modelos...")

# LLM da Groq
groq_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1 
)

# Embeddings Locais (Jina)
print("📥 Loading Jina Embeddings v2...")
jina_embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-code",
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

# 3. CARREGAR DADOS DO MONGODB (CLEAN_SHARK)
print(f"📦 Conectando ao MongoDB em {MONGO_URI}...")
client = MongoClient(MONGO_URI)
db = client["clean_shark"]

docs_langchain = []
target_project = "tez"

# Pula os 100 primeiros documentos (que já foram usados) e pega os próximos 150
print("🔍 Buscando novas amostras no banco de dados...")
for doc in db.rich_issues.find({"project": target_project}).skip(100).limit(150):
    docs_langchain.append(Document(
        page_content=doc.get("text_for_embedding", ""),
        metadata={"source": f"Issue_{doc.get('original_id')}", "type": "issue"}
    ))

for doc in db.rich_commits.find({"project": target_project}).skip(100).limit(150):
    docs_langchain.append(Document(
        page_content=doc.get("text_for_embedding", ""),
        metadata={"source": f"Commit_{doc.get('hash', 'unknown')}", "type": "commit"}
    ))

print(f"✅ Total de documentos carregados: {len(docs_langchain)}")

# CHUNKING
print("✂️ Quebrando documentos em Chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Substitui a lista de documentos inteiros pela lista de pedaços
docs_langchain = text_splitter.split_documents(docs_langchain)

print(f"🧩 Após o corte, temos {len(docs_langchain)} pedaços de documento prontos para o Ragas!")


# 4. GENERATE TESTSET (NO BRAKES)
distributions = {
    simple: 0.2,
    reasoning: 0.4,
    multi_context: 0.4
}

TARGET_QUESTIONS = 50 

print(f"🚀 Generating {TARGET_QUESTIONS} synthetic questions...")
testset = generator.generate_with_langchain_docs(
    documents=docs_langchain,
    test_size=TARGET_QUESTIONS,
    distributions=distributions
)

# 5. EXPORT
df = testset.to_pandas()
# Remove as perguntas inválidas geradas automaticamente para já sair limpo
df = df[df['ground_truth'] != 'The answer to given question is not present in context']
df.to_csv("ragas_testset_novas_50.csv", index=False)

print(f"\n🎉 SUCESSO! Novo dataset (apenas válidas) salvo em 'ragas_testset_novas_50.csv'!")
print(df[["question", "evolution_type"]].head())