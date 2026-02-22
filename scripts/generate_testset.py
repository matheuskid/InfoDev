import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================================================================
# IMPORTAÇÕES ATUALIZADAS DO RAGAS (v0.4+)
# ==============================================================================
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 1. CARREGAR VARIÁVEIS DE AMBIENTE
load_dotenv(override=True)

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ ERRO: GROQ_API_KEY não encontrada no arquivo .env!")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# ==============================================================================
# 2. CONFIGURAR OS MODELOS (GROQ + JINA)
# ==============================================================================
print("🤖 Inicializando modelos...")

# LLM da Groq (Llama 3 70B)
groq_llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.1 
)

# Embeddings Locais (Jina)
print("📥 Carregando modelo de embeddings da Jina AI...")
jina_embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v3",
    model_kwargs={
        'device': 'cpu',
        'trust_remote_code': True 
    }
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
# 4. GERAR O TESTSET
# ==============================================================================
print("⏳ Gerando perguntas sintéticas... (Isso pode demorar dependendo da Groq)")

testset = generator.generate_with_langchain_docs(
    docs_langchain,
    test_size=5 # Reduzi para 5 inicialmente para testar os limites de taxa (Rate Limits) da Groq
)

# ==============================================================================
# 5. SALVAR RESULTADOS
# ==============================================================================
df = testset.to_pandas()
output_file = "ragas_testset_tcc.csv"
df.to_csv(output_file, index=False)

print(f"\n🎉 Sucesso! Testset gerado e salvo em '{output_file}'!")
print("\nUma espiada nas perguntas geradas:")
# O novo dataframe costuma ter as colunas 'user_input' (a pergunta) e 'reference' (o gabarito)
print(df.head())
