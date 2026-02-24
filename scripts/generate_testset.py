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
from ragas.testset.evolutions import simple

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
# CHUNKING
# ==============================================================================
print("✂️ Quebrando documentos em Chunks...")

# Configura o cortador para pedaços de no máximo 1000 caracteres
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200 # Mantém 200 caracteres do pedaço anterior para não perder o contexto
)

# Substitui a lista de documentos inteiros pela lista de pedaços
docs_langchain = text_splitter.split_documents(docs_langchain)

print(f"🧩 Após o corte, temos {len(docs_langchain)} pedaços de documento prontos para o Ragas!")

# ==============================================================================
# 4. GERAR O TESTSET
# ==============================================================================
print("⏳ Gerando perguntas sintéticas... (Isso pode demorar dependendo da Groq)")

# Criamos uma regra: Trabalhe com apenas 1 "trabalhador" (sem requisições simultâneas)
# e se der erro de API, tente de novo 5 vezes com pequenos intervalos.
configuracao_lenta = RunConfig(
    max_workers=1, 
    max_retries=5
)

print("⏳ RAGAS operando em modo de segurança (1 requisição por vez)...")
testset = generator.generate_with_langchain_docs(
    documents=docs_langchain,
    test_size=1, 
    # Forçamos 100% das perguntas a serem do tipo "Simples" (Menos tokens e requisições)
    distributions={simple: 1.0},
    # Injetamos o freio de mão no Ragas
    run_config=configuracao_lenta 
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
