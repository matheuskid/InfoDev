import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from dotenv import load_dotenv

# Importes do LangChain e Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.append('./src')

# Importe a função que constrói o seu grafo do InfoDev
from Graph import build_graph

# 1. SETUP INICIAL E VARIÁVEIS DE AMBIENTE
print("⚙️ Carregando variáveis de ambiente...")
load_dotenv(override=True)

# 2. SETUP DO JUIZ (LLM e Embeddings para o RAGAS)
print("⚖️ Inicializando o LLM Juiz do RAGAS...")
# O juiz do RAGAS permanece fixo para comparações justas
llm_juiz = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# O RAGAS precisa de embeddings para calcular a similaridade de algumas métricas
embeddings_juiz = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-code", 
    model_kwargs={'trust_remote_code': True}
)

# 3. INSTANCIANDO O SEU SISTEMA (InfoDev)
print("🤖 Montando a arquitetura multiagente do InfoDev...")
from Config import Config
# LLM principal (Editor, Planner, Router, Validator, etc)
llm_sistema = ChatGroq(model_name="openai/gpt-oss-120b", temperature=Config.LLM_TEMPERATURE)
# LLM especializado na Extração (tem melhor recall lendo chunks grandes e sujos)
llm_extractor = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=Config.LLM_TEMPERATURE)

print(f"   Modelo Geral: openai/gpt-oss-120b (temp={Config.LLM_TEMPERATURE})")
print(f"   Modelo Extrator: llama-3.3-70b-versatile (temp={Config.LLM_TEMPERATURE})")
workflow = build_graph(llm_sistema, llm_extractor)

# 4. CARREGANDO O DATASET DE TESTES
nome_do_arquivo_csv = "ragas_testset_final_43.csv" # <-- COLOQUE O NOME DO SEU CSV AQUI
print(f"📂 Carregando dataset de testes local: {nome_do_arquivo_csv}")
df_testes = pd.read_csv(nome_do_arquivo_csv)

# Dicionário que o RAGAS vai ler
dados_para_ragas = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

# 5. EXECUÇÃO DA BATERIA DE TESTES
print("\n🚀 Iniciando a bateria de testes no InfoDev...")
config = {"recursion_limit": 100}

# Percorre cada linha do seu CSV
for index, row in df_testes.iterrows():
    pergunta_atual = row["question"]
    resposta_esperada = row["ground_truth"]
    
    print(f"\n[{index+1}/{len(df_testes)}] Testando: '{pergunta_atual}'")
    
    # Prepara o estado inicial (A mesma estrutura do seu App.py)
    estado = {
        "query": pergunta_atual,
        "plan": [],
        "current_step": "",
        "evidence": [],
        "failed_categories": [],
        "retry_count": 0
    }
    
    resposta_gerada = ""
    contextos_recuperados = []
    
    try:
        # Roda o seu grafo
        for event in workflow.stream(estado, config=config):
            for node_name, state_updates in event.items():
                
                # Captura a resposta escrita pelo Agente Editor
                if node_name == "editor" and "final_answer" in state_updates:
                    resposta_gerada = state_updates["final_answer"]
                
                # Captura as evidências resumidas pelo Agente Extrator
                if "evidence" in state_updates:
                    # Garantimos que seja sempre uma lista de strings acumulada
                    contextos_recuperados.extend(state_updates["evidence"])

        print(f"✅ Resposta gerada com sucesso! ({len(contextos_recuperados)} fragmentos de contexto usados).")

    except Exception as e:
        print(f"❌ Erro ao processar a pergunta: {e}")
        resposta_gerada = "Error during execution."
        contextos_recuperados = ["Error"]

    # Alimenta o dicionário do RAGAS com o que o SEU sistema produziu
    dados_para_ragas["question"].append(pergunta_atual)
    dados_para_ragas["ground_truth"].append(resposta_esperada)
    dados_para_ragas["answer"].append(resposta_gerada)
    dados_para_ragas["contexts"].append(contextos_recuperados)

# 6. AVALIAÇÃO COM RAGAS
print("\n" + "="*40)
print("📊 Iniciando Avaliação Matemática com RAGAS...")
print("="*40)

# Converte o dicionário para o formato HuggingFace Dataset na Memória RAM
dataset_avaliacao = Dataset.from_dict(dados_para_ragas)

# Define as 4 métricas principais citadas no seu TCC
metricas = [
    faithfulness,        # Avalia alucinação (a resposta baseia-se apenas nos contextos?)
    answer_relevancy,    # A resposta realmente responde à pergunta?
    context_precision,   # Os contextos recuperados são relevantes?
    context_recall       # Os contextos trazidos cobrem tudo o que era necessário para a resposta certa?
]

# Roda o framework
resultado = evaluate(
    dataset=dataset_avaliacao,
    metrics=metricas,
    llm=llm_juiz,
    embeddings=embeddings_juiz
)

# 7. EXPORTAÇÃO DOS RESULTADOS FINAIS
df_resultado = resultado.to_pandas()
arquivo_saida = "resultado_avaliacao_hybrid_rrf_split.csv"
df_resultado.to_csv(arquivo_saida, index=False)

print("\n🏆 RESULTADOS MÉDIOS FINAIS:")
print(resultado)
print(f"\n✅ Detalhes salvos com sucesso em '{arquivo_saida}'!")