# 🤖 InfoDev: Sistema Multiagente RAG

Sistema inteligente de **Recuperação da Informação** para equipes de desenvolvimento. Combina uma arquitetura de **múltiplos agentes** (LangGraph) com **Retrieval-Augmented Generation (RAG)** e **Busca Híbrida** (Vetorial + BM25 com RRF) para responder perguntas complexas sobre repositórios de software.

## 📋 Sobre o Projeto

* **⚠️ Problema:** A dificuldade de encontrar informações precisas em grandes volumes de dados técnicos não estruturados (commits, issues, emails).
* **🎯 Objetivo:** Desenvolver um sistema capaz de realizar buscas semânticas e responder a consultas complexas com alta fidelidade.
* **💡 Solução:** Pipeline que integra **LangChain + LangGraph**, 7 agentes de IA especializados, busca híbrida (vetorial + lexical) e bancos de dados vetoriais/documentais.

## 🏗️ Arquitetura

### Fluxo Multiagente (LangGraph)

O sistema utiliza **7 agentes especializados** orquestrados por um grafo de estados (`StateGraph`):

```
User Query
    │
    ▼
┌──────────┐
│ Planner  │──► Decompõe a pergunta em passos de investigação
└────┬─────┘
     ▼
┌──────────┐
│  Router  │◄─── (loop) ◄── StepDefiner (próximo passo)
└────┬─────┘                      ▲
     ▼                            │
┌──────────┐               ┌─────┴──────┐
│Librarian │               │StepDefiner │
└────┬─────┘               └─────▲──────┘
     ▼                            │
┌──────────┐               ┌─────┴──────┐
│Extractor │               │ Validator  │── REJECTED → Router (retry)
└────┬─────┘               └────────────┘
     ▼                            ▲
     └────────────────────────────┘
                                  │ (finished)
                                  ▼
                            ┌──────────┐
                            │  Editor  │──► Resposta Final
                            └──────────┘
```

| Agente | Função |
|--------|--------|
| **Planner** | Decompõe a pergunta do usuário em passos de investigação |
| **Router** | Direciona cada passo para a base de dados correta (commits, issues, emails) |
| **Librarian** | Executa **Busca Híbrida** (Vetorial + BM25 → RRF) no VectorStore |
| **Extractor** | Extrai evidências relevantes dos documentos recuperados |
| **Validator** | Audita a qualidade da evidência (APPROVED / REJECTED) |
| **StepDefiner** | Decide se avança para o próximo passo ou retorna ao Router |
| **Editor** | Sintetiza a resposta final com base em todas as evidências coletadas |

### Busca Híbrida (RRF)

O **Librarian** combina dois métodos de recuperação:

1. **Busca Vetorial** (ChromaDB + embeddings) — captura similaridade semântica
2. **Busca Lexical** (BM25) — captura correspondência exata de termos

Os resultados são fundidos com **Reciprocal Rank Fusion (RRF)**, que combina os rankings das duas buscas em um ranking unificado mais robusto:

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```

### Dual-LLM

O sistema utiliza modelos diferentes otimizados por tarefa:

- **Modelo Geral** (`openai/gpt-oss-120b`): Planner, Router, Validator, StepDefiner, Editor
- **Modelo Extrator** (`llama-3.3-70b-versatile`): Extractor — melhor recall na leitura de chunks grandes

## 📂 Estrutura do Projeto

```
InfoDev/
├── App.py                           # Interface Streamlit
├── avaliacaoRAGAS.py                # Pipeline de avaliação com RAGAS
├── docker-compose.yml               # Container MongoDB
├── scripts/
│   ├── script_clean_shark_rich.py   # Extração e enriquecimento de dados
│   ├── generate_testset.py          # Geração do testset com RAGAS
│   ├── build_final_dataset.py       # Curadoria do dataset final (50 questões)
│   └── diagnostico.py               # Diagnóstico do banco de dados
├── src/
│   ├── Config.py                    # Configurações centralizadas
│   ├── Graph.py                     # Orquestração do grafo multiagente
│   ├── GraphState.py                # Estado compartilhado entre agentes
│   ├── VectorStoreManager.py        # Gerenciador de VectorStore + Busca Híbrida
│   ├── Util.py                      # Utilitários
│   └── Agents/
│       ├── Planner.py
│       ├── Router.py
│       ├── Librarian.py
│       ├── Extractor.py
│       ├── Validator.py
│       ├── StepDefiner.py
│       └── Editor.py
└── Playground.ipynb                 # Notebook de experimentação
```

> ⚠️ **Nota:** As pastas `backups/`, `data/mongo_db/`, `vectorstores/` e arquivos `.csv` estão no `.gitignore`. Certifique-se de criá-las manualmente e restaurar os backups antes de executar.

## 🛠️ Metodologia de Desenvolvimento

### 1. 📂 Preparação de Dados ✅

Desde a aquisição da base bruta até a estruturação de um dataset enriquecido:

1. **Aquisição e Restauração SmartSHARK 2.1** — Restaurado em instância Docker
2. **Análise Exploratória (EDA)** — Distribuição e integridade dos dados
3. **Seleção de Projetos Estratégicos** — Identificação dos projetos com mais relações (commits × issues × emails)
4. **Extração e Enriquecimento** — Criação do `clean_shark` com collections enriquecidas (`rich_commits`, `rich_issues`, `rich_emails`)
5. **Geração do Testset** — Dataset de avaliação gerado com RAGAS

### 2. 🧠 Processamento e Vetorização ✅

* **Embeddings especializados:** `jina-embeddings-v2-base-code` (commits) + `nomic-embed-text-v1.5` (issues/emails)
* **Persistência:** ChromaDB com distância cosine
* **Chunking:** `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=300`
* **Índice BM25:** Construído sob demanda (lazy initialization) para busca lexical

### 3. 🤖 Arquitetura Multiagente ✅

* **7 agentes** orquestrados com LangGraph (StateGraph)
* **Busca Híbrida:** Vetorial + BM25 com Reciprocal Rank Fusion
* **Dual-LLM:** Modelos especializados por tarefa
* **Interface:** App Streamlit com feedback visual em tempo real

### 4. 📊 Avaliação com RAGAS ✅

Pipeline automatizado de avaliação (`avaliacaoRAGAS.py`) com métricas:

* **Faithfulness** — Fidelidade da resposta às evidências
* **Answer Relevancy** — Relevância da resposta à pergunta
* **Context Precision** — Precisão do contexto recuperado
* **Context Recall** — Cobertura do contexto recuperado

Dataset final: **43 perguntas** curadas, avaliadas com LLM juiz independente (`llama-3.3-70b-versatile`).

## 🚀 Como Executar

### Pré-requisitos

* Python 3.10+
* Docker (para MongoDB)
* Chave de API: `GROQ_API_KEY` no arquivo `.env`

### Setup

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Subir o MongoDB
docker-compose up -d

# 4. Restaurar backup (se necessário)
# mongorestore --archive=backups/clean_shark_backup.archive --db=clean_shark

# 5. Executar a interface
streamlit run App.py
```
