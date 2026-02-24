# 🤖 InfoDev: Multi-Agent RAG System

O **InfoDev** é um ecossistema inteligente projetado para otimizar a recuperação e geração de informações técnicas. O foco do projeto é resolver a fragmentação de dados em ambientes de desenvolvimento, utilizando uma arquitetura de múltiplos agentes e **Retrieval-Augmented Generation (RAG)**.

## 📋 Sobre o Projeto

* **⚠️ Problema:** A dificuldade de encontrar informações precisas em grandes volumes de dados técnicos não estruturados.
* **🎯 Objetivo:** Desenvolver um sistema capaz de realizar buscas semânticas e responder a consultas complexas com alta fidelidade.
* **💡 Solução:** Implementação de um pipeline que integra **LangChain + LangGraph**, agentes de IA e bancos de dados vetoriais/documentais.

## 🏗️ Estrutura do Projeto

A organização de pastas reflete a separação entre ambiente isolado, persistência e processamento de dados:

* 📂 `.venv/`: Ambiente virtual Python para isolamento de dependências.
* 📦 `backups/`: Armazena o backup do banco de dados `clean_shark_backup.archive`.
* 🗄️ `data/mongo_db/`: Volume de dados para persistência (MongoDB).
* 📜 `scripts/`: Utilitários para manipulação e preparação de dados.
* 💻 `src/`: Código-fonte principal da aplicação e lógica da arquitetura.
* 🐳 `docker-compose.yml`: Configuração dos containers para o MongoDB.
* 📋 `requirements.txt`: Bibliotecas necessárias (LangChain, PyMongo, etc.).

⚠️ Nota sobre o Git: As pastas 📦 `backups/` e 🗄️ `data/mongo_db/` estão listadas no .gitignore para evitar o versionamento de arquivos pesados ou locais. Certifique-se de criá-las manualmente na raiz do projeto antes de iniciar a execução ou restaurar os backups.

## 🛠️ Metodologia de Desenvolvimento

O projeto InfoDev segue um fluxo estruturado em etapas sequenciais, partindo da curadoria de dados brutos até a implementação de uma inteligência multiagente.

### 1. 📂 Preparação de Dados (Concluído)

Esta etapa compreende desde a aquisição da base bruta até a estruturação de um dataset enriquecido para alimentar o sistema RAG.

#### **1.1 📥 Aquisição e Restauração SmartSHARK 2.1**

Download do **SmartSHARK 2.1**, restaurado em uma **instância Docker**.

#### **1.2 🔍 Análise Exploratória (EDA)**

**Explorative Data Analysis (EDA)** para compreender a distribuição dos dados e avaliar a integridade das informações contidas no dump.

#### **1.3 🎯 Seleção de Projetos Estratégicos**

Script para **identificar os projetos com o maior número de relações** entre:

* **Commits**;
* **Issues**;
* **Emails**.

O **objetivo** é selecionar os projetos que ofereciam o melhor contexto para testar o sistema (informações relacionadas, contidas em fontes diferentes).

#### **1.4 🛠️ Extração e Enriquecimento (Criação do Clean Shark)**

Criação de script para a **extração dos dados** selecionados, aplicando técnicas de manipulação para enriquecer os documentos. O objetivo foi consolidar informações dispersas em um formato mais semântico, separados por collections (`rich_commits`, `rich_issues`, `rich_emails`).

#### **1.5 🧪 Geração do Testset**

O script `generate_testset.py` cria o conjunto de dados de teste para validar se a arquitetura multiagente está recuperando as informações corretas após a implementação.

### 2. 🧠 Processamento e Vetorização (Próximos Passos)

* **🔢 Criação de Vetores Textuais:** Transformação do dataset `clean_shark` em representações matemáticas (embeddings).
* **💾 Persistência Vetorial:** Armazenamento desses vetores em um banco de dados especializado para busca semântica, integrando com o volume de dados do MongoDB.

### 3. 🤖 Arquitetura de Agentes (Em Planejamento)

* **🕸️ Orquestração com LangGraph:** Desenvolvimento de uma estrutura de grafos para gerenciar ciclos de raciocínio e tomada de decisão entre múltiplos agentes.
* **🔗 Integração LangChain:** Uso de cadeias para conectar os modelos de linguagem (LLMs) à base vetorial e ferramentas externas.

## 📊 Análise de Resultados

Ao final, os resultados serão avaliados comparando as respostas da arquitetura de agentes contra o **Testset** gerado inicialmente, medindo métricas de fidelidade e relevância da informação recuperada.
