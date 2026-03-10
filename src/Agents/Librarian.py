class Librarian:
    """
    Agente de recuperação (Retriever) dinâmico. 
    Recebe a categoria definida pelo Router e executa a busca no VectorStore correspondente.
    """
    def __init__(self, dicionario_retrievers):
        # 1. Recebe o dicionário completo com todos os retrievers instanciados no App.py
        # Ex: {"estrategico": retriever_est, "executivo": retriever_exe, ...}
        self.retrievers = dicionario_retrievers

    def run(self, state):
        """
        Método de execução (Nó do LangGraph).
        """
        category = state.get("search_category", "general")
        step_query = state.get("current_step", "")
        
        print(f"--- Agente: Bibliotecário (Buscando em '{category}') ---")
        
        # 2. Busca a ferramenta de recuperação correta com base na decisão do Router
        target_retriever = self.retrievers.get(category)
        
        if not target_retriever:
            print(f"⚠️ Aviso: Nenhum banco de dados configurado para o domínio '{category}'.")
            # Pode retornar vazio ou redirecionar para um retriever padrão (fallback)
            return {"documents": []}
        
        # 3. Executa a busca vetorial usando o MultiQueryRetriever
        try:
            print(f"🔍 Executando query: '{step_query}'")
            retrieved_docs = target_retriever.invoke(step_query)
            
            # Extrai apenas o texto útil dos documentos LangChain
            doc_contents = [doc.page_content for doc in retrieved_docs]
            
            print(f"📚 Sucesso: {len(doc_contents)} trechos de documentos recuperados.")
            return {"documents": doc_contents}
            
        except Exception as e:
            print(f"❌ Erro fatal durante a busca no banco vetorial: {e}")
            return {"documents": []}