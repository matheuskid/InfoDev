class Librarian:
    """
    Agente de recuperação (Retriever) dinâmico. 
    Recebe a categoria definida pelo Router e executa a busca no VectorStore correspondente.
    """
    def __init__(self, dicionario_retrievers):
        self.retrievers = dicionario_retrievers

    def run(self, state):
        """
        Método de execução (Nó do LangGraph).
        """
        category = state.get("search_category", "general")
        step_query = state.get("current_step", "")
        
        print(f"--- Agente: Bibliotecário (Buscando em '{category}') ---")
        
        target_retriever = self.retrievers.get(category)
        
        if not target_retriever:
            print(f"⚠️ Aviso: Nenhum banco de dados configurado para o domínio '{category}'.")
            return {"documents": []}
        
        try:
            print(f"🔍 Executando query: '{step_query}'")
            retrieved_docs = target_retriever.invoke(step_query)
            
            doc_contents = [doc.page_content for doc in retrieved_docs]
            
            print(f"📚 Sucesso: {len(doc_contents)} trechos de documentos recuperados.")
            return {"documents": doc_contents}
            
        except Exception as e:
            print(f"❌ Erro fatal durante a busca no banco vetorial: {e}")
            return {"documents": []}