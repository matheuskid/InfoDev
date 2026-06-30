from Config import Config


class Librarian:
    """
    Agente de recuperação (Retriever) dinâmico. 
    Recebe a categoria definida pelo Router e executa a Busca Híbrida (Vetorial + BM25 com RRF)
    no VectorStoreManager correspondente.
    """
    def __init__(self, dicionario_vsm):
        self.vsm_map = dicionario_vsm

    def run(self, state):
        """
        Método de execução (Nó do LangGraph).
        """
        category = state.get("search_category", "general")
        step_query = state.get("current_step", "")
        
        print(f"--- Agente: Bibliotecário (Buscando em '{category}') ---")
        
        target_vsm = self.vsm_map.get(category)
        
        if not target_vsm:
            print(f"⚠️ Aviso: Nenhum banco de dados configurado para o domínio '{category}'.")
            return {"documents": []}
        
        try:
            print(f"🔍 Executando busca híbrida (Vetorial + BM25 → RRF): '{step_query}'")
            
            valid_docs = target_vsm.hybrid_search(
                query=step_query,
                k_vector=Config.RETRIEVER_K_VECTOR,
                k_bm25=Config.RETRIEVER_K_BM25,
                k_final=Config.RETRIEVER_K_FINAL,
                rrf_k=Config.RRF_K
            )
            
            if not valid_docs:
                print(f"⚠️ Atenção: Nenhum documento relevante encontrado.")
                return {"documents": []}
            
            print(f"📚 Sucesso: {len(valid_docs)} trechos de documentos recuperados via RRF.")
            return {"documents": valid_docs}
            
        except Exception as e:
            print(f"❌ Erro fatal durante a busca no banco vetorial: {e}")
            return {"documents": []}