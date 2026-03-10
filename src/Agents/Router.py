from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class Router:
    """
    Agente classificador que direciona a pergunta para o banco de dados (VectorStore) correto,
    respeitando os domínios que já falharam anteriormente (blacklist).
    """
    def __init__(self, llm):
        self.llm = llm
        
        # Melhoria Arquitetural: Um único prompt que lida dinamicamente com a blacklist
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em classificação. Classifique a consulta para o domínio correto.
            
            DOMÍNIOS DISPONÍVEIS:
            1. 'estrategico': Direção, objetivos, metas de longo prazo.
            2. 'executivo': Regras internas, procedimentos, regulamentos internos.
            3. 'legislativo': Leis federais, decretos, estatutos.
            
            Se a variável 'Domínios proibidos' contiver algum nome, você NÃO PODE escolhê-lo em hipótese alguma.
            Domínios proibidos: {blacklist}
            
            Retorne APENAS a palavra do domínio escolhido (estrategico, executivo ou legislativo). Não explique."""),
            ("human", "{query}")
        ])
        
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agente: Router (LLM Classification) ---")
        
        step_query = state.get("current_step", "")
        print(f"A rotear: '{step_query}'")
        
        # Puxar categorias falhas (Blacklist)
        blacklist = state.get("failed_categories", [])
        blacklist_str = ", ".join(blacklist) if blacklist else "Nenhum"
        
        if blacklist:
            print(f"⚠️ Ignorando domínios (Blacklist): {blacklist_str}")
        
        try:
            # Invocar o LLM injetando a blacklist
            response = self.chain.invoke({
                "query": step_query,
                "blacklist": blacklist_str
            })

            new_usage = update_token_usage(state, response)
            category = response.content.strip().lower()
            
            # Limpeza e garantia da saída
            if "estrategico" in category: 
                category = "estrategico"
            elif "executivo" in category: 
                category = "executivo"
            elif "legislativo" in category: 
                category = "legislativo"
            else:
                category = "general" # Fallback caso o LLM invente palavras
                    
        except Exception as e:
            print(f"❌ Erro no Roteamento: {e}")
            category = "general"
            new_usage = state.get("token_usage", {})

        print(f"🧭 Decisão de Roteamento: {category}")
        return {"search_category": category, "token_usage": new_usage}