from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class Editor:
    """
    Agente de síntese final. Pega todas as evidências aprovadas durante a execução
    do plano e redige a resposta final para a pergunta original do usuário.
    """
    def __init__(self, llm):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_template(
            "You are a question-answering assistant. Use the collected evidence to answer the user's original question.\n"
            "Answer only with facts. Remove redundant information.\n\n"
            "Original Question: \"{query}\"\n\n"
            "Collected Evidence:\n---\n{evidence}\n---\n"
            "Final Answer:"
        )
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agente: Editor (Gerando Resposta Final) ---")
        
        evidence_list = state.get("evidence", [])
        full_context = "\n\n".join(evidence_list)
        
        original_query = state.get("query", "")
        
        response = self.chain.invoke({
            "query": original_query, 
            "evidence": full_context
        })

        new_usage = update_token_usage(state, response)
        final_answer = response.content.strip()
        
        print("🎉 Resposta final gerada com sucesso.")
        
        return {
            "final_answer": final_answer, 
            "token_usage": new_usage
        }