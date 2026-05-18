from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class Planner:
    """
    Agente responsável por desconstruir a pergunta do utilizador num plano
    de consultas de pesquisa simples e eficientes.
    """
    def __init__(self, llm):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_template(
            "You are an efficient search planner. Your goal is to break down the user's question into simple and efficient search queries.\n"
            "For each question, determine if it is complex/compound or simple.\n"
            "**If complex:** split it into multiple search queries focused on specific topics.\n"
            "Create clear and direct search queries for each identified topic.\n"
            "Do not create redundant queries. Do not create steps that are not search queries.\n"
            "Strictly follow what was asked. Do not invent extra queries if not requested. Do not create queries intended for internet/web searches.\n"
            "**If simple:** create only one query that directly answers the question.\n"
            "Question: {query}\n\n"
            "Return ONLY the query OR the list of queries separated by newlines."
        )
        
        self.chain = self.prompt | self.llm

    def run(self, state):
        """
        Método de execução (Nó do LangGraph). 
        Recebe o GraphState, executa a lógica e devolve as atualizações.
        """
        print("--- Agente: Líder (Planeamento) ---")
        
        if state.get("plan"):
            print("✅ Plano já existente. A avançar...")
            return {}
        
        print("A gerar plano estratégico...")
        
        response = self.chain.invoke({"query": state["query"]})
        
        new_usage = update_token_usage(state, response)
        plan_text = response.content
        
        plan = [step.strip() for step in plan_text.split("\n") if step.strip()]
        
        print(f"\n📋 PLANO GERADO ({len(plan)} passos):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        print("-" * 30)
        
        return {
            "plan": plan, 
            "current_step": plan[0] if plan else None,
            "retry_count": 0, 
            "evidence": [],
            "token_usage": new_usage
        }