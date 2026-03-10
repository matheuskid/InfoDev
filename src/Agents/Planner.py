from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class Planner:
    """
    Agente responsável por desconstruir a pergunta do utilizador num plano
    de consultas de pesquisa simples e eficientes.
    """
    def __init__(self, llm):
        # 1. Recebe o LLM instanciado do App.py
        self.llm = llm
        
        # 2. Define o prompt fixo no momento da criação do Agente
        self.prompt = ChatPromptTemplate.from_template(
            "Você é um planejador eficiente. Seu objetivo é desconstruir a pergunta do usuário em consultas de busca simples e eficientes.\n"
            "Para cada pergunta, determine se a pergunta é composta ou simples.\n "
            "**Se composta:** divida-a em múltiplas consultas de busca focadas em tópicos específicos.\n"
            "Crie consultas de busca claras e diretas para cada tópico identificado.\n"
            "Não crie consultas redundantes. Não crie passos que não sejam consultas. \n"
            "Siga estritamente ao que foi perguntado. Não invente consultas extras se não forem pedidas. Não crie consultas que realizem buscas na internet.\n"
            "**Se simples:** crie apenas uma consulta que responda diretamente à pergunta.\n"
            "Pergunta: {query}\n\n"
            "Retorne APENAS a lista de consultas separadas por nova linha."
        )
        
        # 3. Monta a chain uma única vez
        self.chain = self.prompt | self.llm

    def run(self, state):
        """
        Método de execução (Nó do LangGraph). 
        Recebe o GraphState, executa a lógica e devolve as atualizações.
        """
        print("--- Agente: Líder (Planeamento) ---")
        
        # Verifica se já existe um plano (evita re-planeamento em ciclos de execução)
        if state.get("plan"):
            print("✅ Plano já existente. A avançar...")
            return {}
        
        print("A gerar plano estratégico...")
        
        # 4. Invocação usando a chain pré-compilada
        response = self.chain.invoke({"query": state["query"]})
        
        # 5. Atualização de métricas e extração de dados
        new_usage = update_token_usage(state, response)
        plan_text = response.content
        
        # Parse do plano ignorando linhas vazias
        plan = [step.strip() for step in plan_text.split("\n") if step.strip()]
        
        # Feedback visual no terminal
        print(f"\n📋 PLANO GERADO ({len(plan)} passos):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        print("-" * 30)
        
        # 6. Retorna apenas os campos do GraphState que este agente deve modificar
        return {
            "plan": plan, 
            "current_step": plan[0] if plan else None, # Proteção caso o LLM retorne vazio
            "retry_count": 0, 
            "evidence": [],
            "token_usage": new_usage
        }