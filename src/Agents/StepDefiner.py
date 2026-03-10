from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class StepDefiner:
    """
    Agente responsável por analisar o progresso atual, verificar redundâncias
    na memória e refinar a consulta do próximo passo estratégico.
    """
    def __init__(self, llm):
        self.llm = llm
        
        # Prompt fixo e compilado uma única vez
        self.prompt = ChatPromptTemplate.from_template(
            "Dado o plano original, o passo que acabamos de finalizar e os resultados (memória), "
            "seu objetivo é refinar o **Próximo Passo Imediato**.\n\n"
            
            "ENTRADAS:\n"
            "Plano (Passos Futuros): {future_steps}\n"
            "Passo Recém-Finalizado: {current_step}\n"
            "Memória (Resultados): {memory}\n\n"
            
            "INSTRUÇÕES:\n"
            "1. **Verificar Redundância:** Se a 'Memória' já contém a resposta completa para o primeiro passo dos 'Passos Futuros', responda apenas 'FINALIZADO'.\n"
            "2. **Refinar Consulta:** Se o passo ainda for necessário, reescreva-o de forma resumida e concisa.\n"
            "3. **Concisão:** Responda APENAS com a string da consulta refinada ou 'FINALIZADO' caso a resposta já esteja na memória. Não escreva 'Tipo de Tarefa' nem explicações.\n\n"
            
            "Próximo Passo:"
        )
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agente: Step Definer (Refinando Próximo Passo) ---")
        
        plan = state.get("plan", [])
        current_step = state.get("current_step")
        evidence = state.get("evidence", [])
        
        # Calcular o que falta fazer
        try:
            current_idx = plan.index(current_step)
            future_steps = plan[current_idx+1:]
        except ValueError:
            return {"feedback": "finished"}

        if not future_steps:
            return {"feedback": "finished"}
        
        # Agrupar a memória em um bloco de texto
        memory_context = "\n".join(evidence)
        
        # Invocar o LLM
        response = self.chain.invoke({
            "future_steps": str(future_steps),
            "current_step": current_step,
            "memory": memory_context
        })

        new_usage = update_token_usage(state, response)
        refined_next_step = response.content.strip()
        
        print(f"Próximo passo refinado: '{refined_next_step}'")

        # --- Lógica de Roteamento de Saída ---
        if "FINALIZADO" in refined_next_step.upper():
            print("✅ O LLM percebeu que a resposta já está na memória.")
            return {"feedback": "finished", "token_usage": new_usage}
        else:
            # Substituir o passo antigo pelo passo detalhado
            future_steps.pop(0) 
            new_future_steps = [refined_next_step] + future_steps
            
            # Reconstruir o plano completo
            done_steps = plan[:current_idx+1]
            updated_full_plan = done_steps + new_future_steps
            
            return {
                "plan": updated_full_plan,  
                "current_step": refined_next_step, # Define como novo alvo
                "feedback": "continue",
                "token_usage": new_usage
            }