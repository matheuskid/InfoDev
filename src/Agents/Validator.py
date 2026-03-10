from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class Validator:
    """
    Agente auditor (Fact Checker). Verifica se a evidência recém-coletada 
    realmente responde ao passo atual. Gerencia tentativas e falhas (Blacklist).
    """
    def __init__(self, llm):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_template(
            "Você é um auditor rigoroso. Analise se as evidências abaixo respondem "
            "ao passo atual do plano.\n"
            "Passo Atual: \"{current_step}\"\n"
            "Evidências: \"{last_evidence}\"\n\n"
            "Regra: Responda APENAS com a palavra 'APROVADO' se for útil, "
            "ou 'REJEITADO' se for inútil ou vazio. Não explique nada."
        )
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agente: Fact Checker (Validação com Blacklist) ---")
        
        current_step = state.get("current_step", "")
        evidence_list = state.get("evidence", [])
        last_evidence = evidence_list[-1] if evidence_list else ""
        current_category = state.get("search_category", "general")
        
        # 1. Invocação da Validação
        response = self.chain.invoke({
            "current_step": current_step, 
            "last_evidence": last_evidence
        })

        new_usage = update_token_usage(state, response)
        decision = response.content.strip().upper()
        print(f"⚖️ Decisão do Auditor: {decision}")

        is_valid = "APROVADO" in decision

        # --- FLUXO REJEITADO ---
        if not is_valid:
            print(f"❌ Falha detectada no domínio: '{current_category}'")
            
            blacklist = state.get("failed_categories", [])
            if current_category not in blacklist:
                blacklist.append(current_category)

            retry_count = state.get("retry_count", 0)

            # Tenta novamente em outro domínio
            if retry_count < 2:
                print(f"🔄 Tentativa {retry_count + 1} falhou. Retentando em outro domínio...")
                return {
                    "feedback": "retry", 
                    "retry_count": retry_count + 1,
                    "failed_categories": blacklist, # Salva a blacklist para o Router ler
                    "token_usage": new_usage
                }
            # Desiste do passo após limite de tentativas
            else:
                print("⚠️ Máximo de tentativas. Pulando este passo.")
                failed_note = f"Passo '{current_step}' FALHOU. Tentativas esgotadas nos domínios: {blacklist}."
                
                return {
                    "evidence": [failed_note], 
                    "feedback": "continue", 
                    "retry_count": 0,
                    "failed_categories": [], # Limpa a blacklist para o próximo passo ter chance limpa
                    "token_usage": new_usage
                }

        # --- FLUXO APROVADO ---
        plan = state.get("plan", [])
        try:
            current_index = plan.index(current_step)
            
            # Se ainda tem passos no plano
            if current_index < len(plan) - 1:
                next_step = plan[current_index + 1]
                print(f"➡️ Avançando para o próximo passo -> {next_step}")
                return {
                    "current_step": next_step, 
                    "feedback": "continue", 
                    "retry_count": 0,
                    "failed_categories": [], # Limpa a blacklist para o novo passo
                    "token_usage": new_usage
                }
            # Se terminou o plano inteiro
            else:
                print("🏁 Plano totalmente concluído.")
                return {
                    "feedback": "finished",
                    "failed_categories": [], 
                    "token_usage": new_usage
                }
                
        except ValueError:
            # Caso de segurança (fallback)
            return {"feedback": "finished", "token_usage": new_usage}