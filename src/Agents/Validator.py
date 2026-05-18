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
            "You are a strict auditor. Analyze whether the evidence below answers "
            "the current step of the plan.\n"
            "Current Step: \"{current_step}\"\n"
            "Evidence: \"{last_evidence}\"\n\n"
            "Rule: Reply ONLY with the word 'APPROVED' if it is useful, "
            "or 'REJECTED' if it is useless or empty. Do not explain anything."
        )
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agente: Fact Checker (Validação com Blacklist) ---")
        
        current_step = state.get("current_step", "")
        evidence_list = state.get("evidence", [])
        last_evidence = evidence_list[-1] if evidence_list else ""
        current_category = state.get("search_category")
        
        response = self.chain.invoke({
            "current_step": current_step, 
            "last_evidence": last_evidence
        })

        new_usage = update_token_usage(state, response)
        
        decision = str(response.content).strip().upper()
        print(f"⚖️ Decisão do Auditor: {decision}")

        is_valid = "APPROVED" in decision

        if not is_valid:
            print(f"❌ Falha detectada no domínio: '{current_category}'")
            
            blacklist = state.get("failed_categories", [])
            if current_category not in blacklist:
                blacklist.append(current_category)

            retry_count = state.get("retry_count", 0)

            if retry_count < 2:
                print(f"🔄 Tentativa {retry_count + 1} falhou. Retentando em outro domínio...")
                return {
                    "feedback": "retry", 
                    "retry_count": retry_count + 1,
                    "failed_categories": blacklist,
                    "token_usage": new_usage
                }

            else:
                print("⚠️ Máximo de tentativas. Pulando este passo...")
                return {
                    "feedback": "APPROVED", # Força a ir para o StepDefiner avançar o plano
                    "token_usage": new_usage
                }

        print("✅ Evidência APROVADA com sucesso!")
        return {
            "feedback": "APPROVED",
            "token_usage": new_usage
        }
    