from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class Extractor:
    """
    Agente responsável por ler os documentos brutos recuperados pelo Bibliotecário
    e extrair/resumir a informação que responde especificamente ao passo atual.
    """
    def __init__(self, llm):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_template(
            "Analise os documentos recuperados para responder especificamente ao passo atual.\n"
            "Passo Atual: \"{step}\"\n\n"
            "Documentos:\n---\n{documents}\n---\n"
            "Resuma a informação encontrada (se houver) de forma concisa."
        )
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agente: Coletor (Resumindo Evidências) ---")
        
        current_step = state.get("current_step", "")
        documents = state.get("documents", [])
        
        # Proteção: Se o bibliotecário não trouxe nada, não gastamos tokens do LLM
        if not documents:
            print("⚠️ Nenhum documento recebido. Pulando extração.")
            return {"evidence": [f"Para o passo '{current_step}', nada foi encontrado."]}

        # Junta os documentos em uma única string
        docs_text = "\n---\n".join(documents)
        
        # Invoca o LLM
        response = self.chain.invoke({
            "step": current_step, 
            "documents": docs_text
        })

        new_usage = update_token_usage(state, response)
        new_evidence = response.content.strip()
        
        # Formata a evidência para o histórico
        new_evidence_entry = f"Passo: {current_step} -> {new_evidence}"
        
        print("✅ Evidência extraída com sucesso.")
        
        return {
            "evidence": [new_evidence_entry], # O reducer (operator.add) vai anexar isso à lista global
            "token_usage": new_usage
        }