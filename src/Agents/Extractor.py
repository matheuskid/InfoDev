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
            "Analyze the retrieved documents to specifically answer the current step.\n"
            "Current Step: \"{step}\"\n\n"
            "Documents:\n---\n{documents}\n---\n"
            "Summarize the found information (if any) in a concise manner."
        )
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agente: Coletor (Resumindo Evidências) ---")
        
        current_step = state.get("current_step", "")
        documents = state.get("documents", [])
        
        if not documents:
            print("⚠️ Nenhum documento recebido. Pulando extração.")
            return {"evidence": [f"Para o passo '{current_step}', nada foi encontrado."]}

        docs_text = "\n---\n".join(documents)
        
        response = self.chain.invoke({
            "step": current_step, 
            "documents": docs_text
        })

        new_usage = update_token_usage(state, response)
        new_evidence = response.content.strip()
        
        new_evidence_entry = f"Passo: {current_step} -> {new_evidence}"
        
        print("✅ Evidência extraída com sucesso.")
        
        return {
            "evidence": [new_evidence_entry],
            "token_usage": new_usage
        }