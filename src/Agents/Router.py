from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage
import re

class Router:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        print("🧭 [ROTEADOR] Escolhendo o banco de dados...")
        
        step_query = state["current_step"]
        blacklist = state.get("failed_categories", [])
        
        todos_dominios = {
            "commits": "Source code, code implementation details, commit messages.",
            "issues": "Bug reports, task descriptions, feature requests.",
            "emails": "Developer discussions, mailing lists, community decisions."
        }

        if blacklist:
            print(f"   -> Ignorando domínios (Blacklist): {blacklist}")
            
            for dominio in blacklist:
                todos_dominios.pop(dominio)

            if todos_dominios.__len__ == 0: return Exception
        
        dominios_str = "\n".join([f"'{chave}': {valor}" for chave, valor in todos_dominios.items()])
        chaves_validas = ", ".join([f"'{chave}'" for chave in todos_dominios.keys()])

        system_prompt = f"""You are a classification expert. Classify the user's query into the correct domain.
        
        AVAILABLE DOMAINS:
        {dominios_str}
        
        Return ONLY the exact word of the chosen domain ({chaves_validas}). Do not explain."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"query": step_query})
            new_usage = update_token_usage(state, response)

            category = str(response.content).strip().lower()
            category = re.sub(r'[^a-z]', '', category) 
            
            if category not in todos_dominios.keys():
                print(f"⚠️ Roteador fugiu do escopo ('{category}'). Forçando a primeira opção válida.")
                category = list(todos_dominios.keys())[0] 
                
        except Exception as e:
            print(f"❌ Erro na execução do Roteador: {e}")
            category = list(todos_dominios.keys())[0] if todos_dominios else "issues"
            new_usage = state.get("token_usage", {})

        print(f"✅ Routing Decision: {category}")
        return {"search_category": category, "token_usage": new_usage}