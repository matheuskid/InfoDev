import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

sys.path.append('./src')

from Graph import build_graph 

st.set_page_config(page_title="InfoDev - RAG Assistant", page_icon="🧠")
st.title("🧠 InfoDev")
st.caption("O seu assistente de Engenharia de Software Multi-Agente")

if "llm" not in st.session_state:
    load_dotenv(override=True)
    st.session_state.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

if "graph" not in st.session_state:
    with st.spinner("Inicializando agentes e conectando aos repositórios..."):
        st.session_state.graph = build_graph(st.session_state.llm)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Pergunte sobre commits, issues ou arquitetura do código..."):
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        
        final_answer = ""
        
        with st.status("Agentes do InfoDev investigando repositórios...", expanded=True) as status:
            
            initial_state = {
                "query": user_query,
                "plan": [],
                "current_step": "",
                "search_category": "",
                "failed_categories": [],
                "documents": [],
                "evidence": [],
                "feedback": "",
                "retry_count": 0,
                "final_answer": "",
                "token_usage": {} 
            }
            
            tokens = {}

            try:

                for event in st.session_state.graph.stream(initial_state):
                    for node_name, state_updates in event.items():

                        if "token_usage" in state_updates:
                            tokens = state_updates["token_usage"]
                        
                        if node_name == "planner":
                            st.write(f"📝 **Planejador:** Criou um plano com {len(state_updates.get('plan', []))} passos.")
                        
                        elif node_name == "router":
                            banco = state_updates.get("search_category", "desconhecido")
                            st.write(f"🧭 **Roteador:** Direcionou a busca para a base de `{banco}`.")
                            
                        elif node_name == "librarian":
                            docs = state_updates.get("documents", [])
                            st.write(f"📚 **Bibliotecário:** Encontrou {len(docs)} fragmentos relevantes.")
                            
                        elif node_name == "extractor":
                            st.write("✂️ **Extrator:** Leu os documentos e extraiu as evidências.")
                            
                        elif node_name == "validator":
                            nota = state_updates.get("feedback", "")
                            icone = "✅" if "APPROVED" in nota else "❌"
                            st.write(f"⚖️ **Auditor:** Avaliou a evidência como {icone} `{nota}`.")
                            
                        elif node_name == "step_definer":
                            prox = state_updates.get("current_step", "")
                            st.write(f"🔄 **Step Definer:** Atualizou a rota (Próximo: {prox}).")
                            
                        elif node_name == "editor":
                            st.write("✨ **Gerador:** Sintetizou a resposta final.")
                            final_answer = state_updates.get("final_answer", "")
                
                status.update(label="Investigação concluída!", state="complete", expanded=False)

                if final_answer:
                    st.markdown(final_answer)

            except Exception as e:
                st.error(f"Erro na execução: {e}")

            finally:
                print("\n" + "="*40)
                print("📊 RELATÓRIO FINAL DE TOKENS")
                print("="*40)
                
                if tokens:
                    prompt_tokens = tokens.get("input", 0)
                    completion_tokens = tokens.get("output", 0)
                    total_tokens = tokens.get("total", 0)
                    
                    st.sidebar.markdown("### 📊 Uso de Tokens")
                    st.sidebar.write(f"**Prompt:** {prompt_tokens}")
                    st.sidebar.write(f"**Completion:** {completion_tokens}")
                    st.sidebar.write(f"**Total:** {total_tokens}")
                else:
                    print("Nenhum token foi consumido ou registrado.")
                
                print("="*40 + "\n")
                
                st.session_state.messages.append({"role": "assistant", "content": final_answer})