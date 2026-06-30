# src/Graph.py
from langgraph.graph import StateGraph, START, END
from GraphState import GraphState
from VectorStoreManager import VectorStoreManager

from Agents.Planner import Planner
from Agents.Router import Router
from Agents.Librarian import Librarian 
from Agents.Extractor import Extractor
from Agents.Validator import Validator
from Agents.StepDefiner import StepDefiner
from Agents.Editor import Editor

def route_validator(state: GraphState):
    """Lê a nota do Auditor e decide para onde ir."""
    feedback = state.get("feedback", "").upper()
    
    if "APPROVED" in feedback:
        return "step_definer"
    
    if state.get("retry_count", 0) >= 3:
        print("⚠️ Limite de retentativas atingido. Avançando mesmo assim...")
        return "step_definer"
        
    return "router"

def route_step_definer(state):
    """Lê a decisão do Step Definer e aponta o próximo agente."""
    feedback = state.get("feedback", "")
    
    if feedback == "finished":
        return "editor"
    else:
        return "router"

def build_graph(llm_geral, llm_extractor=None):
    if llm_extractor is None:
        llm_extractor = llm_geral
    
    print("⚙️ Montando a Arquitetura do InfoDev...")
    
    db_path = "./vectorstores/tez_db"
    vsm_commits = VectorStoreManager(db_path, "commits", "jinaai/jina-embeddings-v2-base-code")
    vsm_issues = VectorStoreManager(db_path, "issues", "nomic-ai/nomic-embed-text-v1.5")
    vsm_emails = VectorStoreManager(db_path, "emails", "nomic-ai/nomic-embed-text-v1.5")
    
    dict_vsm = {
        "commits": vsm_commits,
        "issues": vsm_issues,
        "emails": vsm_emails
    }

    planner_agent = Planner(llm_geral)
    router_agent = Router(llm_geral)
    librarian_agent = Librarian(dict_vsm) 
    extractor_agent = Extractor(llm_extractor)
    validator_agent = Validator(llm_geral)
    step_definer_agent = StepDefiner(llm_geral)
    editor_agent = Editor(llm_geral)

    workflow = StateGraph(GraphState)
    
    workflow.add_node("planner", planner_agent.run)
    workflow.add_node("router", router_agent.run)
    workflow.add_node("librarian", librarian_agent.run)
    workflow.add_node("extractor", extractor_agent.run)
    workflow.add_node("validator", validator_agent.run)
    workflow.add_node("step_definer", step_definer_agent.run)
    workflow.add_node("editor", editor_agent.run)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "router")
    workflow.add_edge("router", "librarian")
    workflow.add_edge("librarian", "extractor")
    workflow.add_edge("extractor", "validator")

    workflow.add_conditional_edges(
        "validator",
        route_validator, 
        {
            "step_definer": "step_definer",
            "router": "router" 
        }
    )

    workflow.add_conditional_edges(
        "step_definer",
        route_step_definer, 
        {
            "editor": "editor",
            "router": "router" 
        }
    )

    workflow.add_edge("editor", END)

    return workflow.compile()