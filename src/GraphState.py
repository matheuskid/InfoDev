import operator
from typing import Annotated, List, TypedDict


class GraphState(TypedDict):
    user_query: str                   # User's original question
    plan: List[str]              # List of steps (e.g., ["Search Law", "Search Hospital Rule"])
    current_step: str            # The specific step currently being processed
    
    # State used to configure the Librarian
    search_category: str         
    failed_categories: List[str] # List of domains to ignore for this step when routing fails
    
    # Evidence accumulation
    documents: List[str]         # Documents found in the current hop
    evidence: Annotated[List[str], operator.add] # Cumulative history of findings
    
    # Control flow
    feedback: str                # "continue", "retry", or "finished"
    retry_count: int             # Safety counter
    final_answer: str            # The output

    token_usage: dict