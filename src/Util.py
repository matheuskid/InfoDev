from GraphState import GraphState

def update_token_usage(state: GraphState, response):
    """
    Extracts token usage from an LLM response and updates the state.
    """
    usage = response.response_metadata.get("token_usage", {})
    
    current_usage = state.get("token_usage", {"input": 0, "output": 0, "total": 0})
    
    new_input = current_usage["input"] + usage.get("prompt_tokens", 0)
    new_output = current_usage["output"] + usage.get("completion_tokens", 0)
    new_total = new_input + new_output
    
    return {"input": new_input, "output": new_output, "total": new_total}