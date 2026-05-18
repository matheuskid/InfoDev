from GraphState import GraphState

def update_token_usage(state, response):
    usage = response.response_metadata.get("token_usage", {})
    
    current_usage = state.get("token_usage", {})
    
    new_input = current_usage.get("input", 0) + usage.get("prompt_tokens", 0)
    new_output = current_usage.get("output", 0) + usage.get("completion_tokens", 0)
    new_total = new_input + new_output
    
    return {
        "input": new_input,
        "output": new_output,
        "total": new_total
    }