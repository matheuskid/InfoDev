from langchain.prompts import ChatPromptTemplate
from Util import update_token_usage

class StepDefiner:
    """
    Agent responsible for analyzing current progress, checking for redundancies
    in memory, and refining the next strategic step.
    """
    def __init__(self, llm):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_template(
            "Given the original plan, the step we just finished, and the results (memory), "
            "your goal is to refine the **Immediate Next Step**.\n\n"
            
            "INPUTS:\n"
            "Plan (Future Steps): {future_steps}\n"
            "Recently Finished Step: {current_step}\n"
            "Memory (Results): {memory}\n\n"
            
            "INSTRUCTIONS:\n"
            "1. If the 'Memory' already contains the complete answer for the first step of the 'Future Steps', reply ONLY with 'DONE'.\n"
            "2. If the step is still necessary, rewrite it concisely and briefly.\n"
            "3. Reply ONLY with the refined query string OR 'DONE' if the answer is already in memory. Do not explain.\n\n"
            
            "Next Step:"
        )
        self.chain = self.prompt | self.llm

    def run(self, state):
        print("--- Agent: Step Definer (Refining Next Step) ---")
        
        plan = state.get("plan", [])
        current_step = state.get("current_step")
        evidence = state.get("evidence", [])
        
        try:
            current_idx = plan.index(current_step)
            future_steps = plan[current_idx+1:]
        except ValueError:
            return {"feedback": "finished"}

        if not future_steps:
            return {"feedback": "finished"}
        
        memory_context = "\n".join(evidence)
        
        response = self.chain.invoke({
            "future_steps": str(future_steps),
            "current_step": current_step,
            "memory": memory_context
        })

        new_usage = update_token_usage(state, response)
        refined_next_step = response.content.strip()
        
        print(f"➡️ Refined next step: '{refined_next_step}'")

        
        if "DONE" in refined_next_step.upper():
            print("✅ Answer already in memory or plan is complete.")
            return {
                "feedback": "finished", 
                "token_usage": new_usage
            }
        
        else:
            future_steps.pop(0) 
            
            new_future_steps = [refined_next_step] + future_steps
            
            done_steps = plan[:current_idx+1]
            updated_full_plan = done_steps + new_future_steps
            
            return {
                "plan": updated_full_plan,  
                "current_step": refined_next_step, # Set as the new target
                "feedback": "continue",
                
                "failed_categories": [], 
                "retry_count": 0,        
                
                "token_usage": new_usage
            }