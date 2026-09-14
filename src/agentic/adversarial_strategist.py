"""
src/agentic/adversarial_strategist.py
=====================================
Module 4B: Agentic loop for evading the ML detector.
Reads detector weaknesses via tools and proposes parameter mutations.
"""

import json
import logging
from typing import Any

from src.agentic.llm_client import LLMClient
from src.agentic.tools import STRATEGIST_TOOLS, ToolRegistry, PROPOSE_MUTATION_SCHEMA

logger = logging.getLogger(__name__)

STRATEGIST_SYSTEM_PROMPT = """You are an Adversarial Strategist agent. Your goal is to mutate a 
manipulation persona's parameters so that it evades our ML detection model in the next simulation round.
You will be provided with the current round metrics, SHAP drift analysis, and enforcement notes.
You MUST call `propose_mutation` to lock in your final config change based on this provided context.
"""

class AdversarialStrategist:
    def __init__(self, config: dict[str, Any], run_dir=None):
        self.config = config
        self.client = LLMClient(config, run_dir=run_dir)
        self.tools = ToolRegistry(run_dir=run_dir)  # ← real disk reads when run_dir is set
        self.max_tool_calls = config.get("max_tool_calls", 6)


    def _execute_tool_call(self, tool_call: dict) -> dict:
        """Dispatches a tool call to the ToolRegistry."""
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        logger.info(f"[Strategist] Executing tool: {name}({args})")
        
        try:
            if name == "get_round_metrics":
                return self.tools.get_round_metrics(**args)
            elif name == "get_shap_drift":
                return self.tools.get_shap_drift(**args)
            elif name == "get_enforcement_notes":
                return self.tools.get_enforcement_notes(**args)
            elif name == "propose_mutation":
                return self.tools.propose_mutation(**args)
            else:
                return {"error": f"Unknown tool {name}"}
        except Exception as e:
            return {"error": str(e)}

    def run_strategist_round(self, round_num: int, persona: str) -> dict:
        """
        Executes the bounded agentic loop.
        """
        logger.info(f"Starting Adversarial Strategist round {round_num} for persona '{persona}'")
        
        # Pre-fetch context to save API calls
        current_metrics = self.tools.get_round_metrics(round_num)
        shap_drift = self.tools.get_shap_drift(persona=persona, round_a=max(0, round_num-1), round_b=round_num)
        notes = self.tools.get_enforcement_notes(persona)
        
        context = f"""We are on round {round_num}. Please mutate the '{persona}' persona.

Here is the context you need:
1. Current Metrics for Round {round_num}:
{json.dumps(current_metrics, indent=2)}

2. SHAP Drift for '{persona}' (Round {max(0, round_num-1)} vs {round_num}):
{json.dumps(shap_drift, indent=2)}

3. Enforcement Notes for '{persona}':
{notes}

Based on this, propose new parameters using the `propose_mutation` tool."""

        messages = [
            {"role": "user", "content": context}
        ]
        
        for i in range(self.max_tool_calls):
            logger.debug(f"[Strategist] Agent iteration {i+1}/{self.max_tool_calls}")
            
            response = self.client.complete(
                system=STRATEGIST_SYSTEM_PROMPT,
                messages=messages,
                tools=[PROPOSE_MUTATION_SCHEMA]
            )
            
            choice = response["choices"][0]["message"]
            
            # If the model just replied with text and no tool calls, it's either done or confused.
            if "tool_calls" not in choice or not choice["tool_calls"]:
                content = choice.get("content", "")
                messages.append({"role": "assistant", "content": content})
                # Prompt it to call propose_mutation if it hasn't yet
                messages.append({"role": "user", "content": "You must call `propose_mutation` to complete your task. Do not stop until you do."})
                continue
                
            # Process tool calls
            # We must append the assistant's tool call intent back to the messages list
            messages.append(choice)
            
            mutation_proposed = False
            
            for tool_call in choice["tool_calls"]:
                result = self._execute_tool_call(tool_call)
                
                # Append tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "content": json.dumps(result)
                })
                
                if tool_call["function"]["name"] == "propose_mutation":
                    mutation_proposed = True
                    # If it succeeded (no error), we break out of the loop
                    if "error" not in result:
                        logger.info(f"Mutation successfully proposed for {persona}!")
                        return {"status": "success", "rounds": i+1, "result": result}
                    else:
                        # Validation failed. We let the loop continue so the LLM can try again.
                        logger.warning(f"Mutation validation failed: {result}")
                        
        logger.error("[Strategist] Max tool calls reached without a valid mutation. Falling back to mutation_library.py.")
        return {"status": "fallback", "reason": "max_tool_calls_reached"}
