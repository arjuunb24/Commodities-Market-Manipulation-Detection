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

    def run_strategist_round(self, round_num: int, personas: list[str]) -> dict:
        """
        Executes the bounded agentic loop to mutate multiple personas in one session.
        """
        logger.info(f"Starting Adversarial Strategist round {round_num} for personas: {personas}")
        
        # Pre-fetch context to save API calls
        current_metrics = self.tools.get_round_metrics(round_num)
        
        context_parts = [
            f"We are on round {round_num}.",
            f"Please mutate the following personas: {', '.join(personas)}.",
            "",
            "CRITICAL SCHEMA INFORMATION:",
            "You MUST ONLY propose parameters that exist in the following schema. Any other keys will be completely ignored by the simulation engine:",
            " - spoofing:",
            "   * cancellation_delay_ticks (int, default 5, range 1 to 20)",
            "   * order_size_multiplier (float, default 8.0, range 1.0 to 15.0)",
            "   * frequency (float, default 0.4, range 0.05 to 1.0)",
            "   * price_aggressiveness (float, default 0.003, range 0.0001 to 0.01)",
            " - wash_trading:",
            "   * trade_frequency (float, default 0.5, range 0.05 to 1.0)",
            "   * price_deviation_from_mid (float, default 0.001, range 0.0001 to 0.01)",
            "   * n_colluding_pairs (int, default 1, range 1 to 5)",
            " - pump_and_dump:",
            "   * burst_duration_ticks (int, default 200, range 50 to 500)",
            "   * n_coordinated_accounts (int, default 3, range 2 to 10)",
            "   * dump_delay_ticks (int, default 100, range 10 to 300)",
            "   * accumulation_size (int, default 150, range 50 to 500)",
            " - layering:",
            "   * n_layers (int, default 4, range 2 to 8)",
            "   * layer_spacing (float, default 0.002, range 0.0001 to 0.01)",
            "   * cancellation_delay_ticks (int, default 3, range 1 to 20)",
            "",
            "Here is the context you need:",
            f"1. Current Metrics for Round {round_num}:",
            json.dumps(current_metrics, indent=2),
            ""
        ]
        
        for p in personas:
            shap_drift = self.tools.get_shap_drift(persona=p, round_a=max(0, round_num-1), round_b=round_num)
            notes = self.tools.get_enforcement_notes(p)
            context_parts.extend([
                f"--- Context for '{p}' ---",
                f"SHAP Drift (Round {max(0, round_num-1)} vs {round_num}):",
                json.dumps(shap_drift, indent=2),
                f"Enforcement Notes:",
                notes,
                ""
            ])
            
        context_parts.append(
            "Based on this, propose new parameters using the `propose_mutation` tool. "
            "You MUST call `propose_mutation` multiple times (once for EACH persona listed above) in a single response to complete your task."
        )
        
        context = "\n".join(context_parts)

        messages = [
            {"role": "user", "content": context}
        ]
        
        mutated_personas = set()
        
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
                
                missing = set(personas) - mutated_personas
                messages.append({"role": "user", "content": f"You must call `propose_mutation` for the remaining personas: {missing}. Do not stop until you do."})
                continue
                
            # Process tool calls
            # We must append the assistant's tool call intent back to the messages list
            messages.append(choice)
            
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
                    if "error" not in result:
                        args = json.loads(tool_call["function"]["arguments"])
                        mutated_personas.add(args.get("persona"))
                        logger.info(f"Mutation successfully proposed for {args.get('persona')}!")
                    else:
                        logger.warning(f"Mutation validation failed: {result}")
                        
            if mutated_personas.issuperset(set(personas)):
                logger.info("All requested personas successfully mutated!")
                return {"status": "success", "rounds": i+1}
                        
        if mutated_personas:
            logger.warning(f"[Strategist] Max tool calls reached. Mutated: {mutated_personas}, Missing: {set(personas) - mutated_personas}")
            return {"status": "partial_success", "mutated": list(mutated_personas)}

        logger.error("[Strategist] Max tool calls reached without a valid mutation. Falling back to mutation_library.py.")
        return {"status": "fallback", "reason": "max_tool_calls_reached"}
