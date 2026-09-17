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

STRATEGIST_SYSTEM_PROMPT = """You are an Adversarial Strategist agent playing a high-stakes cat-and-mouse game against an ML detection model.
Your goal is to mutate a manipulation persona's parameters so that it evades our ML detection model in the next simulation round.
You must make the manipulations tougher and harder to detect at every round by learning from past rounds' metrics and failed parameters.
You will be provided with the current round metrics, historical parameters that failed, SHAP drift analysis, and CFTC enforcement notes.
Draw inspiration from real CFTC enforcement case notes to find new angles of attack. Do NOT repeat past failed parameters.
You MUST call `propose_mutation` to lock in your final config change based on this provided context, and provide detailed step-by-step reasoning.

CRITICAL FORMATTING INSTRUCTION: In the `reasoning` field of the `propose_mutation` tool, you MUST format your reasoning as a clean, easily readable numbered list where each number explicitly justifies a specific parameter change.
Example format:
"In Round [X], [Persona] was detected due to [Reason]. To evade detection:
1. Increased/Decreased `parameter_name` from [Old] to [New]: [Explanation for why this evades detection].
2. Increased/Decreased `another_parameter` from [Old] to [New]: [Explanation...]"
Do NOT output a single massive paragraph. Use newlines to separate the numbered items.
"""

class AdversarialStrategist:
    def __init__(self, config: dict[str, Any], run_dir=None):
        self.config = config
        self.client = LLMClient(config, run_dir=run_dir)
        self.tools = ToolRegistry(run_dir=run_dir)  # ← real disk reads when run_dir is set
        self.max_tool_calls = config.get("max_tool_calls", 15)


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
        
        # Pre-fetch historical parameters
        import yaml
        from pathlib import Path
        config_path = Path("configs/persona_config.yaml")
        historical_params = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                p_cfg = yaml.safe_load(f) or {}
                for k, v in p_cfg.items():
                    if k.startswith("round_"):
                        historical_params[k] = v
        
        context_parts = [
            f"We are on round {round_num}.",
            f"Please mutate the following personas: {', '.join(personas)}.",
            "",
            "CRITICAL SCHEMA INFORMATION AND REAL-WORLD CONSTRAINTS:",
            "You MUST ONLY propose parameters that exist in the following schema. Mutations must be realistic. Over-mutating will make your trades unprofitable or obvious.",
            " - spoofing:",
            "   * cancellation_delay_ticks (int, range 1 to 20, default 5). Too low: order won't affect market. Too high: risk of execution.",
            "   * order_size_multiplier (float, range 1.0 to 15.0, default 8.0). Too low: won't scare others. Too high: blatantly obvious to regulators.",
            "   * frequency (float, range 0.05 to 1.0, default 0.4). High frequency flags anomaly detection.",
            "   * price_aggressiveness (float, range 0.0001 to 0.01, default 0.003).",
            " - wash_trading:",
            "   * trade_frequency (float, range 0.05 to 1.0, default 0.5). Highly frequent fixed intervals look like bots. Low frequency won't inflate volume.",
            "   * price_deviation_from_mid (float, range 0.0001 to 0.01, default 0.001).",
            "   * n_colluding_pairs (int, range 1 to 5, default 1).",
            " - pump_and_dump:",
            "   * burst_duration_ticks (int, range 50 to 1000, default 200). A slow pump (e.g. 800) evades short-term volume detectors but requires more capital.",
            "   * n_coordinated_accounts (int, range 2 to 15, default 3).",
            "   * dump_delay_ticks (int, range 10 to 500, default 100).",
            "   * accumulation_size (int, range 50 to 1000, default 150). Must be large enough to secure profit.",
            " - layering:",
            "   * n_layers (int, range 2 to 8, default 4).",
            "   * layer_spacing (float, range 0.0001 to 0.01, default 0.002).",
            "   * cancellation_delay_ticks (int, range 1 to 20, default 3).",
            "",
            "Here is the context you need:",
            f"1. Current Metrics for Round {round_num}:",
            json.dumps(current_metrics, indent=2),
            "",
            f"2. Historical Parameters Used in Past Rounds (DO NOT REPEAT FAILED CONFIGURATIONS):",
            json.dumps(historical_params, indent=2),
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
        reasoning_log = {}
        
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
                        persona_name = args.get("persona")
                        mutated_personas.add(persona_name)
                        logger.info(f"Mutation successfully proposed for {persona_name}!")
                        
                        # Print the parameter diff to the console
                        old_p = result.get("old_params", {})
                        new_p = result.get("new_params", {})
                        reasoning = result.get("reasoning", "No reasoning provided.")
                        reasoning_log[persona_name] = reasoning
                        
                        print(f"\n[{persona_name.upper()}] Adversarial Strategist Mutated Parameters:")
                        for k, v in new_p.items():
                            old_val = old_p.get(k, 'N/A')
                            if old_val != v:
                                print(f"  * {k}: {old_val} -> {v}")
                        print("")
                    else:
                        logger.warning(f"Mutation validation failed: {result}")
                        
            if mutated_personas.issuperset(set(personas)):
                logger.info("All requested personas successfully mutated!")
                return {"status": "success", "rounds": i+1, "reasoning": reasoning_log}
                        
        if mutated_personas:
            logger.warning(f"[Strategist] Max tool calls reached. Mutated: {mutated_personas}, Missing: {set(personas) - mutated_personas}")
            return {"status": "partial_success", "mutated": list(mutated_personas), "reasoning": reasoning_log}

        logger.error("[Strategist] Max tool calls reached without a valid mutation. Falling back to mutation_library.py.")
        return {"status": "fallback", "reason": "max_tool_calls_reached", "reasoning": reasoning_log}
