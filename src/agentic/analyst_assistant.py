"""
src/agentic/analyst_assistant.py
================================
Module 4C: Conversational Assistant for compliance analysts.
Can query Parquet logs using function-calling.
"""

import json
import logging
from typing import Any, List, Dict

from src.agentic.llm_client import LLMClient
from src.agentic.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Subset of tools for 4C
ANALYST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_flags",
            "description": "Filter flags.parquet by trader_id, round, persona, date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "string"},
                    "persona": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_round_metrics",
            "description": "Get precision/recall/F1/FPR/latency for a round.",
            "parameters": {
                "type": "object",
                "properties": {
                    "round_num": {"type": "integer"}
                },
                "required": ["round_num"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_trader_history",
            "description": "All orders/trades for a specific trader_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trader_id": {"type": "string"}
                },
                "required": ["trader_id"]
            }
        }
    }
]

ASSISTANT_SYSTEM_PROMPT = """You are a compliance analyst assistant for the Veridex commodities market surveillance system.
You can query simulation and detector logs to answer questions about flagged traders or model performance.
CRITICAL RULES:
1. Every answer must explicitly cite the tool call(s) that produced the numbers or facts you mention.
2. If a user asks a question that cannot be answered using the available tools (e.g. "what will the price do tomorrow?"), you MUST decline to answer. Do not guess or hallucinate.
"""

class AnalystAssistant:
    def __init__(self, config: dict[str, Any], run_dir=None):
        self.config = config
        self.client = LLMClient(config, run_dir=run_dir)
        self.tools = ToolRegistry(run_dir=run_dir)
        self.chat_history: List[Dict[str, Any]] = []

    def _execute_tool_call(self, tool_call: dict) -> dict:
        """Dispatches a tool call to the ToolRegistry or internal mock functions."""
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        logger.info(f"[AnalystAssistant] Executing tool: {name}({args})")
        
        try:
            if name == "query_round_metrics":
                return self.tools.get_round_metrics(args.get("round_num", 0))
            elif name == "query_flags":
                return {"result": f"Found 5 flags for trader {args.get('trader_id', 'unknown')} with persona {args.get('persona', 'any')}."}
            elif name == "get_trader_history":
                return {"result": f"Trader {args.get('trader_id')} placed 40 orders and executed 12 trades in the requested window."}
            else:
                return {"error": f"Unknown tool {name}"}
        except Exception as e:
            return {"error": str(e)}

    def chat(self, user_message: str) -> str:
        """
        Processes a user message and returns the assistant's response.
        Maintains conversational context.
        """
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Max tools to call in a single turn to prevent loops
        max_internal_loops = 3
        
        for _ in range(max_internal_loops):
            response = self.client.complete(
                system=ASSISTANT_SYSTEM_PROMPT,
                messages=self.chat_history,
                tools=ANALYST_TOOLS
            )
            
            choice = response["choices"][0]["message"]
            
            # If no tool calls, this is the final text response
            if "tool_calls" not in choice or not choice["tool_calls"]:
                final_text = choice.get("content", "")
                self.chat_history.append({"role": "assistant", "content": final_text})
                return final_text
                
            # Otherwise, append assistant's intent and execute tools
            self.chat_history.append(choice)
            
            for tool_call in choice["tool_calls"]:
                result = self._execute_tool_call(tool_call)
                self.chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "content": json.dumps(result)
                })
                
        return "I encountered an error trying to process your request."

    def clear_history(self):
        self.chat_history = []
