"""
src/agentic/tools.py
====================
Shared tool definitions and schemas for the LLM agents (Module 4B, 4C).
"""

from typing import Any

# =============================================================================
# SCHEMAS (OpenAI / LiteLLM format)
# =============================================================================

GET_ROUND_METRICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_round_metrics",
        "description": "Returns precision, recall, and F1 score per persona for a given round.",
        "parameters": {
            "type": "object",
            "properties": {
                "round_num": {
                    "type": "integer",
                    "description": "The simulation round number (e.g., 0 for baseline)"
                }
            },
            "required": ["round_num"]
        }
    }
}

GET_SHAP_DRIFT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_shap_drift",
        "description": "Returns which features gained or lost importance for a given persona between two rounds.",
        "parameters": {
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": "The manipulation persona (e.g., spoofing, wash_trading)"
                },
                "round_a": {
                    "type": "integer",
                    "description": "The baseline round number"
                },
                "round_b": {
                    "type": "integer",
                    "description": "The subsequent round number to compare against"
                }
            },
            "required": ["persona", "round_a", "round_b"]
        }
    }
}

GET_ENFORCEMENT_NOTES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_enforcement_notes",
        "description": "Returns the qualitative CFTC enforcement case-note grounding text for a persona type.",
        "parameters": {
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": "The manipulation persona to look up"
                }
            },
            "required": ["persona"]
        }
    }
}

PROPOSE_MUTATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_mutation",
        "description": "Proposes a parameter dict for the next round's persona mutation. THIS IS THE ONLY TOOL THAT PRODUCES A SIDE EFFECT; it writes to persona_config.yaml.",
        "parameters": {
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": "The persona to mutate"
                },
                "parameters": {
                    "type": "object",
                    "description": "The mutated parameter dictionary (e.g., {'cancellation_delay_ticks': 10})"
                }
            },
            "required": ["persona", "parameters"]
        }
    }
}

# Combine for 4B Strategist
STRATEGIST_TOOLS = [
    GET_ROUND_METRICS_SCHEMA,
    GET_SHAP_DRIFT_SCHEMA,
    GET_ENFORCEMENT_NOTES_SCHEMA,
    PROPOSE_MUTATION_SCHEMA
]

# =============================================================================
# IMPLEMENTATIONS (Stubs / Accessors)
# =============================================================================

class ToolRegistry:
    """Provides the actual Python implementations for the tools."""
    
    def __init__(self, run_dir=None):
        self.run_dir = run_dir
        
    def get_round_metrics(self, round_num: int) -> dict:
        # In a real run, this would read from data/runs/round_{round_num}/metrics.json
        # For now, return a mock response for testing
        return {
            "spoofing": {"precision": 0.90, "recall": 0.86, "f1": 0.88},
            "wash_trading": {"precision": 0.85, "recall": 0.80, "f1": 0.82}
        }
        
    def get_shap_drift(self, persona: str, round_a: int, round_b: int) -> dict:
        return {
            "persona": persona,
            "drift": {
                "cancel_rate": "-0.15 (decreased importance)",
                "volume_concentration": "+0.08 (increased importance)"
            }
        }
        
    def get_enforcement_notes(self, persona: str) -> str:
        # Read from docs/enforcement_case_notes.md
        # Returning a stub for prompt grounding
        if "spoof" in persona.lower():
            return "CFTC vs Navinder Sarao: Spoofer placed large passive orders away from best bid/ask to create false impression of supply, then cancelled before execution. Strategy: high cancel rate, large out-of-the-money orders."
        return f"Enforcement notes for {persona}: Traders coordinated to falsely inflate volume."

    def propose_mutation(self, persona: str, parameters: dict) -> dict:
        """
        Validates the proposed parameters and writes them to configs/persona_config.yaml.
        """
        # 1. Validate
        if "cancellation_delay_ticks" in parameters:
            if not isinstance(parameters["cancellation_delay_ticks"], int) or parameters["cancellation_delay_ticks"] <= 0:
                return {"error": "cancellation_delay_ticks must be a positive integer."}
                
        # 2. Write side-effect (mock logic)
        import yaml
        from pathlib import Path
        
        config_path = Path("configs/persona_config.yaml")
        existing = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                existing = yaml.safe_load(f) or {}
                
        if persona not in existing:
            existing[persona] = {}
            
        existing[persona].update(parameters)
        
        with open(config_path, "w") as f:
            yaml.dump(existing, f)
            
        return {"status": "success", "message": f"Successfully mutated {persona} config."}
