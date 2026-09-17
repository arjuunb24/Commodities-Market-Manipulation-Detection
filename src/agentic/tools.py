"""
src/agentic/tools.py
====================
Shared tool definitions and schemas for the LLM agents (Module 4B, 4C).
"""

from typing import Any
from pathlib import Path

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
                },
                "reasoning": {
                    "type": "string",
                    "description": "Provide a detailed, step-by-step chain of thought explaining WHY you chose these specific parameters. Reference CFTC case notes and previous round metrics to justify how this mutation will make it harder for the detector."
                }
            },
            "required": ["persona", "parameters", "reasoning"]
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
        self.run_dir = Path(run_dir) if run_dir else None

    def get_round_metrics(self, round_num: int) -> dict:
        """
        Reads metrics.json from the round directory written by RoundOrchestrator.
        Falls back to mock data if the file doesn't exist (e.g., during testing).
        """
        if self.run_dir:
            metrics_path = self.run_dir / f"round_{round_num}" / "metrics.json"
            if metrics_path.exists():
                import json
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                return data.get("personas", {})

        # Fallback mock for standalone tests
        return {
            "spoofing": {"precision": 0.90, "recall": 0.86, "f1": 0.88},
            "wash_trading": {"precision": 0.85, "recall": 0.80, "f1": 0.82},
        }

    def get_shap_drift(self, persona: str, round_a: int, round_b: int) -> dict:
        """
        Reads metrics for both rounds from disk and computes deltas.
        """
        m_a = self.get_round_metrics(round_a).get(persona, {})
        m_b = self.get_round_metrics(round_b).get(persona, {})

        if not m_a or not m_b:
            return {
                "persona": persona,
                "note": f"Could not compute drift: missing metrics for round {round_a} or {round_b}.",
            }

        return {
            "persona": persona,
            "round_a": round_a,
            "round_b": round_b,
            "f1_delta": round(m_b.get("f1", 0) - m_a.get("f1", 0), 4),
            "precision_delta": round(m_b.get("precision", 0) - m_a.get("precision", 0), 4),
            "recall_delta": round(m_b.get("recall", 0) - m_a.get("recall", 0), 4),
            "interpretation": (
                "Negative f1_delta = detector got worse (evasion worked). "
                "Positive = detector recovered."
            ),
        }

    def get_enforcement_notes(self, persona: str) -> str:
        """Reads qualitative grounding from docs/enforcement_case_notes.md."""
        notes_path = Path("docs/enforcement_case_notes.md")
        if notes_path.exists():
            content = notes_path.read_text(encoding="utf-8")
            # Return the first section mentioning this persona
            persona_keyword = persona.replace("_", " ")
            lines = content.splitlines()
            relevant = []
            in_section = False
            for line in lines:
                if persona_keyword.lower() in line.lower():
                    in_section = True
                if in_section:
                    relevant.append(line)
                if in_section and len(relevant) > 20:
                    break
            if relevant:
                return "\n".join(relevant[:20])

        # Fallback hardcoded notes
        if "spoof" in persona.lower():
            return (
                "CFTC vs Navinder Sarao: Spoofer placed large passive orders away "
                "from best bid/ask to create false impression of supply, then cancelled "
                "before execution. Key signal: high cancel rate, large out-of-the-money orders."
            )
        return f"Enforcement notes for {persona}: Traders coordinated to falsely inflate volume."

    def propose_mutation(self, persona: str, parameters: dict, reasoning: str = "") -> dict:
        """
        Validates the proposed parameters and writes them to configs/persona_config.yaml
        under the top-level persona key (separate from round_0 baseline).
        """
        import yaml

        # Validate types
        if "cancellation_delay_ticks" in parameters:
            if not isinstance(parameters["cancellation_delay_ticks"], (int, float)) or parameters["cancellation_delay_ticks"] <= 0:
                return {"error": "cancellation_delay_ticks must be a positive number."}

        config_path = Path("configs/persona_config.yaml")
        existing = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                existing = yaml.safe_load(f) or {}

        old_params = existing.get(persona, existing.get("round_0", {}).get(persona, {})).copy()

        if persona not in existing:
            existing[persona] = {}

        existing[persona].update(parameters)

        with open(config_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)

        return {
            "status": "success", 
            "message": f"Successfully mutated {persona} config.",
            "old_params": old_params,
            "new_params": existing[persona],
            "reasoning": reasoning
        }
