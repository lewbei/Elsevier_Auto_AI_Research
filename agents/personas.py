"""Multi‑persona agent definitions and helpers.

Provides lightweight role prompts and response helpers using DeepSeek via
utils.llm_utils. Designed to be UI-friendly and safe by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from lab.config import get
from utils.llm_utils import chat_text_cached, LLMError
from utils.llm_utils import chat_text_cached, LLMError


@dataclass
class Persona:
    name: str
    system: str


def _goal() -> str:
    g = get("project.goal", None)
    return str(g or "conduct an ML research study end-to-end on a small image dataset")


def make_personas() -> Dict[str, Persona]:
    goal = _goal()
    return {
        "Professor": Persona(
            name="Professor",
            system=(
                f"You are a seasoned professor and principal investigator advising a research team. The project goal is: {goal}.\n"
                "Objectives: provide rigorous high-level guidance, critique novelty and experimental design, identify risks, clarify assumptions, and ensure the research narrative is coherent.\n"
                "Style: precise, academic tone; favor explicit reasoning about trade-offs; cite common pitfalls and recommended mitigations.\n"
                "Constraints: limited compute (CPU or single GPU), <=1 epoch per run, <=100 steps; minimal dependencies; reproducible defaults.\n"
                "Output format: short, numbered bullets with specific actions, measurable criteria, and rationale; avoid vague advice."
            ),
        ),
        "Postdoc": Persona(
            name="Postdoc",
            system=(
                f"You are a postdoctoral researcher bridging research rigor and pragmatic execution. The project goal is: {goal}.\n"
                "Identify subtle risks and edge cases, propose strong ablations/controls, and refine ideas into testable, constraints-aware steps."
            ),
        ),
        "PhD": Persona(
            name="PhD",
            system=(
                f"You are a diligent PhD student translating ideas into concrete, testable plans aligned to: {goal}.\n"
                "Tasks: propose small sets of experiments (baseline, novelty, ablation), define datasets/paths/splits, metrics, and early-stopping rules.\n"
                "Be explicit about variables to sweep, expected deltas, and how to interpret negative results.\n"
                "Respect constraints: <=1 epoch, <=100 steps, small batch sizes, safe fallbacks when data/models missing.\n"
                "Output: clear bullets with exact parameter values/ranges and brief justifications."
            ),
        ),
        "ML": Persona(
            name="ML",
            system=(
                f"You are an ML engineer responsible for implementation and runnable code paths in service of: {goal}.\n"
                "Provide concrete code-oriented suggestions (transforms, heads, learning-rate tweaks), minimal patches, and validation checks.\n"
                "Prefer torchvision and torch.nn primitives, explain when to toggle features, and keep CPU-first constraints.\n"
                "Output: actionable steps (file:line or function names when relevant), guardrails, and quick sanity tests."
            ),
        ),
        "SW": Persona(
            name="SW",
            system=(
                f"You are a pragmatic software engineer ensuring integration quality for: {goal}.\n"
                "Focus on: configuration consistency, error handling, logging, reproducibility, and small diffs that preserve tests.\n"
                "Recommend where to log artifacts, how to structure prompts/config, and how to gate risky ops behind flags.\n"
                "Output: concise checklists and dependency considerations with rollback strategies."
            ),
        ),
    }


def respond(persona: Persona, history: List[Dict[str, str]], user: str, temperature: float = 0.2) -> str:
    """Return persona response given prior chat history and current user message.

    History messages are role-tagged: [{role: system|user|assistant, content: str}].
    """
    msgs: List[Dict[str, str]] = [{"role": "system", "content": persona.system}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user})
    model = get("pipeline.personas.model", None)
    profile = get("pipeline.personas.llm", None)
    return chat_text_cached(msgs, temperature=temperature, model=model, profile=profile)


class DialogueManager:
    """Minimal dialogue orchestrator across multiple personas.

    Keeps a linear history shared across personas and supports auto turn-taking.
    """

    def __init__(self, order: Optional[List[str]] = None):
        self._personas = make_personas()
        self.history: List[Dict[str, str]] = []
        self.order = order or ["PhD", "Professor", "SW", "ML"]

    def list_personas(self) -> List[str]:
        return list(self._personas.keys())

    def post(self, author: str, text: str) -> None:
        self.history.append({"role": "user", "content": f"[{author}] {text}"})

    def step_auto(self) -> Dict[str, str]:
        """Advance one persona turn following the configured order."""
        # Count how many assistant turns are in history to select next persona
        turns = sum(1 for m in self.history if m.get("role") == "assistant")
        role_name = self.order[turns % len(self.order)]
        persona = self._personas[role_name]
        try:
            reply = respond(persona, self.history, user=f"Next step as {role_name}: be concrete and short.")
        except LLMError as exc:
            reply = f"[LLM error: {exc}]"
        msg = {"role": "assistant", "content": f"[{role_name}] {reply}"}
        self.history.append(msg)
        return {"role": role_name, "text": reply}

    def step_role(self, role_name: str, prompt: str) -> Dict[str, str]:
        persona = self._personas.get(role_name)
        if persona is None:
            return {"role": role_name, "text": f"Unknown role: {role_name}"}
        try:
            reply = respond(persona, self.history, user=prompt)
        except LLMError as exc:
            reply = f"[LLM error: {exc}]"
        msg = {"role": "assistant", "content": f"[{role_name}] {reply}"}
        self.history.append(msg)
        return {"role": role_name, "text": reply}
