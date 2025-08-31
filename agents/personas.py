"""Multi‑persona agent definitions and helpers.

Provides lightweight role prompts and response helpers using DeepSeek via
utils.llm_utils. Designed to be UI-friendly and safe by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from lab.config import get
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
                f"You are a careful professor advising a research team. The project goal is: {goal}. "
                "Provide high-level guidance, critique plans, and keep the team focused on novelty and clarity."
            ),
        ),
        "PhD": Persona(
            name="PhD",
            system=(
                f"You are a PhD student formulating concrete plans for experiments. The project goal is: {goal}. "
                "Propose simple, testable experiments and articulate expected outcomes, datasets, and metrics."
            ),
        ),
        "ML": Persona(
            name="ML",
            system=(
                f"You are an ML engineer implementing the plan. The project goal is: {goal}. "
                "Suggest concise code fragments and minimal changes; be pragmatic and keep CPU/1-epoch limits in mind."
            ),
        ),
        "SW": Persona(
            name="SW",
            system=(
                f"You are a software engineer coordinating with the ML engineer. The project goal is: {goal}. "
                "Focus on integration, readability, and runnable steps; keep suggestions actionable."
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
    return chat_text_cached(msgs, temperature=temperature)


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

