"""Multi‑persona agent definitions and helpers.

Provides lightweight role prompts and response helpers using DeepSeek via
utils.llm_utils. Designed to be UI-friendly and safe by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lab.config import get
from utils.llm_utils import chat_text_cached, LLMError
import json


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
        "Statistician": Persona(
            name="Statistician",
            system=(
                f"You are a statistician ensuring rigor for: {goal}.\n"
                "Enforce stratified k-fold CV, macro/micro metrics, CIs via bootstrap, calibration (ECE), and leakage checks.\n"
                "Output: JSON plan with metrics, CV protocol, and sample size/budget constraints."
            ),
        ),
        "Auditor": Persona(
            name="Auditor",
            system=(
                f"You are a reproducibility auditor for: {goal}.\n"
                "Require seeds, deterministic flags, artifact paths, data provenance, and error/rollback.\n"
                "Output: JSON with risks, gating checks, and artifacts to log."
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

def respond_json(persona: Persona, history: List[Dict[str, str]], user: str, temperature: float = 0.2) -> Dict:
    """Return persona response as strict JSON for downstream execution/planning."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": persona.system + "\n\nOutput JSON only. No prose."}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user})
    model = get("pipeline.personas.model", None)
    profile = get("pipeline.personas.llm", None)
    txt = chat_text_cached(msgs, temperature=temperature, model=model, profile=profile)
    try:
        return json.loads(txt)
    except Exception:
        raise LLMError(f"Non-JSON from {persona.name}: {txt[:200]}")


class DialogueManager:
    """Minimal dialogue orchestrator across multiple personas.

    Keeps a linear history shared across personas and supports auto turn-taking.
    """

    @dataclass
    class ResearchState:
        plan: Dict[str, List[Dict]] = field(default_factory=lambda: {"experiments": []})
        risks: List[str] = field(default_factory=list)
        decided: Dict[str, str] = field(default_factory=dict)

    def __init__(self, order: Optional[List[str]] = None):
        self._personas = make_personas()
        self.history: List[Dict[str, str]] = []
        self.order = order or ["PhD", "Professor", "Statistician", "ML", "Auditor", "SW"]
        self.state = self.ResearchState()

    def _next_role(self) -> str:
        # Gap-driven routing: select persona by what's missing in state
        try:
            if not self.state.plan.get("experiments"):
                return "PhD"
            if not any("cv" in str(e.get("notes", "")).lower() for e in self.state.plan.get("experiments", [])):
                return "Statistician"
            if not self.state.risks:
                return "Professor"
            if not self.state.decided.get("artifacts"):
                return "Auditor"
        except Exception:
            pass
        return "ML"

    def list_personas(self) -> List[str]:
        return list(self._personas.keys())

    def post(self, author: str, text: str) -> None:
        self.history.append({"role": "user", "content": f"[{author}] {text}"})

    def step_auto(self) -> Dict[str, str]:
        """Advance one persona turn using gap-driven order and JSON outputs where possible."""
        role_name = self._next_role()
        persona = self._personas[role_name]
        try:
            js = respond_json(persona, self.history, user=f"Next step as {role_name}: fill the JSON schema only.")
        except LLMError as exc:
            js = {"actions": [{"type": "decision", "id": "llm_error"}], "risks": [str(exc)]}
        # Merge minimal state
        try:
            self.state.plan.setdefault("experiments", []).extend(js.get("experiments", []))
            self.state.risks.extend(js.get("risks", []))
        except Exception:
            pass
        # Log a compact JSON snippet to history to prevent token blow-up
        payload = json.dumps(js, ensure_ascii=False)
        msg = {"role": "assistant", "content": f"[{role_name}] {payload[:2000]}"}
        self.history.append(msg)
        return {"role": role_name, "json": js}

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
