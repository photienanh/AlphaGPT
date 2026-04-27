# state.py
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class State:

    # ── Input ────────────────────────────────────────────────────────
    trading_idea: str = ""

    # ── Hypothesis (Ideation stage) ──────────────────────────────────
    hypothesis: str = ""
    reason: str = ""
    iteration: int = 0

    # ── Implementation stage ─────────────────────────────────────────
    seed_alphas: List[Dict[str, Any]] = field(default_factory=list)
    candidate_alphas: List[Dict[str, Any]] = field(default_factory=list)

    # ── Review stage ─────────────────────────────────────────────────
    evaluated_alphas: List[Dict[str, Any]] = field(default_factory=list)
    sota_alphas: List[Dict[str, Any]] = field(default_factory=list)
    analyst_summary: str = ""
    refinement_directions: str = ""
    analyst_weak_ids: List[str] = field(default_factory=list)

    # ── Loop control ─────────────────────────────────────────────────
    max_iterations: int = 3
    should_continue: bool = True

    # ── History ──────────────────────────────────────────────────────
    hypothesis_history: List[Dict[str, Any]] = field(default_factory=list)
    alpha_history: List[Dict[str, Any]] = field(default_factory=list)

    # ── Data reference ────────────────────────────────────────────────
    thread_id: str = ""