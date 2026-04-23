# agents/hypothesis_agent.py
import json
import logging
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from state import State
from prompts.hypothesis_prompts import (
    HYPOTHESIS_SYSTEM_PROMPT,
    HYPOTHESIS_INITIAL_PROMPT,
    HYPOTHESIS_ITERATION_PROMPT,
    HYPOTHESIS_OUTPUT_FORMAT,
)

log = logging.getLogger(__name__)


async def hypothesis_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Ideation stage — paper Section 2.1.
    Vòng 1: generate từ trading_idea.
    Vòng N: refine dựa trên analyst_feedback + hypothesis_history.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    is_first = not state.hypothesis_history

    if is_first:
        user_prompt = HYPOTHESIS_INITIAL_PROMPT.format(
            trading_idea=state.trading_idea,
            output_format=HYPOTHESIS_OUTPUT_FORMAT,
        )
    else:
        # Format history kèm backtest metrics nếu có
        history_text = ""
        for i, h in enumerate(state.hypothesis_history[-3:], 1):
            history_text += f"\nVòng {h.get('iteration', i)}: {h.get('hypothesis', '')}\n"
            if h.get("alpha_summary"):
                history_text += f"  Kết quả: {h['alpha_summary']}\n"

        user_prompt = HYPOTHESIS_ITERATION_PROMPT.format(
            hypothesis_history=history_text,
            analyst_feedback=state.analyst_feedback or "Chưa có feedback.",
            output_format=HYPOTHESIS_OUTPUT_FORMAT,
        )

    response = await llm.ainvoke([
        {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ])

    content = response.content
    j_start = content.find("{")
    j_end   = content.rfind("}") + 1
    data = json.loads(content[j_start:j_end] if j_start >= 0 else content)

    iteration = state.iteration + 1 if not is_first else 1
    log.info(f"[Hypothesis] Iteration {iteration}: {data.get('hypothesis', '')[:80]}")

    return {
        "hypothesis":             data.get("hypothesis", ""),
        "reason":                 data.get("reason", ""),
        "concise_reason":         data.get("concise_reason", ""),
        "concise_observation":    data.get("concise_observation", ""),
        "concise_justification":  data.get("concise_justification", ""),
        "concise_knowledge":      data.get("concise_knowledge", ""),
        "iteration":              iteration,
    }