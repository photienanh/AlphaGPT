# agents/hypothesis_agent.py
"""
Ideation stage — paper Section 2.1.
Tích hợp RAG từ knowledge base trước khi call LLM.
"""
import json
import logging
import os
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
from knowledge.retriever import retrieve_similar_alphas
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)


def _format_rag_examples(alphas: list) -> str:
    if not alphas:
        return ""
    lines = ["## Relevant alpha examples from knowledge base\n"]
    for a in alphas:
        lines.append(
            f"- **{a['id']}**: {a.get('description', '')}\n"
            f"  `{a.get('expression', '')[:100]}`"
        )
    return "\n".join(lines)

    
async def hypothesis_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Vòng 1: generate từ trading_idea + RAG examples.
    Vòng N: refine dựa trên analyst_feedback + RAG examples.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    is_first = not state.hypothesis_history

    # RAG: retrieve similar alphas
    query = state.trading_idea if is_first else (state.analyst_feedback or state.trading_idea)
    rag_alphas = retrieve_similar_alphas(query, top_k=DEFAULT_CONFIG.rag_top_k)
    rag_block = _format_rag_examples(rag_alphas)

    if is_first:
        user_prompt = HYPOTHESIS_INITIAL_PROMPT.format(
            trading_idea=state.trading_idea,
            rag_examples=rag_block,
            output_format=HYPOTHESIS_OUTPUT_FORMAT,
        )
    else:
        history_text = ""
        for i, h in enumerate(state.hypothesis_history[-3:], 1):
            history_text += f"\nVòng {h.get('iteration', i)}: {h.get('hypothesis', '')}\n"
            if h.get("alpha_summary"):
                history_text += f"  Kết quả: {h['alpha_summary']}\n"
            if h.get("round_summary"):
                history_text += f"  Nhận xét: {h['round_summary']}\n"

        user_prompt = HYPOTHESIS_ITERATION_PROMPT.format(
            hypothesis_history=history_text,
            analyst_feedback=state.analyst_feedback or "Chưa có feedback.",
            rag_examples=rag_block,
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