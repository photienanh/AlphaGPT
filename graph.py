# graph.py
"""
LangGraph workflow theo paper Alpha-GPT.
DATA_STORE[thread_id] = (df_panel, ticker_dfs, fwd_ret_multi)
"""
import os
import logging
from typing import Literal, Dict, Any, Tuple
import pandas as pd
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from state import State
from agents.hypothesis_agent      import hypothesis_agent
from agents.alpha_generator_agent import alpha_generator_agent
from agents.gp_agent               import gp_agent
from agents.backtest_agent         import backtest_agent
from agents.analyst_agent          import analyst_agent
from agents.persist_agent          import persist_agent
from data_loader import (
    load_multi_stock, make_sample_data_multi
)
from config import DEFAULT_CONFIG

log = logging.getLogger(__name__)

# DATA_STORE[thread_id] = (df_panel, ticker_dfs, fwd_ret_multi)
DATA_STORE: Dict[str, Tuple] = {}


async def data_injector(state: State, config) -> dict:
    """Load multi-stock data một lần duy nhất."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    if thread_id in DATA_STORE:
        return {"thread_id": thread_id}

    data_dir = os.environ.get("ALPHAGPT_DATA_DIR", "data")
    if os.path.isdir(data_dir):
        df_panel, ticker_dfs, fwd_ret_multi = load_multi_stock(
            data_dir,
            min_history_days=DEFAULT_CONFIG.min_history_days,
        )
        if ticker_dfs:
            log.info(f"[DataInjector] Loaded {len(ticker_dfs)} tickers")
        else:
            log.warning("[DataInjector] Không load được ticker nào, dùng synthetic data")
            ticker_dfs, fwd_ret_multi = make_sample_data_multi(n_days=500, n_tickers=10)
            df_panel = pd.DataFrame()
    else:
        log.warning("[DataInjector] data_dir không tồn tại, dùng synthetic data")
        ticker_dfs, fwd_ret_multi = make_sample_data_multi(n_days=500, n_tickers=10)
        df_panel = pd.DataFrame()

    DATA_STORE[thread_id] = (df_panel, ticker_dfs, fwd_ret_multi)
    return {"thread_id": thread_id}


def should_loop(state: State) -> Literal["hypothesis_generator", "__end__"]:
    if state.should_continue and state.iteration < state.max_iterations:
        log.info(f"[Router] Continue to iteration {state.iteration + 1}")
        return "hypothesis_generator"
    log.info(f"[Router] Pipeline complete after {state.iteration} iterations")
    return "__end__"


def create_graph():
    workflow = StateGraph(State)

    workflow.add_node("data_injector",        data_injector)
    workflow.add_node("hypothesis_generator", hypothesis_agent)
    workflow.add_node("alpha_generator",      alpha_generator_agent)
    workflow.add_node("gp_enhancement",       gp_agent)
    workflow.add_node("backtest",             backtest_agent)
    workflow.add_node("analyst",              analyst_agent)
    workflow.add_node("persist",              persist_agent)

    workflow.add_edge("__start__",            "data_injector")
    workflow.add_edge("data_injector",        "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "alpha_generator")
    workflow.add_edge("alpha_generator",      "gp_enhancement")
    workflow.add_edge("gp_enhancement",       "backtest")
    workflow.add_edge("backtest",             "analyst")
    workflow.add_edge("analyst",              "persist")

    workflow.add_conditional_edges(
        "persist",
        should_loop,
        {
            "hypothesis_generator": "hypothesis_generator",
            "__end__":              "__end__",
        },
    )

    graph = workflow.compile(checkpointer=MemorySaver())
    graph.name = "Alpha-GPT Pipeline (Paper-aligned)"
    return graph


graph = create_graph()