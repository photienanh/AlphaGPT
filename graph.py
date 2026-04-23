# graph.py
"""
LangGraph workflow theo đúng paper Alpha-GPT:

Ideation:       user_input → hypothesis_generator
Implementation: alpha_generator → gp_enhancement
Review:         backtest → analyst → persist
Loop:           should_continue? → hypothesis_generator (next round) | END

Nodes truy cập data qua:
    from graph import DATA_STORE
    df, fwd_ret = DATA_STORE[thread_id]
"""
import os
import logging
from typing import Literal, Dict, Any, Tuple
import pandas as pd
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from state import State
from agents.hypothesis_agent  import hypothesis_agent
from agents.alpha_generator_agent import alpha_generator_agent
from agents.gp_agent           import gp_agent
from agents.backtest_agent     import backtest_agent
from agents.analyst_agent      import analyst_agent
from agents.persist_agent      import persist_agent
from data_loader import load_from_csv, load_from_dataframe, make_forward_return, make_sample_data

log = logging.getLogger(__name__)

DATA_STORE: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
# ── Data injector node ────────────────────────────────────────────────

async def data_injector(state: State, config) -> dict:
    """
    Load data một lần duy nhất và inject vào state.
    Đọc từ DATA_PATH env var hoặc dùng synthetic data để test.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")
 
    # Đã load rồi thì bỏ qua
    if thread_id in DATA_STORE:
        return {"thread_id": thread_id}

    data_path = os.environ.get("ALPHAGPT_DATA_PATH", "")
    if data_path and os.path.exists(data_path):
        df = load_from_csv(data_path)
        log.info(f"[DataInjector] Loaded {len(df)} rows from {data_path}")
    else:
        log.warning("[DataInjector] No data path set, using synthetic data")
        df = make_sample_data(n_days=500)
        if "time" in df.columns:
            import pandas as pd
            df = df.set_index("time")

    fwd_ret = make_forward_return(df, horizon=1)
    DATA_STORE[thread_id] = (df, fwd_ret)
 
    return {"thread_id": thread_id}


# ── Router: tiếp tục loop hay dừng ───────────────────────────────────

def should_loop(state: State) -> Literal["hypothesis_generator", "__end__"]:
    """
    Sau persist: nếu should_continue=True → quay lại hypothesis_generator.
    Ngược lại → END.
    """
    if state.should_continue and state.iteration < state.max_iterations:
        log.info(f"[Router] Continue to iteration {state.iteration + 1}")
        return "hypothesis_generator"
    log.info(f"[Router] Pipeline complete after {state.iteration} iterations")
    return "__end__"


# ── Build graph ───────────────────────────────────────────────────────

def create_graph():
    workflow = StateGraph(State)

    # Nodes
    workflow.add_node("data_injector",       data_injector)
    workflow.add_node("hypothesis_generator", hypothesis_agent)
    workflow.add_node("alpha_generator",     alpha_generator_agent)
    workflow.add_node("gp_enhancement",      gp_agent)
    workflow.add_node("backtest",            backtest_agent)
    workflow.add_node("analyst",             analyst_agent)
    workflow.add_node("persist",             persist_agent)

    # Edges — sequential pipeline
    workflow.add_edge("__start__",           "data_injector")
    workflow.add_edge("data_injector",       "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "alpha_generator")
    workflow.add_edge("alpha_generator",     "gp_enhancement")
    workflow.add_edge("gp_enhancement",      "backtest")
    workflow.add_edge("backtest",            "analyst")
    workflow.add_edge("analyst",             "persist")

    # Conditional edge: loop hay END
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