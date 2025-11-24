# graph_builder.py
from langgraph.graph import StateGraph
from state_model import OutreachState

from nodes import (
    discover_companies_node,
    find_contacts_node,
    collect_research_node,
    draft_emails_node
)



def build_graph():
    graph = StateGraph(OutreachState)

    # Register nodes with config/runtime support
    graph.add_node(
        "discover_companies",
        discover_companies_node,
        config=True,
        runtime=True
    )
    graph.add_node(
        "find_contacts",
        find_contacts_node,
        config=True,
        runtime=True
    )
    graph.add_node(
        "collect_research",
        collect_research_node,
        config=True,
        runtime=True
    )
    graph.add_node(
        "draft_emails",
        draft_emails_node,
        config=True,
        runtime=True
    )

    # Edges in sequence
    graph.add_edge("discover_companies", "find_contacts")
    graph.add_edge("find_contacts", "collect_research")
    graph.add_edge("collect_research", "draft_emails")
    graph.add_edge("__start__", "discover_companies")


    # Start + end points
    graph.set_entry_point("discover_companies")
    graph.set_finish_point("draft_emails")

    # Compile graph
    return graph.compile()
