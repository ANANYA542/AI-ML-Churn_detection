"""Agentic workflow for churn retention strategy planning using LangGraph.

Builds a four-node StateGraph that takes customer data and a churn
probability, then produces a risk level, contributing factors, a
retention strategy, and supporting reasoning/disclaimers.

Nodes (executed in order):
    analyze_risk → identify_factors → generate_strategy → add_disclaimers
"""

import json
from typing import TypedDict, List, Dict

from langgraph.graph import StateGraph, END

from src.llm_client import get_llm_response


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ChurnAgentState(TypedDict):
    customer_data: Dict
    churn_probability: float
    risk_level: str
    contributing_factors: List[str]
    retention_strategy: str
    sources: List[str]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def analyze_risk(state: ChurnAgentState) -> dict:
    """Classify churn probability into Low / Medium / High risk using the LLM."""

    prob = state["churn_probability"]

    prompt = (
        f"A customer has a churn probability of {prob:.2%}.\n\n"
        "Classify this into exactly one of the following risk levels:\n"
        "- Low (probability ≤ 0.3)\n"
        "- Medium (probability between 0.3 and 0.7)\n"
        "- High (probability > 0.7)\n\n"
        "Respond with ONLY the single word: Low, Medium, or High."
    )

    system = (
        "You are a risk-classification assistant. "
        "Return exactly one word — Low, Medium, or High — with no extra text."
    )

    raw = get_llm_response(prompt, system_prompt=system).strip().rstrip(".")

    # Normalise the LLM response to one of the three canonical labels
    normalised = raw.capitalize()
    if normalised not in {"Low", "Medium", "High"}:
        # Fallback: deterministic rule if the LLM drifts
        if prob > 0.7:
            normalised = "High"
        elif prob > 0.3:
            normalised = "Medium"
        else:
            normalised = "Low"

    return {"risk_level": normalised}


def identify_factors(state: ChurnAgentState) -> dict:
    """Identify the top contributing factors for potential churn."""

    customer_json = json.dumps(state["customer_data"], indent=2, default=str)

    prompt = (
        f"CUSTOMER PROFILE:\n{customer_json}\n\n"
        f"RISK LEVEL: {state['risk_level']}\n\n"
        "Based on the customer profile above, identify the top 3-5 factors "
        "that most likely contribute to this customer's churn risk.\n\n"
        "Rules:\n"
        "- Each factor must reference a specific field and value from the profile.\n"
        "- Explain in one sentence why that field/value raises or lowers churn risk.\n"
        "- Do not invent data that is not in the profile.\n\n"
        "Return ONLY a JSON array of strings. Example:\n"
        '["Month-to-month contract increases flexibility to leave", '
        '"High monthly charges of $95 exceed segment average"]'
    )

    system = (
        "You are a customer-churn analyst. "
        "Return a JSON array of concise factor strings — nothing else."
    )

    raw = get_llm_response(prompt, system_prompt=system)

    # Parse the JSON array from the LLM response
    try:
        factors = json.loads(raw)
        if not isinstance(factors, list):
            raise ValueError
        factors = [str(f) for f in factors]
    except (json.JSONDecodeError, ValueError):
        # Best-effort: split by newlines and clean up
        factors = [
            line.lstrip("- •0123456789.").strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    return {"contributing_factors": factors}


def generate_strategy(state: ChurnAgentState) -> dict:
    """Generate an actionable retention strategy based on all preceding analysis."""

    customer_json = json.dumps(state["customer_data"], indent=2, default=str)
    factors_text = "\n".join(f"- {f}" for f in state["contributing_factors"])

    prompt = (
        f"CUSTOMER PROFILE:\n{customer_json}\n\n"
        f"RISK LEVEL: {state['risk_level']}\n\n"
        f"CONTRIBUTING FACTORS:\n{factors_text}\n\n"
        "Based on the information above, propose a concrete retention strategy "
        "consisting of 2-4 actionable steps the business can take to retain this customer.\n\n"
        "Rules:\n"
        "- Each step must directly address one of the contributing factors.\n"
        "- Prefer low-cost, measurable, and ethical actions.\n"
        "- Be specific (e.g., 'offer a 12-month contract at a 10 % discount') rather than vague.\n"
        "- If the risk level is Low, focus on monitoring and lightweight engagement.\n\n"
        "Return the strategy as a numbered list in plain text."
    )

    system = (
        "You are a customer retention strategist for a telecom company. "
        "Be concise, practical, and ground every recommendation in the data provided."
    )

    strategy = get_llm_response(prompt, system_prompt=system)

    return {"retention_strategy": strategy}


def add_disclaimers(state: ChurnAgentState) -> dict:
    """Provide reasoning sources and disclaimers behind the decisions."""

    prompt = (
        f"RISK LEVEL: {state['risk_level']}\n\n"
        f"CONTRIBUTING FACTORS:\n"
        + "\n".join(f"- {f}" for f in state["contributing_factors"])
        + f"\n\nRETENTION STRATEGY:\n{state['retention_strategy']}\n\n"
        "Provide 3-5 brief supporting reasons or references that justify the "
        "retention strategy above. Include a final disclaimer noting that these "
        "are AI-generated suggestions and should be reviewed by a human.\n\n"
        "Return ONLY a JSON array of strings. Example:\n"
        '["Reason 1", "Reason 2", "Disclaimer: ..."]'
    )

    system = (
        "You are a compliance-aware AI assistant. "
        "Return a JSON array of concise reasoning strings — nothing else."
    )

    raw = get_llm_response(prompt, system_prompt=system)

    try:
        sources = json.loads(raw)
        if not isinstance(sources, list):
            raise ValueError
        sources = [str(s) for s in sources]
    except (json.JSONDecodeError, ValueError):
        sources = [
            line.lstrip("- •0123456789.").strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    # Always ensure a disclaimer is present
    has_disclaimer = any("disclaimer" in s.lower() for s in sources)
    if not has_disclaimer:
        sources.append(
            "Disclaimer: These suggestions are AI-generated and must be "
            "reviewed by a human before being acted upon."
        )

    return {"sources": sources}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph workflow."""

    graph = StateGraph(ChurnAgentState)

    # Add nodes
    graph.add_node("analyze_risk", analyze_risk)
    graph.add_node("identify_factors", identify_factors)
    graph.add_node("generate_strategy", generate_strategy)
    graph.add_node("add_disclaimers", add_disclaimers)

    # Set entry point
    graph.set_entry_point("analyze_risk")

    # Connect nodes in sequence
    graph.add_edge("analyze_risk", "identify_factors")
    graph.add_edge("identify_factors", "generate_strategy")
    graph.add_edge("generate_strategy", "add_disclaimers")
    graph.add_edge("add_disclaimers", END)

    return graph.compile()


# Pre-compile the graph once at module level
_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(customer_data: dict, churn_probability: float) -> dict:
    """Run the churn-retention agent and return structured output.

    Args:
        customer_data: Dictionary of customer attributes (e.g. tenure,
            MonthlyCharges, Contract, etc.).
        churn_probability: Float between 0 and 1 representing the ML
            model's predicted churn probability.

    Returns:
        Dictionary with keys:
            - risk_level (str)
            - contributing_factors (list[str])
            - retention_strategy (str)
            - sources (list[str])
    """

    initial_state: ChurnAgentState = {
        "customer_data": customer_data,
        "churn_probability": churn_probability,
        "risk_level": "",
        "contributing_factors": [],
        "retention_strategy": "",
        "sources": [],
    }

    final_state = _compiled_graph.invoke(initial_state)

    return {
        "risk_level": final_state["risk_level"],
        "contributing_factors": final_state["contributing_factors"],
        "retention_strategy": final_state["retention_strategy"],
        "sources": final_state["sources"],
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
    }

    result = run_agent(sample_customer, churn_probability=0.82)

    print("=" * 60)
    print(f"Risk Level        : {result['risk_level']}")
    print(f"Contributing Factors:")
    for f in result["contributing_factors"]:
        print(f"  • {f}")
    print(f"\nRetention Strategy:\n{result['retention_strategy']}")
    print(f"\nSources / Reasoning:")
    for s in result["sources"]:
        print(f"  • {s}")
    print("=" * 60)
