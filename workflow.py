"""LangGraph multi-step research workflow: plan → research → critique → write."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


class ResearchState(TypedDict):
    topic: str
    plan: list[str]
    findings: list[str]
    critique: str
    report: str
    revision_count: int
    approved: bool


class ResearchPlan(BaseModel):
    questions: list[str] = Field(description="3-5 focused research questions")


class CritiqueResult(BaseModel):
    approved: bool = Field(description="True if the draft is good enough")
    feedback: str = Field(description="What to improve if not approved")


def get_llm(temp: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temp)


def plan_node(state: ResearchState) -> dict:
    llm = get_llm().with_structured_output(ResearchPlan)
    result: ResearchPlan = llm.invoke(
        [
            SystemMessage(
                content="You are a research planner. Create 3-5 sharp research questions."
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    return {"plan": result.questions, "findings": [], "revision_count": 0, "approved": False}


def research_node(state: ResearchState) -> dict:
    findings: list[str] = list(state.get("findings") or [])
    questions = state["plan"]
    with DDGS() as ddgs:
        for q in questions:
            hits = list(ddgs.text(q, max_results=3))
            snippets = " | ".join(h.get("body", "")[:220] for h in hits) or "No results"
            findings.append(f"Q: {q}\nEvidence: {snippets}")
    return {"findings": findings}


def write_node(state: ResearchState) -> dict:
    llm = get_llm(temp=0.4)
    critique = state.get("critique") or "None yet"
    prompt = (
        f"Write a concise research brief on: {state['topic']}\n\n"
        f"Research plan:\n{json.dumps(state['plan'], indent=2)}\n\n"
        f"Findings:\n" + "\n\n".join(state["findings"]) + "\n\n"
        f"Previous critique to address:\n{critique}\n\n"
        "Structure: Executive summary, Key findings, Risks/unknowns, Recommendations."
    )
    report = llm.invoke([HumanMessage(content=prompt)]).content
    return {"report": report}


def critique_node(state: ResearchState) -> dict:
    llm = get_llm().with_structured_output(CritiqueResult)
    result: CritiqueResult = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Critique the research brief for clarity, evidence quality, and actionability. "
                    "Approve only if it is solid enough for a decision-maker."
                )
            ),
            HumanMessage(
                content=f"Topic: {state['topic']}\n\nDraft:\n{state['report']}"
            ),
        ]
    )
    return {
        "approved": result.approved,
        "critique": result.feedback,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def route_after_critique(state: ResearchState) -> Literal["write", "save"]:
    if state.get("approved") or state.get("revision_count", 0) >= 2:
        return "save"
    return "write"


def save_node(state: ResearchState) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in state["topic"])[:60]
    path = OUTPUT_DIR / f"{safe}_report.md"
    path.write_text(
        f"# Research Report: {state['topic']}\n\n{state['report']}\n\n"
        f"## Critique notes\n{state.get('critique', '')}\n",
        encoding="utf-8",
    )
    return {}


def build_workflow():
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("critique", critique_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"write": "write", "save": "save"})
    graph.add_edge("save", END)
    return graph.compile()


def run_research(topic: str) -> ResearchState:
    app = build_workflow()
    return app.invoke(
        {
            "topic": topic,
            "plan": [],
            "findings": [],
            "critique": "",
            "report": "",
            "revision_count": 0,
            "approved": False,
        }
    )


def main() -> None:
    print("LangGraph Research Workflow (plan → research → write → critique)")
    topic = input("Research topic> ").strip()
    if not topic:
        print("No topic provided.")
        return
    result = run_research(topic)
    print("\n===== FINAL REPORT =====\n")
    print(result["report"])
    print(f"\nApproved={result['approved']} revisions={result['revision_count']}")
    print(f"Saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
