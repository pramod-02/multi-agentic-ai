import os
import time
import json
import psycopg
import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/multi_agent"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return psycopg.connect(DATABASE_URL)


def create_database_tables():

    sql = """

    CREATE TABLE IF NOT EXISTS tasks (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(100),
        goal TEXT NOT NULL,
        status VARCHAR(30),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS subtasks (
        id SERIAL PRIMARY KEY,
        task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
        agent VARCHAR(50),
        status VARCHAR(30),
        output TEXT
    );

    CREATE TABLE IF NOT EXISTS agent_runs (
        id SERIAL PRIMARY KEY,
        subtask_id INTEGER REFERENCES subtasks(id) ON DELETE CASCADE,
        agent_name VARCHAR(100),
        input TEXT,
        output TEXT,
        duration NUMERIC,
        retries INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        from_agent VARCHAR(100),
        to_agent VARCHAR(100),
        content TEXT,
        task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS tool_calls (
        id SERIAL PRIMARY KEY,
        agent_run_id INTEGER REFERENCES agent_runs(id) ON DELETE CASCADE,
        tool_name VARCHAR(100),
        input TEXT,
        output TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS evaluations (
        id SERIAL PRIMARY KEY,
        task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
        with_verifier BOOLEAN,
        accuracy NUMERIC,
        completeness NUMERIC,
        quality NUMERIC,
        revisions INTEGER,
        time_sec NUMERIC,
        cost NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def create_task(goal):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO tasks
                (user_id, goal, status)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                ("student", goal, "running")
            )

            return cur.fetchone()[0]


def update_task(task_id, status):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                UPDATE tasks
                SET status = %s
                WHERE id = %s
                """,
                (status, task_id)
            )


def create_subtask(task_id, agent):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO subtasks
                (task_id, agent, status)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (task_id, agent, "running")
            )

            return cur.fetchone()[0]


def finish_subtask(subtask_id, output):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                UPDATE subtasks
                SET status = %s,
                    output = %s
                WHERE id = %s
                """,
                ("completed", output, subtask_id)
            )


def save_agent_run(
    subtask_id,
    agent_name,
    input_text,
    output_text,
    duration
):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO agent_runs
                (
                    subtask_id,
                    agent_name,
                    input,
                    output,
                    duration
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    subtask_id,
                    agent_name,
                    input_text,
                    output_text,
                    duration
                )
            )


def save_message(
    task_id,
    from_agent,
    to_agent,
    content
):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO messages
                (
                    from_agent,
                    to_agent,
                    content,
                    task_id
                )
                VALUES (%s, %s, %s, %s)
                """,
                (
                    from_agent,
                    to_agent,
                    content,
                    task_id
                )
            )


def save_evaluation(
    task_id,
    revisions,
    elapsed,
    quality
):

    with get_connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO evaluations
                (
                    task_id,
                    with_verifier,
                    accuracy,
                    completeness,
                    quality,
                    revisions,
                    time_sec,
                    cost
                )
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    task_id,
                    True,
                    quality * 10,
                    quality * 10,
                    quality,
                    revisions,
                    elapsed,
                    0
                )
            )


# ============================================================
# OPTIONAL OPENAI
# ============================================================

def ask_llm(prompt):

    if DEMO_MODE or not OPENAI_API_KEY:
        return None

    try:

        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini"
            ),
            temperature=0
        )

        response = model.invoke(prompt)

        return response.content

    except Exception:
        return None


# ============================================================
# AGENT 1 - PLANNER
# ============================================================

def planner_agent(state):

    start = time.perf_counter()

    task_id = state["task_id"]
    goal = state["goal"]

    subtask_id = create_subtask(
        task_id,
        "Planner"
    )

    prompt = f"""
    Create a research plan for:

    {goal}

    Include:
    1. Research questions
    2. Evidence collection
    3. Analysis
    4. Report generation
    5. Verification
    """

    llm_result = ask_llm(prompt)

    if llm_result:

        plan = llm_result

    else:

        plan = json.dumps(
            {
                "research_question": goal,
                "steps": [
                    "Understand the research problem",
                    "Collect relevant evidence",
                    "Analyze the evidence",
                    "Generate a structured report",
                    "Verify the report"
                ]
            },
            indent=2
        )

    duration = time.perf_counter() - start

    finish_subtask(
        subtask_id,
        plan
    )

    save_agent_run(
        subtask_id,
        "Planner",
        goal,
        plan,
        duration
    )

    save_message(
        task_id,
        "Planner",
        "Researcher",
        "Research plan created."
    )

    return {
        "plan": plan
    }


# ============================================================
# AGENT 2 - RESEARCHER
# ============================================================

def researcher_agent(state):

    start = time.perf_counter()

    task_id = state["task_id"]

    subtask_id = create_subtask(
        task_id,
        "Researcher"
    )

    goal = state["goal"]

    # Demo research evidence
    evidence = f"""
RESEARCH EVIDENCE

Research Topic:
{goal}

Source 1:
General academic literature on multi-agent systems.

Source 2:
Research literature concerning agent orchestration,
task decomposition and verification.

Source 3:
Research concerning autonomous AI workflows.

Important:
These are demonstration evidence records.
For production research, connect a real search API
and validate every source before publication.
"""

    duration = time.perf_counter() - start

    finish_subtask(
        subtask_id,
        evidence
    )

    save_agent_run(
        subtask_id,
        "Researcher",
        goal,
        evidence,
        duration
    )

    save_message(
        task_id,
        "Researcher",
        "Analyst",
        "Research evidence collected."
    )

    return {
        "evidence": evidence
    }


# ============================================================
# AGENT 3 - ANALYST
# ============================================================

def analyst_agent(state):

    start = time.perf_counter()

    task_id = state["task_id"]

    subtask_id = create_subtask(
        task_id,
        "Analyst"
    )

    evidence = state["evidence"]

    analysis = f"""
ANALYSIS

The research evidence was divided into the following
analysis areas:

1. Task decomposition
2. Agent specialization
3. Inter-agent communication
4. Verification
5. Report generation

The multi-agent architecture allows different stages
of a research workflow to be separated into specialized
processing units.

Evidence used:

{evidence}
"""

    duration = time.perf_counter() - start

    finish_subtask(
        subtask_id,
        analysis
    )

    save_agent_run(
        subtask_id,
        "Analyst",
        evidence,
        analysis,
        duration
    )

    save_message(
        task_id,
        "Analyst",
        "Writer",
        "Analysis completed."
    )

    return {
        "analysis": analysis
    }


# ============================================================
# AGENT 4 - WRITER
# ============================================================

def writer_agent(state):

    start = time.perf_counter()

    task_id = state["task_id"]

    subtask_id = create_subtask(
        task_id,
        "Writer"
    )

    goal = state["goal"]
    evidence = state["evidence"]
    analysis = state["analysis"]

    previous_feedback = state.get(
        "verification_feedback",
        ""
    )

    prompt = f"""
    Create a technical research report.

    Research question:
    {goal}

    Evidence:
    {evidence}

    Analysis:
    {analysis}

    Previous verifier feedback:
    {previous_feedback}

    Required sections:

    Executive Summary
    Architecture
    Results
    Limitations
    References
    """

    llm_result = ask_llm(prompt)

    if llm_result:

        report = llm_result

    else:

        report = f"""
# Executive Summary

This project implements a multi-agent research,
verification and report generation workflow using
LangGraph and PostgreSQL.

The system separates planning, research, analysis,
writing and verification into specialized agents.

# Architecture

User
↓
Planner
↓
Researcher
↓
Analyst
↓
Writer
↓
Verifier
↓
Approve / Revise

LangGraph controls the workflow.

PostgreSQL stores task execution data.

# Results

The system successfully demonstrates:

1. Task decomposition.
2. Specialized agent execution.
3. Agent-to-agent communication.
4. Report generation.
5. Automated verification.
6. Revision routing.

# Limitations

The demonstration mode does not provide a real
benchmark of research accuracy.

Real deployment should connect verified external
research sources and an LLM.

# References

LangGraph documentation:
https://langchain-ai.github.io/langgraph/

PostgreSQL documentation:
https://www.postgresql.org/docs/

Research Question:

{goal}

Evidence:

{evidence}

Analysis:

{analysis}
"""

    duration = time.perf_counter() - start

    finish_subtask(
        subtask_id,
        report
    )

    save_agent_run(
        subtask_id,
        "Writer",
        goal,
        report,
        duration
    )

    save_message(
        task_id,
        "Writer",
        "Verifier",
        "Report generated and sent for verification."
    )

    return {
        "report": report
    }


# ============================================================
# AGENT 5 - VERIFIER
# ============================================================

def verifier_agent(state):

    start = time.perf_counter()

    task_id = state["task_id"]

    subtask_id = create_subtask(
        task_id,
        "Verifier"
    )

    report = state["report"]

    required_sections = [
        "Executive Summary",
        "Architecture",
        "Results",
        "References"
    ]

    missing = []

    for section in required_sections:

        if section.lower() not in report.lower():

            missing.append(section)

    word_count = len(report.split())

    if word_count < 80:

        decision = "REVISE"

        feedback = (
            "The report is too short. "
            "Add more explanation."
        )

    elif missing:

        decision = "REVISE"

        feedback = (
            "Missing sections: "
            + ", ".join(missing)
        )

    elif "http" not in report.lower():

        decision = "REVISE"

        feedback = (
            "Add source URLs to References."
        )

    else:

        decision = "APPROVE"

        feedback = (
            "All deterministic verification "
            "requirements passed."
        )

    result = {
        "decision": decision,
        "feedback": feedback,
        "word_count": word_count
    }

    duration = time.perf_counter() - start

    finish_subtask(
        subtask_id,
        json.dumps(result)
    )

    save_agent_run(
        subtask_id,
        "Verifier",
        report,
        json.dumps(result),
        duration
    )

    return {
        "verification": result,
        "verification_feedback": feedback
    }


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ResearchState(TypedDict, total=False):

    task_id: int

    goal: str

    plan: str

    evidence: str

    analysis: str

    report: str

    verification: dict

    verification_feedback: str

    revisions: int

    start_time: float


# ============================================================
# REVISION NODE
# ============================================================

def revision_node(state):

    revisions = state.get(
        "revisions",
        0
    )

    return {
        "revisions": revisions + 1
    }


# ============================================================
# ROUTING
# ============================================================

def verification_router(state):

    verification = state["verification"]

    decision = verification["decision"]

    revisions = state.get(
        "revisions",
        0
    )

    if decision == "APPROVE":

        return "finish"

    if revisions >= 2:

        return "finish"

    return "revise"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        ResearchState
    )

    graph.add_node(
        "planner",
        planner_agent
    )

    graph.add_node(
        "researcher",
        researcher_agent
    )

    graph.add_node(
        "analyst",
        analyst_agent
    )

    graph.add_node(
        "writer",
        writer_agent
    )

    graph.add_node(
        "verifier",
        verifier_agent
    )

    graph.add_node(
        "revision",
        revision_node
    )

    graph.add_edge(
        START,
        "planner"
    )

    graph.add_edge(
        "planner",
        "researcher"
    )

    graph.add_edge(
        "researcher",
        "analyst"
    )

    graph.add_edge(
        "analyst",
        "writer"
    )

    graph.add_edge(
        "writer",
        "verifier"
    )

    graph.add_conditional_edges(
        "verifier",
        verification_router,
        {
            "revise": "revision",
            "finish": END
        }
    )

    graph.add_edge(
        "revision",
        "writer"
    )

    return graph.compile()


# ============================================================
# EXECUTE WORKFLOW
# ============================================================

def execute_project(goal):

    create_database_tables()

    task_id = create_task(
        goal
    )

    start_time = time.perf_counter()

    initial_state = {

        "task_id": task_id,

        "goal": goal,

        "revisions": 0,

        "start_time": start_time
    }

    graph = build_graph()

    result = graph.invoke(
        initial_state
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    verification = result.get(
        "verification",
        {}
    )

    if verification.get(
        "decision"
    ) == "APPROVE":

        quality = 10

        update_task(
            task_id,
            "completed"
        )

    else:

        quality = 5

        update_task(
            task_id,
            "completed_with_limit"
        )

    save_evaluation(
        task_id,
        result.get(
            "revisions",
            0
        ),
        elapsed,
        quality
    )

    result["elapsed"] = elapsed

    return result


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Research System",
    layout="wide"
)

st.title(
    "🤖 Multi-Agent Research, Verification & Report Generation"
)

st.write(
    "LangGraph + PostgreSQL + Python + Streamlit"
)

st.divider()

st.subheader(
    "Research Question"
)

goal = st.text_area(
    "Enter your research problem:",
    value=(
        "Research the impact of multi-agent "
        "AI systems on software development."
    ),
    height=120
)

run_button = st.button(
    "🚀 Run Multi-Agent System",
    type="primary"
)

if run_button:

    if not goal.strip():

        st.error(
            "Please enter a research question."
        )

    else:

        try:

            with st.spinner(
                "Running LangGraph agents..."
            ):

                result = execute_project(
                    goal.strip()
                )

            st.success(
                "Multi-Agent Workflow Completed!"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Revisions",
                    result.get(
                        "revisions",
                        0
                    )
                )

            with col2:

                st.metric(
                    "Execution Time",
                    f"{result.get('elapsed', 0):.2f}s"
                )

            with col3:

                st.metric(
                    "Verifier",
                    result.get(
                        "verification",
                        {}
                    ).get(
                        "decision",
                        "UNKNOWN"
                    )
                )

            st.divider()

            st.subheader(
                "1️⃣ Planner Agent"
            )

            st.code(
                result.get(
                    "plan",
                    ""
                )
            )

            st.subheader(
                "2️⃣ Researcher Agent"
            )

            st.write(
                result.get(
                    "evidence",
                    ""
                )
            )

            st.subheader(
                "3️⃣ Analyst Agent"
            )

            st.write(
                result.get(
                    "analysis",
                    ""
                )
            )

            st.subheader(
                "4️⃣ Writer Agent"
            )

            st.markdown(
                result.get(
                    "report",
                    ""
                )
            )

            st.subheader(
                "5️⃣ Verifier Agent"
            )

            st.json(
                result.get(
                    "verification",
                    {}
                )
            )

            st.divider()

            st.success(
                "FINAL OUTPUT GENERATED"
            )

        except Exception as error:

            st.error(
                f"Execution error: {error}"
            )