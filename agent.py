"""
agent.py — Study Buddy Physics Agent (LangGraph)
Exports: build_agent() → compiled LangGraph app
"""

import re
from typing import TypedDict, List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb

load_dotenv()

# ── Constants ─────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


# ── Knowledge Base Documents ─────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Newton's Laws of Motion",
        "text": (
            "Newton's Laws of Motion describe the relationship between a body and the forces acting on it.\n"
            "The First Law states that a body remains at rest or in uniform motion unless acted upon by an external force.\n"
            "The Second Law states that force equals mass times acceleration (F = ma), meaning acceleration is proportional to force.\n"
            "The Third Law states that for every action, there is an equal and opposite reaction.\n"
            "These laws are fundamental to classical mechanics and are used to analyze motion in everyday systems."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Kinematics",
        "text": (
            "Kinematics is the study of motion without considering forces.\n"
            "Important equations include v = u + at, s = ut + 1/2 at^2, and v^2 = u^2 + 2as.\n"
            "Where u is initial velocity, v is final velocity, a is acceleration, and s is displacement.\n"
            "These equations are valid only for constant acceleration.\n"
            "Kinematics helps describe motion in straight lines and is widely used in physics problems."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Work Energy Theorem",
        "text": (
            "The work-energy theorem states that the work done on an object is equal to the change in its kinetic energy.\n"
            "Work is defined as force multiplied by displacement (W = F*d).\n"
            "Kinetic energy is given by KE = 1/2 mv^2.\n"
            "If positive work is done, kinetic energy increases.\n"
            "If negative work is done, kinetic energy decreases.\n"
            "This theorem simplifies solving problems involving motion and forces."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Gravitation",
        "text": (
            "Gravitation is the force of attraction between two masses.\n"
            "Newton's law of gravitation states F = G(m1*m2)/r^2.\n"
            "G is the gravitational constant.\n"
            "This force is responsible for planetary motion.\n"
            "Near Earth, acceleration due to gravity is approximately 9.8 m/s^2.\n"
            "Gravitation explains orbits, tides, and falling objects."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Thermodynamics",
        "text": (
            "Thermodynamics deals with heat, work, and energy.\n"
            "The First Law states energy cannot be created or destroyed.\n"
            "The Second Law states heat flows from hot to cold bodies.\n"
            "Important quantities include internal energy, entropy, and temperature.\n"
            "Thermodynamics is applied in engines, refrigerators, and many real-world systems."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Electrostatics",
        "text": (
            "Electrostatics studies charges at rest.\n"
            "Coulomb's law states force between charges is proportional to product of charges "
            "and inversely proportional to square of distance.\n"
            "F = k(q1*q2)/r^2.\n"
            "Electric field is defined as force per unit charge.\n"
            "This concept is important for understanding electric forces."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Current Electricity",
        "text": (
            "Current electricity deals with flow of charges.\n"
            "Ohm's law states V = IR.\n"
            "Where V is voltage, I is current, and R is resistance.\n"
            "Power is given by P = VI.\n"
            "This is important for circuits and electrical systems."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Magnetism",
        "text": (
            "Magnetism is related to moving charges.\n"
            "A current-carrying conductor produces a magnetic field.\n"
            "The direction is given by right-hand rule.\n"
            "Magnetic force on a moving charge is given by F = qvB sin(theta)."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Waves",
        "text": (
            "Waves transfer energy without transferring matter.\n"
            "Types include transverse and longitudinal waves.\n"
            "Wave speed is v = f*lambda.\n"
            "Where f is frequency and lambda is wavelength.\n"
            "Examples include sound waves and light waves."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Modern Physics",
        "text": (
            "Modern physics includes quantum mechanics and relativity.\n"
            "Einstein's equation E = mc^2 relates mass and energy.\n"
            "Quantum theory explains atomic structure.\n"
            "This field explains behavior at microscopic level."
        ),
    },
]


# ── State Schema ──────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    calc_result: str
    rewritten_query: str


# ══════════════════════════════════════════════════════════
#  build_agent()  — one-time setup, returns compiled graph
# ══════════════════════════════════════════════════════════
def build_agent():
    """Initialise LLM, embedder, ChromaDB, and return compiled LangGraph app."""

    # ── Models ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── ChromaDB collection ───────────────────────────────
    client = chromadb.Client()
    # Delete old collection if it exists so we start fresh
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass

    collection = client.create_collection("capstone_kb")

    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    # ═══════════════════ NODE FUNCTIONS ═══════════════════

    def memory_node(state: CapstoneState) -> dict:
        """Append user question to conversation history (sliding window of 6)."""
        msgs = list(state.get("messages", []))
        msgs.append({"role": "user", "content": state["question"]})
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(state: CapstoneState) -> dict:
        """Classify the question into retrieve / memory_only / tool."""
        question = state["question"].lower()

        # 1. STRICT TOOL DETECTION (highest priority)
        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", question):
            return {"route": "tool"}
        if "calculate" in question:
            return {"route": "tool"}

        # 2. MEMORY DETECTION
        if "my name is" in question or "what is my name" in question:
            return {"route": "memory_only"}
        if any(x in question for x in ["my", "i said", "remember"]):
            return {"route": "memory_only"}

        # 3. LLM fallback
        prompt = f"""You are a router for a Physics Study Assistant.

Options:
- retrieve  -> physics concept questions
- memory_only -> conversation / personal questions
- tool -> numerical calculations

Question: {question}

Reply with ONE word: retrieve / memory_only / tool"""

        response = llm.invoke(prompt)
        decision = response.content.strip().lower()

        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        else:
            decision = "retrieve"

        return {"route": decision}

    def rewrite_node(state: CapstoneState) -> dict:
        """Rewrite the question for better KB retrieval."""
        question = state["question"]

        prompt = f"""Rewrite the question for better search in a Physics knowledge base.

STRICT RULES:
- Do NOT add new information
- Do NOT assume anything not in the question
- Only clarify wording
- Keep meaning EXACTLY the same
- Keep it short (1 sentence)
- Keep the intent (explain/define/calculate) unchanged

Question: {question}

Rewritten:"""

        response = llm.invoke(prompt)
        rewritten = response.content.strip()

        print(f"[Original]  {state['question']}")
        print(f"[Rewritten] {rewritten}")

        return {"rewritten_query": rewritten}

    def retrieval_node(state: CapstoneState) -> dict:
        """Retrieve top-5 chunks from ChromaDB."""
        query = state.get("rewritten_query", "")
        if len(query) < 5:
            query = state["question"]

        q_emb = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=5)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        """No retrieval needed (memory-only path)."""
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        """Simple calculator — evaluates safe math expressions."""
        question = state["question"]
        try:
            if re.fullmatch(r"\s*[0-9\.\+\-\*\/\(\)\s]+\s*", question):
                result = eval(question)  # safe: only digits & operators
                return {"tool_result": f"Calculated result: {result}"}
            else:
                return {"tool_result": "No valid calculation expression found."}
        except Exception as e:
            return {"tool_result": f"Calculation error: {e}"}

    def answer_node(state: CapstoneState) -> dict:
        """Generate an answer using the LLM, grounded in retrieved context."""
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        # For tool results, return directly
        if tool_result:
            return {"answer": tool_result}

        # Build context
        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        context = "\n\n".join(context_parts)

        # System prompt
        if context:
            system_content = f"""You are a Physics Study Assistant for B.Tech students.

Rules:
- Answer ONLY using the given context
- Do NOT invent formulas
- If not found, say: I don't have that information in my knowledge base
- Explain concepts clearly

{context}"""
        else:
            system_content = "You are a helpful assistant. Answer based on the conversation history."

        # If retrying after eval failure
        if eval_retries > 0:
            system_content += (
                "\n\nIMPORTANT: Your previous answer did not meet quality standards. "
                "Answer using ONLY information explicitly stated in the context above."
            )

        # Build LangChain message list
        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    def eval_node(state: CapstoneState) -> dict:
        """Score faithfulness of the answer against retrieved context."""
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:1500]
        retries = state.get("eval_retries", 0)

        # Skip eval for tool / memory_only routes
        if state.get("route") in ("tool", "memory_only"):
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            score = 0.5

        gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️"
        print(f"  [eval] Faithfulness: {score:.2f} {gate}")
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        """Append assistant answer to conversation history."""
        msgs = list(state.get("messages", []))
        msgs.append({"role": "assistant", "content": state["answer"]})
        return {"messages": msgs}

    # ═══════════════════ ROUTING LOGIC ════════════════════

    def route_decision(state: CapstoneState) -> str:
        """After router_node: decide which path to take."""
        route = state.get("route", "retrieve")
        if route == "tool":
            return "tool"
        if route == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        """After eval_node: retry answer or save and finish."""
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"  # retry

    # ═══════════════════ GRAPH ASSEMBLY ═══════════════════

    graph = StateGraph(CapstoneState)

    # Add nodes
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    # Entry point
    graph.set_entry_point("memory")

    # Fixed edges
    graph.add_edge("memory", "router")
    graph.add_edge("rewrite", "retrieve")

    # Router → conditional branch
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "rewrite", "skip": "skip", "tool": "tool"},
    )

    # All paths converge at answer
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")

    # Eval gate
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )
    graph.add_edge("save", END)

    # Compile with memory checkpointer
    app = graph.compile(checkpointer=MemorySaver())
    return app