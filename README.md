# Study Buddy – Physics

An Agentic AI application designed to help B.Tech students learn and revise physics concepts with accuracy and reliability.

---

## Overview

Study Buddy – Physics uses a **Retrieval-Augmented Generation (RAG)** pipeline with a **LangGraph-based agent** to provide context-grounded answers without hallucination. The assistant retrieves knowledge from a curated physics knowledge base, maintains conversation memory, performs basic calculations, and self-evaluates responses for faithfulness.

---

## Problem Statement

Students often need conceptual clarity in physics outside classroom hours, but traditional AI assistants tend to hallucinate or provide unreliable answers.

This project solves that by building a domain-specific AI assistant that:

- Answers strictly from a predefined knowledge base
- Avoids hallucinations through self-evaluation
- Maintains conversational context across a session
- Provides reliable explanations for academic use

---

## Key Features

| Feature | Description |
|---|---|
| **RAG Pipeline (ChromaDB)** | Retrieves relevant physics concepts from a structured knowledge base |
| **LangGraph Agent Architecture** | Multi-node pipeline: routing, retrieval, reasoning, evaluation, and memory |
| **Conversation Memory** | Maintains session context using `MemorySaver` and `thread_id` |
| **Tool Integration** | Supports numerical calculations through a dedicated tool node |
| **Self-Evaluation** | Faithfulness check with automatic retry if hallucination is detected |
| **Streamlit UI** | Interactive chat interface for real-time usage |

---

## Architecture

```
User Input
    ↓
memory_node
    ↓
router_node
    ↓
retrieval_node / tool_node / skip_node
    ↓
answer_node
    ↓
eval_node
    ↓
save_node
    ↓
Response
```

---

## Tech Stack

- **Python**
- **Streamlit** – Frontend UI
- **LangGraph** – Agent orchestration
- **ChromaDB** – Vector database for RAG
- **SentenceTransformers** – `all-MiniLM-L6-v2` for embeddings
- **Groq LLM** – `llama-3.3-70b-versatile`
- **dotenv** – Environment variable management

---

## Project Structure

```
.
├── agent.py                 # Core agent logic (LangGraph pipeline)
├── capstone_streamlit.py    # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Knowledge Base

The knowledge base covers **10 physics topics**, each document focused on a single concept:

1. Newton's Laws
2. Kinematics
3. Work-Energy Theorem
4. Gravitation
5. Thermodynamics
6. Electrostatics
7. Current Electricity
8. Magnetism
9. Waves
10. Modern Physics

---

## Evaluation

### Internal Evaluation
- **Faithfulness scoring** (0.0–1.0)
- **Retry mechanism** if score falls below 0.7

### RAGAS Metrics (Baseline)
- Faithfulness
- Answer Relevancy
- Context Precision

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/AryaBharaty/CapstoneProject
cd CapstoneProject
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add environment variables**

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Running the Application

```bash
streamlit run capstone_streamlit.py
```

---

## Usage

Ask physics-related questions, perform simple calculations, or continue multi-turn conversations — memory is retained per session.

**Example Queries:**

```
"Explain Newton's second law"
"What is kinetic energy?"
"2 * 9.8 * 5"
"What did I ask before?"
```

---

## Limitations

- Knowledge is limited to the predefined physics dataset
- No real-time data or web access
- Basic calculator only (no advanced symbolic math)

---

## Future Improvements

- Expand knowledge base with more topics
- Add diagram and image support
- Improve tool capabilities (advanced physics solver)
- Integrate external APIs for dynamic data
- Add multilingual support

---

This project was developed as part of an Agentic AI Capstone Project.
