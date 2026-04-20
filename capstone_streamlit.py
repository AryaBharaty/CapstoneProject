"""
capstone_streamlit.py — Study Buddy - Physics Agent
Run: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Study Buddy - Physics", page_icon="🤖", layout="centered")
st.title("🤖 Study Buddy - Physics")
st.caption("AI assistant for B.Tech students to learn physics concepts without hallucination")

# ── Load agent ────────────────────────────────────────────
@st.cache_resource
def load_agent():
    from agent import build_agent
    return build_agent()


try:
    agent_app = load_agent()
    st.success("✅ Agent loaded successfully")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("AI assistant for B.Tech students to learn physics concepts without hallucination")
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    for t in [
        "Newton's Laws of Motion",
        "Kinematics",
        "Work Energy Theorem",
        "Gravitation",
        "Thermodynamics",
        "Electrostatics",
        "Current Electricity",
        "Magnetism",
        "Waves",
        "Modern Physics",
    ]:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Display history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
