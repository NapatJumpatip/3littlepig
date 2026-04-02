"""
Self-Debate Chat — A Claude-inspired Streamlit chat app
powered by Gemini 1.5 Pro with a 3-agent thinking pipeline.
"""

import streamlit as st
import google.generativeai as genai
import time
import uuid
from datetime import datetime

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Debate Chat",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS — Claude-inspired aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #FAF9F7;
    --bg-secondary: #F0EEEB;
    --bg-user-msg: #EDE9E3;
    --bg-ai-msg: transparent;
    --text-primary: #2D2B28;
    --text-secondary: #6B6560;
    --text-muted: #9C9690;
    --accent: #C4703E;
    --accent-light: #E8C9AD;
    --border: #E5E2DD;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
    --radius: 16px;
}

/* ── Global Overrides ── */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Chat Messages ── */
.stChatMessage {
    background-color: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
    max-width: 760px !important;
    margin: 0 auto !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stChatMessage [data-testid="chatAvatarIcon-user"] {
    background-color: var(--accent) !important;
}

.stChatMessage [data-testid="chatAvatarIcon-assistant"] {
    background-color: var(--text-primary) !important;
}

/* ── Chat Input ── */
.stChatInput {
    max-width: 760px !important;
    margin: 0 auto !important;
}

.stChatInput > div {
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: white !important;
    box-shadow: var(--shadow-md) !important;
    transition: border-color 0.2s ease !important;
}

.stChatInput > div:focus-within {
    border-color: var(--accent) !important;
}

.stChatInput textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--text-primary) !important;
}

/* ── Status / Expander (Thinking Process) ── */
.stExpander {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stExpander summary span {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* ── Code Blocks ── */
.stMarkdown code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stMarkdown pre {
    background-color: #1E1E1E !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1px solid var(--border) !important;
    background: white !important;
    color: var(--text-primary) !important;
    transition: all 0.2s ease !important;
    padding: 0.4rem 1rem !important;
}

.stButton > button:hover {
    background: var(--bg-secondary) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Sidebar Chat History Items ── */
.chat-history-item {
    padding: 10px 14px;
    margin: 4px 0;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.15s ease;
    font-size: 0.88rem;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border: 1px solid transparent;
}

.chat-history-item:hover {
    background: rgba(196, 112, 62, 0.08);
    border-color: var(--accent-light);
}

.chat-history-item.active {
    background: rgba(196, 112, 62, 0.12);
    border-color: var(--accent);
    font-weight: 500;
}

/* ── Thinking Step Labels ── */
.thinking-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 6px;
}

.label-generator { background: #E8F0FE; color: #1A56DB; }
.label-critic { background: #FEF3E2; color: #B45309; }
.label-synthesizer { background: #E6F7ED; color: #0E7A3A; }

/* ── Welcome Screen ── */
.welcome-container {
    text-align: center;
    padding: 6rem 2rem 2rem;
    max-width: 600px;
    margin: 0 auto;
}

.welcome-container h1 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1.8rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.welcome-container p {
    color: var(--text-secondary);
    font-size: 1rem;
    line-height: 1.6;
}

.welcome-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
}

/* ── Divider ── */
.subtle-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}

/* hide default streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
def init_session():
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}  # {chat_id: {title, messages, created_at}}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gemini-1.5-flash"

init_session()


def get_active_messages():
    cid = st.session_state.active_chat_id
    if cid and cid in st.session_state.all_chats:
        return st.session_state.all_chats[cid]["messages"]
    return []


def create_new_chat():
    cid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().strftime("%H:%M"),
    }
    st.session_state.active_chat_id = cid
    return cid


def set_chat_title(cid, user_msg):
    """Set the chat title from the first user message."""
    title = user_msg[:50].strip()
    if len(user_msg) > 50:
        title += "…"
    st.session_state.all_chats[cid]["title"] = title


# ─────────────────────────────────────────────
# API Key Resolution
# ─────────────────────────────────────────────
def resolve_api_key():
    # 1. st.secrets
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass
    # 2. session state (from sidebar input)
    if st.session_state.api_key:
        return st.session_state.api_key
    return None


# ─────────────────────────────────────────────
# Gemini Helpers
# ─────────────────────────────────────────────
def call_gemini(api_key: str, system_prompt: str, messages: list[dict], temperature: float = 0.7) -> str:
    """Call Gemini with retry on 429 and robust error handling."""
    genai.configure(api_key=api_key)

    model_name = st.session_state.get("model_name", "gemini-1.5-flash")

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature),
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    )

    # Build Gemini-style conversation history
    gemini_history = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [m["content"]]})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            chat = model.start_chat(history=gemini_history[:-1])
            response = chat.send_message(gemini_history[-1]["parts"][0])
            return response.text
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "resource" in err or "quota" in err:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    return f"⚠️ Rate limit exceeded. Please wait a moment and try again.\n\n`{e}`"
            elif "safety" in err or "block" in err:
                return "⚠️ The response was blocked by Gemini's safety filters. Please try rephrasing your message."
            elif "404" in err or "not found" in err:
                return (
                    f"⚠️ Model `{model_name}` not found. "
                    "Please select a different model in the sidebar.\n\n"
                    f"`{e}`"
                )
            else:
                return f"⚠️ An error occurred: `{e}`"
    return "⚠️ Unexpected error."


def run_self_debate(api_key: str, chat_messages: list[dict], status_container) -> tuple[str, dict]:
    """
    Execute the 3-agent pipeline:
      1. Generator → Draft
      2. Critic → Analysis
      3. Synthesizer → Final Answer
    Returns (final_answer, thinking_steps).
    """
    thinking = {}
    _error_prefix = "⚠️ **Agent Error**"

    # ── Step 1: Generator ──
    status_container.update(label="🔵 Generating initial draft…", state="running")
    generator_system = (
        "คุณคือ Generator Agent คุณมีหน้าที่ร่างคำตอบเบื้องต้นจากข้อความของผู้ใช้ "
        "โดยอ้างอิงบริบทจากประวัติการสนทนาทั้งหมด ตอบเป็นภาษาไทย ให้ละเอียดและครอบคลุม "
        "ใช้ Markdown ได้เต็มรูปแบบ (ตาราง, โค้ด, หัวข้อ)"
    )
    try:
        draft = call_gemini(api_key, generator_system, chat_messages, temperature=0.7)
    except Exception as e:
        draft = f"{_error_prefix} — Generator failed: `{e}`"
    thinking["generator"] = draft

    # If Generator failed critically, skip remaining steps
    if draft.startswith("⚠️"):
        thinking["critic"] = "⏭️ Skipped — Generator did not produce a valid draft."
        thinking["synthesizer"] = draft
        status_container.update(label="⚠️ Completed with errors", state="error")
        return draft, thinking

    # ── Step 2: Critic ──
    status_container.update(label="🟠 Analyzing draft for improvements…", state="running")
    critic_system = (
        "คุณคือ Critic Agent คุณจะได้รับร่างคำตอบจาก Generator "
        "ให้วิเคราะห์จุดอ่อน ข้อผิดพลาด ข้อมูลที่ขาดหาย หรือส่วนที่อาจทำให้เข้าใจผิด "
        "และเสนอแนะการปรับปรุงอย่างชัดเจน ตอบเป็นภาษาไทย"
    )
    critic_messages = chat_messages + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": "วิเคราะห์คำตอบข้างต้น: จุดอ่อน ข้อผิดพลาด และข้อเสนอแนะ"},
    ]
    try:
        critique = call_gemini(api_key, critic_system, critic_messages, temperature=0.4)
    except Exception as e:
        critique = f"{_error_prefix} — Critic failed: `{e}`"
    thinking["critic"] = critique

    # If Critic failed, Synthesizer works with draft alone
    if critique.startswith("⚠️"):
        critique_for_synth = "ไม่มีข้อเสนอแนะเพิ่มเติม (Critic agent ไม่สามารถวิเคราะห์ได้)"
    else:
        critique_for_synth = critique

    # ── Step 3: Synthesizer ──
    status_container.update(label="🟢 Synthesizing final answer…", state="running")
    synth_system = (
        "คุณคือ Synthesizer Agent คุณจะได้รับ Draft และ Critique "
        "ให้สังเคราะห์คำตอบสุดท้ายที่ดีที่สุด โดยรักษาจุดแข็งของ Draft "
        "และแก้ไขตามข้อเสนอแนะของ Critic ตอบเป็นภาษาไทย "
        "ใช้ Markdown เต็มรูปแบบ ห้ามกล่าวถึง Draft หรือ Critique ในคำตอบ "
        "ตอบเสมือนเป็นคำตอบสุดท้ายที่สมบูรณ์โดยตรง"
    )
    synth_messages = chat_messages + [
        {"role": "assistant", "content": f"**Draft:**\n{draft}\n\n**Critique:**\n{critique_for_synth}"},
        {"role": "user", "content": "สังเคราะห์คำตอบสุดท้ายที่ดีที่สุดจาก Draft และ Critique"},
    ]
    try:
        final = call_gemini(api_key, synth_system, synth_messages, temperature=0.5)
    except Exception as e:
        final = f"{_error_prefix} — Synthesizer failed: `{e}`\n\n---\n\n**Fallback (Generator Draft):**\n{draft}"
    thinking["synthesizer"] = final

    status_container.update(label="✅ Done thinking", state="complete")
    return final, thinking


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Self-Debate Chat")
    st.caption(f"Powered by {st.session_state.model_name}")
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Model selector
    model_options = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash",
    ]
    current_idx = model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
    selected_model = st.selectbox(
        "Model",
        model_options,
        index=current_idx,
        label_visibility="collapsed",
    )
    if selected_model != st.session_state.model_name:
        st.session_state.model_name = selected_model
        st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # New Chat button
    if st.button("＋  New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # API Key fallback input
    api_key = resolve_api_key()
    if not api_key:
        st.markdown("##### 🔑 API Key")
        key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your Gemini API key",
            label_visibility="collapsed",
        )
        if key_input:
            st.session_state.api_key = key_input
            st.rerun()
    else:
        st.success("API Key active", icon=":material/check_circle:")

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Recent Chats list
    st.markdown("##### Recent Chats")
    if st.session_state.all_chats:
        # Show newest first
        sorted_ids = sorted(
            st.session_state.all_chats.keys(),
            key=lambda k: st.session_state.all_chats[k]["created_at"],
            reverse=True,
        )
        for cid in sorted_ids:
            chat_data = st.session_state.all_chats[cid]
            is_active = cid == st.session_state.active_chat_id
            label = f"{'●' if is_active else '○'}  {chat_data['title']}"
            if st.button(label, key=f"chat_{cid}", use_container_width=True):
                st.session_state.active_chat_id = cid
                st.rerun()
    else:
        st.caption("No chats yet. Start one!")


# ─────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────
api_key = resolve_api_key()

# Welcome screen when no active chat
if st.session_state.active_chat_id is None or st.session_state.active_chat_id not in st.session_state.all_chats:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-badge">🧠 Self-Debate System</div>
        <h1>What can I help you with?</h1>
        <p>
            Every answer goes through a 3-step thinking pipeline —<br>
            <strong>Generate → Critique → Synthesize</strong> — for higher quality responses.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started.")

# Display existing messages
messages = get_active_messages()
for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])
        # Show thinking steps if available
        if msg["role"] == "assistant" and "thinking" in msg:
            with st.expander("💭 How Gemini thought about this", expanded=False):
                t = msg["thinking"]
                st.markdown('<span class="thinking-label label-generator">🔵 Generator — Initial Draft</span>', unsafe_allow_html=True)
                st.markdown(t.get("generator", ""), unsafe_allow_html=False)
                st.markdown("---")
                st.markdown('<span class="thinking-label label-critic">🟠 Critic — Analysis</span>', unsafe_allow_html=True)
                st.markdown(t.get("critic", ""), unsafe_allow_html=False)
                st.markdown("---")
                st.markdown('<span class="thinking-label label-synthesizer">🟢 Synthesizer — Final</span>', unsafe_allow_html=True)
                st.markdown(t.get("synthesizer", ""), unsafe_allow_html=False)

# ─────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Message Self-Debate Chat…", disabled=not api_key):
    # Auto-create chat if none active
    if st.session_state.active_chat_id is None or st.session_state.active_chat_id not in st.session_state.all_chats:
        create_new_chat()

    cid = st.session_state.active_chat_id
    chat_store = st.session_state.all_chats[cid]

    # Set title from first message
    if not chat_store["messages"]:
        set_chat_title(cid, prompt)

    # Append user message
    chat_store["messages"].append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Run the self-debate pipeline
    with st.chat_message("assistant", avatar="🧠"):
        with st.status("🧠 Thinking…", expanded=True) as status:
            final_answer, thinking = run_self_debate(
                api_key,
                chat_store["messages"],
                status,
            )

        # Display final answer
        st.markdown(final_answer)

        # Show thinking process
        with st.expander("💭 How Gemini thought about this", expanded=False):
            st.markdown('<span class="thinking-label label-generator">🔵 Generator — Initial Draft</span>', unsafe_allow_html=True)
            st.markdown(thinking.get("generator", ""), unsafe_allow_html=False)
            st.markdown("---")
            st.markdown('<span class="thinking-label label-critic">🟠 Critic — Analysis</span>', unsafe_allow_html=True)
            st.markdown(thinking.get("critic", ""), unsafe_allow_html=False)
            st.markdown("---")
            st.markdown('<span class="thinking-label label-synthesizer">🟢 Synthesizer — Final</span>', unsafe_allow_html=True)
            st.markdown(thinking.get("synthesizer", ""), unsafe_allow_html=False)

    # Save assistant message with thinking
    chat_store["messages"].append({
        "role": "assistant",
        "content": final_answer,
        "thinking": thinking,
    })
