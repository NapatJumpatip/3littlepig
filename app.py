"""
Self-Debate Chat — Gemini-powered multi-agent chat with Claude-inspired UI.

═══════════════════════════════════════════════════════════════════
BUG FIX (v5): API Key Widget Disappearance
═══════════════════════════════════════════════════════════════════
ROOT CAUSE:  When st.text_input(key="X") renders on rerun 1 and stores
a value, but then on rerun 2 the widget is conditionally hidden (because
we show "API Key active" instead), Streamlit DELETES session_state["X"]
because the widget no longer exists in the component tree.

FIX:  Use on_change callback to copy the widget value into a SEPARATE
session_state key ("api_key_saved") that is never bound to any widget.
Streamlit's garbage collector can't touch it.
═══════════════════════════════════════════════════════════════════
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
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #FAF9F7;
    --bg-secondary: #F0EEEB;
    --text-primary: #2D2B28;
    --text-secondary: #6B6560;
    --text-muted: #9C9690;
    --accent: #C4703E;
    --accent-light: #E8C9AD;
    --border: #E5E2DD;
    --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
    --radius: 16px;
}

.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}
header[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
    border-bottom: 1px solid var(--border) !important;
}
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
.stMarkdown code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stMarkdown pre {
    background-color: #1E1E1E !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}
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
.thinking-label {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 6px;
    font-size: 0.78rem; font-weight: 500;
    font-family: 'DM Sans', sans-serif; margin-bottom: 6px;
}
.label-generator { background: #E8F0FE; color: #1A56DB; }
.label-critic    { background: #FEF3E2; color: #B45309; }
.label-synthesizer { background: #E6F7ED; color: #0E7A3A; }
.welcome-container {
    text-align: center; padding: 6rem 2rem 2rem;
    max-width: 600px; margin: 0 auto;
}
.welcome-container h1 {
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    font-size: 1.8rem; color: var(--text-primary); margin-bottom: 0.5rem;
}
.welcome-container p {
    color: var(--text-secondary); font-size: 1rem; line-height: 1.6;
}
.welcome-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 20px;
    background: var(--bg-secondary); border: 1px solid var(--border);
    font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1.5rem;
}
.subtle-divider {
    border: none; border-top: 1px solid var(--border); margin: 1rem 0;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════
# "api_key_saved" is the PERSISTENT store — never bound to a widget.
# "api_key_widget" is the transient widget key — Streamlit may GC it.
_DEFAULTS = {
    "all_chats": {},
    "active_chat_id": None,
    "api_key_saved": "",       # ← THE FIX: persistent, widget-independent
    "model_name": "gemini-2.0-flash",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════
# API KEY — single source of truth
# ═════════════════════════════════════════════
def _save_key_from_widget():
    """
    on_change callback for the text_input widget.
    Copies the widget value into the persistent 'api_key_saved' key.
    This runs BEFORE the next rerun, so the value is safe.
    """
    widget_val = st.session_state.get("api_key_widget", "")
    if isinstance(widget_val, str) and widget_val.strip():
        st.session_state.api_key_saved = widget_val.strip()


def get_api_key() -> str | None:
    """
    Return a non-empty API key string or None.
    Priority: st.secrets > session_state.api_key_saved
    """
    # 1. Secrets (Streamlit Cloud / .streamlit/secrets.toml)
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(key, str) and key.strip():
            return key.strip()
    except FileNotFoundError:
        pass

    # 2. Persistent session state (populated by on_change callback)
    key = st.session_state.get("api_key_saved", "")
    if isinstance(key, str) and key.strip():
        return key.strip()

    return None


def validate_api_key(key: str | None) -> bool:
    """Non-empty string ≥ 20 chars."""
    if not key or not isinstance(key, str):
        return False
    return len(key.strip()) >= 20


# ═════════════════════════════════════════════
# CHAT HELPERS
# ═════════════════════════════════════════════
def get_active_messages() -> list[dict]:
    cid = st.session_state.active_chat_id
    if cid and cid in st.session_state.all_chats:
        return st.session_state.all_chats[cid]["messages"]
    return []


def create_new_chat() -> str:
    cid = str(uuid.uuid4())[:8]
    st.session_state.all_chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    st.session_state.active_chat_id = cid
    return cid


def set_chat_title(cid: str, user_msg: str):
    title = user_msg[:50].strip()
    if len(user_msg) > 50:
        title += "…"
    st.session_state.all_chats[cid]["title"] = title


# ═════════════════════════════════════════════
# GEMINI — explicit args, configure per-call
# ═════════════════════════════════════════════
def call_gemini(
    api_key: str,
    model_name: str,
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> str:
    """
    Call Gemini with:
      • genai.configure() immediately before EVERY request attempt
      • Explicit api_key/model_name (no globals)
      • Retry on 429, graceful 400/404/safety handling
    """
    if not messages:
        return "⚠️ No messages to send."

    # Build Gemini-format history once
    gemini_history = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [m["content"]]})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # ── Fresh configure on EVERY attempt ──
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )

            chat = model.start_chat(history=gemini_history[:-1])
            response = chat.send_message(gemini_history[-1]["parts"][0])
            return response.text

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()

            # Rate limit → retry
            if "429" in err_str or "resource" in err_lower or "quota" in err_lower:
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return (
                    "⚠️ **Rate Limit** — API overloaded.\n\n"
                    f"Retried {max_retries}x. Wait ~30s.\n\n```\n{err_str}\n```"
                )

            # Invalid key
            if "api_key_invalid" in err_lower or ("400" in err_str and "api key" in err_lower):
                return (
                    "⚠️ **Invalid API Key** — rejected by Google.\n\n"
                    "Check your key at [aistudio.google.com](https://aistudio.google.com/app/apikey)\n\n"
                    f"```\n{err_str}\n```"
                )

            # Model not found
            if "404" in err_str or "not found" in err_lower:
                return (
                    f"⚠️ **Model Not Found** — `{model_name}` unavailable.\n\n"
                    f"Switch model in sidebar.\n\n```\n{err_str}\n```"
                )

            # Safety block
            if "safety" in err_lower or "block" in err_lower:
                return "⚠️ **Safety Filter** — blocked. Try rephrasing."

            # Unknown
            return f"⚠️ **Error**: `{err_str}`"

    return "⚠️ Unexpected error."


# ═════════════════════════════════════════════
# SELF-DEBATE PIPELINE
# ═════════════════════════════════════════════
def run_self_debate(
    api_key: str,
    model_name: str,
    chat_messages: list[dict],
    status_container,
) -> tuple[str, dict]:
    """Generator → Critic → Synthesizer. All args are explicit snapshots."""
    thinking = {}

    # ── 1. Generator ──
    status_container.update(label="🔵 Step 1/3 — Drafting…", state="running")
    draft = call_gemini(
        api_key=api_key,
        model_name=model_name,
        system_prompt=(
            "คุณคือ Generator Agent ร่างคำตอบเบื้องต้นจากข้อความผู้ใช้ "
            "อ้างอิงบริบทจากประวัติสนทนาทั้งหมด ตอบภาษาไทย ละเอียด "
            "ใช้ Markdown เต็มรูปแบบ"
        ),
        messages=chat_messages,
        temperature=0.7,
    )
    thinking["generator"] = draft

    if draft.startswith("⚠️"):
        thinking["critic"] = "⏭️ Skipped — Generator failed."
        thinking["synthesizer"] = draft
        status_container.update(label="⚠️ Error", state="error")
        return draft, thinking

    # ── 2. Critic ──
    status_container.update(label="🟠 Step 2/3 — Analyzing…", state="running")
    critic_msgs = chat_messages + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": "วิเคราะห์คำตอบข้างต้น: จุดอ่อน ข้อผิดพลาด ข้อเสนอแนะ"},
    ]
    critique = call_gemini(
        api_key=api_key,
        model_name=model_name,
        system_prompt=(
            "คุณคือ Critic Agent วิเคราะห์จุดอ่อน ข้อผิดพลาด ข้อมูลที่ขาด "
            "เสนอแนะการปรับปรุง ตอบภาษาไทย"
        ),
        messages=critic_msgs,
        temperature=0.4,
    )
    thinking["critic"] = critique

    critique_for_synth = critique
    if critique.startswith("⚠️"):
        critique_for_synth = "ไม่มีข้อเสนอแนะ (Critic ไม่สามารถวิเคราะห์ได้)"

    # ── 3. Synthesizer ──
    status_container.update(label="🟢 Step 3/3 — Synthesizing…", state="running")
    synth_msgs = chat_messages + [
        {"role": "assistant", "content": f"**Draft:**\n{draft}\n\n**Critique:**\n{critique_for_synth}"},
        {"role": "user", "content": "สังเคราะห์คำตอบสุดท้ายที่ดีที่สุด"},
    ]
    final = call_gemini(
        api_key=api_key,
        model_name=model_name,
        system_prompt=(
            "คุณคือ Synthesizer Agent สังเคราะห์คำตอบสุดท้ายจาก Draft+Critique "
            "รักษาจุดแข็ง แก้ตามข้อเสนอแนะ ตอบภาษาไทย Markdown เต็มรูปแบบ "
            "ห้ามกล่าวถึง Draft/Critique ตอบเสมือนคำตอบสมบูรณ์โดยตรง"
        ),
        messages=synth_msgs,
        temperature=0.5,
    )
    thinking["synthesizer"] = final

    if final.startswith("⚠️"):
        final = f"{final}\n\n---\n**Fallback (Draft):**\n{draft}"

    status_container.update(label="✅ Done", state="complete")
    return final, thinking


# ═════════════════════════════════════════════
# RENDER THINKING (reusable)
# ═════════════════════════════════════════════
def render_thinking(thinking: dict):
    with st.expander("💭 How Gemini thought about this", expanded=False):
        st.markdown('<span class="thinking-label label-generator">🔵 Generator</span>', unsafe_allow_html=True)
        st.markdown(thinking.get("generator", "—"))
        st.markdown("---")
        st.markdown('<span class="thinking-label label-critic">🟠 Critic</span>', unsafe_allow_html=True)
        st.markdown(thinking.get("critic", "—"))
        st.markdown("---")
        st.markdown('<span class="thinking-label label-synthesizer">🟢 Synthesizer</span>', unsafe_allow_html=True)
        st.markdown(thinking.get("synthesizer", "—"))


# ═════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 Self-Debate Chat")
    st.caption(f"Powered by `{st.session_state.model_name}`")
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Model ──
    MODEL_OPTIONS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
    ]
    _cur = MODEL_OPTIONS.index(st.session_state.model_name) if st.session_state.model_name in MODEL_OPTIONS else 0
    sel = st.selectbox("Model", MODEL_OPTIONS, index=_cur, label_visibility="collapsed")
    if sel != st.session_state.model_name:
        st.session_state.model_name = sel
        st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── New Chat ──
    if st.button("＋  New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── API Key ──────────────────────────────
    # THE FIX: Always show ONE of two states, but the persistent
    # value lives in api_key_saved, not in a widget key.
    # ─────────────────────────────────────────
    current_key = get_api_key()
    has_valid_key = validate_api_key(current_key)

    if not has_valid_key:
        st.markdown("##### 🔑 Gemini API Key")
        st.text_input(
            "API Key",
            type="password",
            placeholder="Paste your Gemini API key here",
            label_visibility="collapsed",
            key="api_key_widget",            # transient widget key
            on_change=_save_key_from_widget,  # copies to api_key_saved
        )
        # Hint
        saved = st.session_state.get("api_key_saved", "")
        if saved and not validate_api_key(saved):
            st.warning("Key too short. Gemini keys are ~39 chars.")
    else:
        st.success("API Key active", icon=":material/check_circle:")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("✕", key="clear_key", help="Clear API key"):
                st.session_state.api_key_saved = ""
                st.rerun()
        with col1:
            if st.checkbox("Show key", value=False, key="show_key_cb"):
                masked = current_key[:8] + "•" * max(0, len(current_key) - 12) + current_key[-4:]
                st.code(masked, language=None)

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Recent Chats ──
    st.markdown("##### Recent Chats")
    if st.session_state.all_chats:
        for cid in sorted(
            st.session_state.all_chats,
            key=lambda k: st.session_state.all_chats[k]["created_at"],
            reverse=True,
        ):
            data = st.session_state.all_chats[cid]
            icon = "●" if cid == st.session_state.active_chat_id else "○"
            if st.button(f"{icon}  {data['title']}", key=f"c_{cid}", use_container_width=True):
                st.session_state.active_chat_id = cid
                st.rerun()
    else:
        st.caption("No chats yet.")

    # ── Debug info ──
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
    with st.expander("🔧 Debug", expanded=False):
        st.caption(f"api_key_saved length: {len(st.session_state.get('api_key_saved', ''))}")
        st.caption(f"get_api_key() returns: {'✅ key present' if get_api_key() else '❌ None'}")
        st.caption(f"validate: {validate_api_key(get_api_key())}")
        st.caption(f"model: {st.session_state.model_name}")
        st.caption(f"active_chat: {st.session_state.active_chat_id}")


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════
API_KEY = get_api_key()
HAS_KEY = validate_api_key(API_KEY)

# Welcome
if st.session_state.active_chat_id is None or st.session_state.active_chat_id not in st.session_state.all_chats:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-badge">🧠 Self-Debate System</div>
        <h1>What can I help you with?</h1>
        <p>Every answer goes through <strong>Generate → Critique → Synthesize</strong></p>
    </div>
    """, unsafe_allow_html=True)
    if not HAS_KEY:
        st.info("👈 Enter your Gemini API key in the sidebar to start.")

# History
for msg in get_active_messages():
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "thinking" in msg:
            render_thinking(msg["thinking"])

# ═════════════════════════════════════════════
# CHAT INPUT
# ═════════════════════════════════════════════
if prompt := st.chat_input("Message Self-Debate Chat…", disabled=not HAS_KEY):

    # Gate — re-resolve at submission time
    _submit_key = get_api_key()
    if not validate_api_key(_submit_key):
        st.error(
            "⚠️ **API Key missing.** Please enter a valid key in the sidebar.\n\n"
            f"Debug: saved=`{len(st.session_state.get('api_key_saved', ''))}` chars"
        )
        st.stop()

    # Snapshot immutable locals for the pipeline
    _key: str = _submit_key  # type: ignore
    _model: str = st.session_state.model_name

    # Auto-create chat
    if st.session_state.active_chat_id is None or st.session_state.active_chat_id not in st.session_state.all_chats:
        create_new_chat()

    cid = st.session_state.active_chat_id
    store = st.session_state.all_chats[cid]

    if not store["messages"]:
        set_chat_title(cid, prompt)

    store["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        with st.status("🧠 Thinking…", expanded=True) as status:
            final_answer, thinking = run_self_debate(
                api_key=_key,
                model_name=_model,
                chat_messages=store["messages"],
                status_container=status,
            )
        st.markdown(final_answer)
        render_thinking(thinking)

    store["messages"].append({
        "role": "assistant",
        "content": final_answer,
        "thinking": thinking,
    })
