"""
Self-Debate Chat — A Claude-inspired Streamlit chat app
powered by Gemini with a 3-agent thinking pipeline.

Key architecture: API key flows through a SINGLE function `get_api_key()`
that checks secrets → session_state in a deterministic order. The key is
passed as an explicit string argument to every Gemini call — never read
from a global or closure mid-pipeline.
"""

import streamlit as st
import google.generativeai as genai
import time
import uuid
from datetime import datetime

# ─────────────────────────────────────────────
# Page Config (must be first Streamlit call)
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

.subtle-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# SESSION STATE — single init block
# ═════════════════════════════════════════════
_DEFAULTS = {
    "all_chats": {},
    "active_chat_id": None,
    "api_key_input": "",       # raw text from sidebar input widget
    "model_name": "gemini-1.5-flash",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════
# API KEY — single source of truth
# ═════════════════════════════════════════════
def get_api_key() -> str | None:
    """
    Return a validated, non-empty API key string or None.
    Priority: st.secrets > sidebar input (stored in session_state).
    """
    # 1. Try secrets (Streamlit Cloud / .streamlit/secrets.toml)
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(key, str) and key.strip():
            return key.strip()
    except FileNotFoundError:
        pass

    # 2. Try session state (from sidebar text_input)
    key = st.session_state.get("api_key_input", "")
    if isinstance(key, str) and key.strip():
        return key.strip()

    return None


def validate_api_key(key: str) -> bool:
    """Quick sanity check — non-empty string of reasonable length."""
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
# GEMINI CALL — key is an explicit argument
# ═════════════════════════════════════════════
def call_gemini(
    api_key: str,
    model_name: str,
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.7,
) -> str:
    """
    Single Gemini call with:
      - genai.configure() right before every request
      - explicit api_key & model_name args (no globals)
      - retry on 429, graceful handling of 400/404/safety
    """
    # ── Configure immediately before use ──
    genai.configure(api_key=api_key)

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

    # Build Gemini conversation history
    gemini_history = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [m["content"]]})

    if not gemini_history:
        return "⚠️ No messages to send."

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Re-configure on every retry to ensure key is fresh
            genai.configure(api_key=api_key)

            chat = model.start_chat(history=gemini_history[:-1])
            response = chat.send_message(gemini_history[-1]["parts"][0])
            return response.text

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()

            # ── Rate limit (429) — retry with backoff ──
            if "429" in err_str or "resource" in err_lower or "quota" in err_lower:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    return (
                        "⚠️ **Rate Limit** — Gemini API is overloaded.\n\n"
                        f"Retried {max_retries} times. Please wait ~30s and try again.\n\n"
                        f"```\n{err_str}\n```"
                    )
                continue  # retry

            # ── Invalid API key (400) ──
            if "api_key_invalid" in err_lower or (
                "400" in err_str and "api key" in err_lower
            ):
                return (
                    "⚠️ **Invalid API Key** — Google rejected the key.\n\n"
                    "Please check:\n"
                    "1. The key is a valid Gemini API key from "
                    "[aistudio.google.com](https://aistudio.google.com/app/apikey)\n"
                    "2. No extra spaces or newlines were copied\n"
                    "3. The key has not been revoked or expired\n\n"
                    f"```\n{err_str}\n```"
                )

            # ── Model not found (404) ──
            if "404" in err_str or "not found" in err_lower:
                return (
                    f"⚠️ **Model Not Found** — `{model_name}` is unavailable.\n\n"
                    "Try switching to a different model in the sidebar.\n\n"
                    f"```\n{err_str}\n```"
                )

            # ── Safety filter block ──
            if "safety" in err_lower or "block" in err_lower:
                return (
                    "⚠️ **Safety Filter** — Gemini blocked this response.\n\n"
                    "Please try rephrasing your message."
                )

            # ── Unknown error — don't retry ──
            return f"⚠️ **Error**: `{err_str}`"

    return "⚠️ Unexpected error after all retries."


# ═════════════════════════════════════════════
# SELF-DEBATE PIPELINE
# ═════════════════════════════════════════════
def run_self_debate(
    api_key: str,
    model_name: str,
    chat_messages: list[dict],
    status_container,
) -> tuple[str, dict]:
    """
    3-agent pipeline: Generator → Critic → Synthesizer.
    api_key and model_name are explicit args — never read from globals mid-run.
    """
    thinking = {}

    # ── Step 1: Generator ──
    status_container.update(label="🔵 Step 1/3 — Generating initial draft…", state="running")
    draft = call_gemini(
        api_key=api_key,
        model_name=model_name,
        system_prompt=(
            "คุณคือ Generator Agent คุณมีหน้าที่ร่างคำตอบเบื้องต้นจากข้อความของผู้ใช้ "
            "โดยอ้างอิงบริบทจากประวัติการสนทนาทั้งหมด ตอบเป็นภาษาไทย ให้ละเอียดและครอบคลุม "
            "ใช้ Markdown ได้เต็มรูปแบบ (ตาราง, โค้ด, หัวข้อ)"
        ),
        messages=chat_messages,
        temperature=0.7,
    )
    thinking["generator"] = draft

    # If Generator failed → abort early
    if draft.startswith("⚠️"):
        thinking["critic"] = "⏭️ Skipped — Generator did not produce a valid draft."
        thinking["synthesizer"] = draft
        status_container.update(label="⚠️ Completed with errors", state="error")
        return draft, thinking

    # ── Step 2: Critic ──
    status_container.update(label="🟠 Step 2/3 — Analyzing for improvements…", state="running")
    critic_messages = chat_messages + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": "วิเคราะห์คำตอบข้างต้น: จุดอ่อน ข้อผิดพลาด และข้อเสนอแนะ"},
    ]
    critique = call_gemini(
        api_key=api_key,
        model_name=model_name,
        system_prompt=(
            "คุณคือ Critic Agent คุณจะได้รับร่างคำตอบจาก Generator "
            "ให้วิเคราะห์จุดอ่อน ข้อผิดพลาด ข้อมูลที่ขาดหาย หรือส่วนที่อาจทำให้เข้าใจผิด "
            "และเสนอแนะการปรับปรุงอย่างชัดเจน ตอบเป็นภาษาไทย"
        ),
        messages=critic_messages,
        temperature=0.4,
    )
    thinking["critic"] = critique

    # If Critic failed → Synthesizer uses draft alone
    critique_for_synth = critique
    if critique.startswith("⚠️"):
        critique_for_synth = "ไม่มีข้อเสนอแนะเพิ่มเติม (Critic ไม่สามารถวิเคราะห์ได้)"

    # ── Step 3: Synthesizer ──
    status_container.update(label="🟢 Step 3/3 — Synthesizing final answer…", state="running")
    synth_messages = chat_messages + [
        {"role": "assistant", "content": f"**Draft:**\n{draft}\n\n**Critique:**\n{critique_for_synth}"},
        {"role": "user", "content": "สังเคราะห์คำตอบสุดท้ายที่ดีที่สุดจาก Draft และ Critique"},
    ]
    final = call_gemini(
        api_key=api_key,
        model_name=model_name,
        system_prompt=(
            "คุณคือ Synthesizer Agent คุณจะได้รับ Draft และ Critique "
            "ให้สังเคราะห์คำตอบสุดท้ายที่ดีที่สุด โดยรักษาจุดแข็งของ Draft "
            "และแก้ไขตามข้อเสนอแนะของ Critic ตอบเป็นภาษาไทย "
            "ใช้ Markdown เต็มรูปแบบ ห้ามกล่าวถึง Draft หรือ Critique ในคำตอบ "
            "ตอบเสมือนเป็นคำตอบสุดท้ายที่สมบูรณ์โดยตรง"
        ),
        messages=synth_messages,
        temperature=0.5,
    )
    thinking["synthesizer"] = final

    # If Synthesizer failed → fall back to Generator draft
    if final.startswith("⚠️"):
        final = f"{final}\n\n---\n\n**Fallback (Generator Draft):**\n{draft}"

    status_container.update(label="✅ Done thinking", state="complete")
    return final, thinking


# ═════════════════════════════════════════════
# RENDER: Thinking expander (reusable)
# ═════════════════════════════════════════════
def render_thinking(thinking: dict):
    """Render the 3-step thinking process inside the current container."""
    with st.expander("💭 How Gemini thought about this", expanded=False):
        st.markdown(
            '<span class="thinking-label label-generator">🔵 Generator — Initial Draft</span>',
            unsafe_allow_html=True,
        )
        st.markdown(thinking.get("generator", "—"))
        st.markdown("---")
        st.markdown(
            '<span class="thinking-label label-critic">🟠 Critic — Analysis</span>',
            unsafe_allow_html=True,
        )
        st.markdown(thinking.get("critic", "—"))
        st.markdown("---")
        st.markdown(
            '<span class="thinking-label label-synthesizer">🟢 Synthesizer — Final</span>',
            unsafe_allow_html=True,
        )
        st.markdown(thinking.get("synthesizer", "—"))


# ═════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 Self-Debate Chat")
    st.caption(f"Powered by `{st.session_state.model_name}`")
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Model selector ──
    MODEL_OPTIONS = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    _cur_idx = (
        MODEL_OPTIONS.index(st.session_state.model_name)
        if st.session_state.model_name in MODEL_OPTIONS
        else 0
    )
    selected_model = st.selectbox(
        "Model", MODEL_OPTIONS, index=_cur_idx, label_visibility="collapsed"
    )
    if selected_model != st.session_state.model_name:
        st.session_state.model_name = selected_model
        st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── New Chat ──
    if st.button("＋  New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── API Key ──
    current_key = get_api_key()

    if not current_key:
        st.markdown("##### 🔑 API Key")
        st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your Gemini API key",
            label_visibility="collapsed",
            key="api_key_input",  # writes directly to st.session_state.api_key_input
        )
        # Show validation hint if something was typed but invalid
        typed = st.session_state.get("api_key_input", "")
        if typed and not validate_api_key(typed):
            st.warning("Key looks too short. Gemini keys are typically 39 characters starting with `AIza…`")
    else:
        st.success("API Key active", icon=":material/check_circle:")

        # Debug toggle — show masked key
        if st.checkbox("Show masked key", value=False):
            masked = current_key[:8] + "•" * (len(current_key) - 12) + current_key[-4:]
            st.code(masked, language=None)

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Recent Chats ──
    st.markdown("##### Recent Chats")
    if st.session_state.all_chats:
        sorted_ids = sorted(
            st.session_state.all_chats.keys(),
            key=lambda k: st.session_state.all_chats[k]["created_at"],
            reverse=True,
        )
        for cid in sorted_ids:
            chat_data = st.session_state.all_chats[cid]
            is_active = cid == st.session_state.active_chat_id
            icon = "●" if is_active else "○"
            if st.button(
                f"{icon}  {chat_data['title']}", key=f"chat_{cid}", use_container_width=True
            ):
                st.session_state.active_chat_id = cid
                st.rerun()
    else:
        st.caption("No chats yet. Start one!")


# ═════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════

# Resolve key once for this render cycle
API_KEY = get_api_key()
HAS_KEY = API_KEY is not None and validate_api_key(API_KEY)

# Welcome screen
if (
    st.session_state.active_chat_id is None
    or st.session_state.active_chat_id not in st.session_state.all_chats
):
    st.markdown(
        """
    <div class="welcome-container">
        <div class="welcome-badge">🧠 Self-Debate System</div>
        <h1>What can I help you with?</h1>
        <p>
            Every answer goes through a 3-step thinking pipeline —<br>
            <strong>Generate → Critique → Synthesize</strong> — for higher quality responses.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if not HAS_KEY:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started.")

# Display existing messages
for msg in get_active_messages():
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "thinking" in msg:
            render_thinking(msg["thinking"])

# ═════════════════════════════════════════════
# CHAT INPUT
# ═════════════════════════════════════════════
if prompt := st.chat_input("Message Self-Debate Chat…", disabled=not HAS_KEY):

    # ── GATE: Re-resolve key at the exact moment of submission ──
    submit_key = get_api_key()
    if not submit_key or not validate_api_key(submit_key):
        st.error(
            "⚠️ **API Key is missing or invalid.** "
            "Please enter a valid Gemini API key in the sidebar before sending a message."
        )
        st.stop()

    # Auto-create chat if none active
    if (
        st.session_state.active_chat_id is None
        or st.session_state.active_chat_id not in st.session_state.all_chats
    ):
        create_new_chat()

    cid = st.session_state.active_chat_id
    chat_store = st.session_state.all_chats[cid]

    # Title from first message
    if not chat_store["messages"]:
        set_chat_title(cid, prompt)

    # Save user message
    chat_store["messages"].append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # ── Run pipeline ──
    # Snapshot key + model into local vars so they can't change mid-pipeline
    _key = submit_key
    _model = st.session_state.model_name

    with st.chat_message("assistant", avatar="🧠"):
        with st.status("🧠 Thinking…", expanded=True) as status:
            final_answer, thinking = run_self_debate(
                api_key=_key,
                model_name=_model,
                chat_messages=chat_store["messages"],
                status_container=status,
            )

        st.markdown(final_answer)
        render_thinking(thinking)

    # Save assistant message
    chat_store["messages"].append({
        "role": "assistant",
        "content": final_answer,
        "thinking": thinking,
    })
