# 🧠 Self-Debate Chat

A Claude-inspired Streamlit chat app powered by **Gemini 1.5 Pro** with a 3-agent "Self-Debate" thinking pipeline.

Every response goes through: **Generator → Critic → Synthesizer** for higher quality answers.

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create secrets file
mkdir -p .streamlit
echo '[secrets]' > .streamlit/secrets.toml
echo 'GEMINI_API_KEY = "your-api-key-here"' >> .streamlit/secrets.toml

# 3. Run the app
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

Push `app.py` and `requirements.txt` to a GitHub repository.

### Step 2 — Create the app on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your repo, branch, and set the main file to `app.py`

### Step 3 — Add Secrets

1. In the Streamlit Cloud dashboard, open your app's **Settings → Secrets**
2. Add:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

3. Save. The app will reload automatically.

> **Get a Gemini API key** at [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)

---

## Features

| Feature | Description |
|---|---|
| **Multi-turn chat** | Full conversation history with context |
| **Chat sessions** | Create and switch between multiple chats |
| **3-agent pipeline** | Generator → Critic → Synthesizer |
| **Thinking process** | Expandable view of each agent's output |
| **Claude-like UI** | Clean, minimal, centered layout |
| **Error handling** | Retries on rate limits, safety filter fallback |
| **Thai language** | All agent prompts output in Thai |
| **Markdown support** | Tables, code blocks, headers, etc. |

---

## Architecture

```
User Message
    │
    ▼
┌─────────────┐
│  Generator   │  → Draft response (Thai, creative, temp=0.7)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Critic     │  → Analyze weaknesses & suggest fixes (temp=0.4)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Synthesizer  │  → Final polished answer (temp=0.5)
└─────────────┘
       │
       ▼
  Final Response
```
