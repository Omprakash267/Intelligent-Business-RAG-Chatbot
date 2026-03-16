import streamlit as st
import chatbot

st.set_page_config(page_title="Business Chatbot", layout="centered")
st.title("Business Chatbot")

# ---- Sidebar controls ----
k = st.sidebar.slider(
    "Top-K docs",
    min_value=1,
    max_value=10,
    value=4,
    key="topk_slider",
)

model = st.sidebar.text_input(
    "Ollama model",
    value=getattr(chatbot, "OLLAMA_MODEL", "llama2:latest"),
    key="model_input",
)

if st.sidebar.button("Clear chat", key="clear_chat_btn"):
    st.session_state.messages = []
    st.rerun()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Render history ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant" and m.get("sentiment"):
            s = m["sentiment"]
            st.caption(f"Sentiment: {s['label']} (confidence: {s['score']:.2f})")
        if m["role"] == "assistant" and m.get("confidence"):
            c = m["confidence"]
            st.caption(f"Grounding confidence: {c['label']} ({c['score']:.2f})")
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander("Sources used"):
                for src in m["sources"]:
                    st.markdown(
                        f"- **#{src['rank']} {src['source']}**  \n"
                        f"Distance: `{src['distance']}`  \n"
                        f"Snippet: {src['snippet']}"
                    )

# ---- Single chat input ----
user_query = st.chat_input(
    "Ask a question...",
    key="main_chat_input",
)

# ---- Handle new message ----
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    result = chatbot.answer_with_sentiment(user_query, k=k, model=model)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sentiment": result["sentiment"],
            "confidence": result.get("confidence"),
            "sources": result.get("sources", []),
        }
    )

    with st.chat_message("assistant"):
        st.write(result["answer"])
        s = result["sentiment"]
        st.caption(f"Sentiment: {s['label']} (confidence: {s['score']:.2f})")
        c = result.get("confidence")
        if c:
            st.caption(f"Grounding confidence: {c['label']} ({c['score']:.2f})")
        sources = result.get("sources", [])
        if sources:
            with st.expander("Sources used"):
                for src in sources:
                    st.markdown(
                        f"- **#{src['rank']} {src['source']}**  \n"
                        f"Distance: `{src['distance']}`  \n"
                        f"Snippet: {src['snippet']}"
                    )
