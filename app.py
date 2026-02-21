import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import pandas as pd
from datetime import datetime
from io import BytesIO

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Mental Health AI Companion",
    page_icon="💙",
    layout="wide"
)

MODEL_PATH = "saved_models/emotion_model"

emotion_labels = ["sadness", "fear", "anger", "joy", "neutral"]

emotion_emojis = {
    "sadness": "😔",
    "fear": "😨",
    "anger": "😠",
    "joy": "😊",
    "neutral": "😐"
}

crisis_keywords = [
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "no reason to live"
]

# ===============================
# LOAD MODEL
# ===============================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

# ===============================
# SESSION STATE
# ===============================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

if "detailed_log" not in st.session_state:
    st.session_state.detailed_log = []

# ===============================
# FUNCTIONS
# ===============================

def detect_crisis(text):
    text = text.lower()
    for word in crisis_keywords:
        if word in text:
            return True
    return False


def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()

    return emotion_labels[prediction], confidence


def generate_response(emotion):
    responses = {
        "sadness": "I'm really sorry you're feeling this way. I'm here to listen 💙",
        "fear": "It sounds like you're feeling anxious. Try taking a slow deep breath.",
        "anger": "I understand anger can feel intense. Let's pause and reflect calmly.",
        "joy": "That's wonderful to hear! I'm happy for you 😊",
        "neutral": "Thank you for sharing. Tell me more about how you're feeling."
    }
    return responses.get(emotion, "I'm here for you.")


def generate_pdf_report(dataframe, mood_counts):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("<b>Mental Health Mood Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    # Summary Section
    elements.append(Paragraph("<b>Mood Summary:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for mood, count in mood_counts.items():
        elements.append(Paragraph(f"{mood}: {count}", styles["Normal"]))

    elements.append(Spacer(1, 0.5 * inch))

    # Detailed Table
    elements.append(Paragraph("<b>Detailed Session Log:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [list(dataframe.columns)] + dataframe.values.tolist()

    table = Table(table_data)
    table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])

    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("💙 Mental Health AI")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.mood_log = []
    st.session_state.detailed_log = []

st.sidebar.markdown("---")
st.sidebar.write("### 🔎 Emotion Legend")
for emotion in emotion_labels:
    st.sidebar.write(f"{emotion_emojis[emotion]} {emotion.capitalize()}")

# ===============================
# MAIN TITLE
# ===============================

st.title("💙 Mental Health AI Companion")
st.write("Talk to your AI emotional support assistant.")

# ===============================
# CHAT DISPLAY
# ===============================

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# ===============================
# USER INPUT
# ===============================

user_input = st.chat_input("Type how you're feeling...")

if user_input:

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if detect_crisis(user_input):
        crisis_message = (
            "⚠️ I'm really concerned about your safety.\n\n"
            "Please contact a trusted person or local emergency support immediately."
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": crisis_message}
        )
        st.chat_message("assistant").error(crisis_message)

    else:
        emotion, confidence = predict_emotion(user_input)
        emoji = emotion_emojis.get(emotion, "")

        st.session_state.mood_log.append(emotion)

        st.session_state.detailed_log.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "User Input": user_input,
            "Emotion": emotion,
            "Confidence": round(confidence, 2)
        })

        ai_response = generate_response(emotion)

        full_response = (
            f"{emoji} **Detected Emotion:** {emotion.capitalize()}\n\n"
            f"**Confidence:** {confidence:.2f}\n\n"
            f"{ai_response}"
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response}
        )

        st.chat_message("assistant").write(full_response)

# ===============================
# MOOD ANALYTICS
# ===============================

if st.session_state.mood_log:

    st.markdown("---")
    st.subheader("📊 Mood Analytics")

    mood_counts = Counter(st.session_state.mood_log)
    st.bar_chart(mood_counts)

    df = pd.DataFrame(st.session_state.detailed_log)

    # CSV Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Mood Report (CSV)",
        csv,
        "mental_health_report.csv",
        "text/csv"
    )

    # PDF Download
    pdf_buffer = generate_pdf_report(df, mood_counts)
    st.download_button(
        "Download Mood Report (PDF)",
        pdf_buffer,
        "mental_health_report.pdf",
        "application/pdf"
    )