# app.py — TruthLens
import streamlit as st
import joblib
import numpy as np
import sqlite3
import pandas as pd
from collections import Counter
import altair as alt
from datetime import datetime
import math  # added for NaN/invalid label checks

# -------------------------
# Page config & branding
# -------------------------
st.set_page_config(page_title="TruthLens", page_icon="🕵️", layout="wide")
st.title("🕵️ TruthLens — AI Misinformation Radar")
st.caption("Detect • Explain • Educate • Flag")

# -------------------------
# Load model (joblib) safely
# -------------------------
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}: {e}")
    st.stop()

# -------------------------
# Database (flagged storage)
# -------------------------
DB_PATH = "misinfo.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS flagged")
c.execute('''CREATE TABLE IF NOT EXISTS flagged
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              text TEXT,
              prediction TEXT,
              confidence REAL,
              ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()
rows = c.execute(
    "SELECT id, text, prediction, confidence, ts FROM flagged ORDER BY id DESC"
).fetchall()

# -------------------------
# Session state initialization
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {text, prediction, confidence, explanation, ts}
if "counts" not in st.session_state:
    st.session_state.counts = Counter()

# -------------------------
# Knowledge & keyword lists
# -------------------------
geo_facts = {
    "telangana": "india", "kerala": "india", "delhi": "india", "hyderabad": "india",
    "london": "uk", "paris": "france", "tokyo": "japan",
    "sydney": "australia", "toronto": "canada", "berlin": "germany",
    "moscow": "russia", "beijing": "china", "dubai": "uae", "rome": "italy"
}

# expanded scam keywords
fraud_keywords = [
    "otp", "one time password", "verification code", "cvv", "cvv2", "pin",
    "bank details", "account number", "account locked", "click this link", "scammer", 
    "provide your", "send your otp", "share your otp", "aadhar details", "aadhar", "pan card"
]
health_misinfo_keywords = ["bleach", "cancer cure", "5g", "miracle drug", "no vaccine"]
casual_keywords = ["hey", "hi", "hello", "thanks", "birthday", "lunch", "meeting", "plans", "share notes", "free tonight", "party"]

# -------------------------
# Generative explanation placeholder
# -------------------------
def generate_explanation(text: str, prediction: str) -> str:
    """
    Placeholder for generative explanations.
    Replace the body with a call to an LLM (OpenAI / Google Vertex / other).
    Example: Use OpenAI's chat/completions or Google Vertex to create a short explanation.

    Example (OpenAI pseudo-code):
      import openai
      openai.api_key = OPENAI_KEY
      resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[...])
      return resp['choices'][0]['message']['content']

    For now this returns a safe canned explanation based on prediction.
    """
    if prediction == "fraudulent":
        return "This message contains scam-like language (requests for OTP/CVV/bank details). Do not share personal credentials."
    if prediction == "misinformation":
        return "This claim may be misleading or false. Consider verifying with trusted sources (official sites, reputable news, or fact-checkers)."
    if prediction == "normal_chat":
        return "This appears to be casual conversation (low risk)."
    if prediction == "genuine":
        return "This claim appears consistent with accepted facts or common knowledge."
    return "Model-based classification. Consider manual review."

# -------------------------
# Helper: analyze single text
# -------------------------
def analyze_text(text: str):
    text_lower = text.lower()
    prediction = None
    confidence = 0.0
    reason = ""

    # 1) High-priority scam keyword detection
    for kw in fraud_keywords:
        if kw in text_lower:
            prediction = "fraudulent"
            confidence = 99.0
            reason = f"Detected fraud keyword: '{kw}'"
            break

    # 2) Geography check (only if not already fraud)
    if prediction is None:
        for place, country in geo_facts.items():
            if place in text_lower and country not in text_lower:
                prediction = "misinformation"
                confidence = 99.0
                reason = f"Geography check: {place.title()} belongs to {country.title()}, mismatch detected."
                break

    # 3) Health misinformation keywords (if still none)
    if prediction is None:
        for kw in health_misinfo_keywords:
            if kw in text_lower:
                prediction = "misinformation"
                confidence = 99.0
                reason = f"Detected health-misinformation keyword: '{kw}'"
                break

    # 4) ML fallback (if no rules triggered)
    if prediction is None:
        proba = model.predict_proba([text])[0]
        labels = model.classes_
        idx = int(np.argmax(proba))

        # attempt to assign label safely
        try:
            raw_label = labels[idx]
        except Exception:
            raw_label = None

        confidence = float(proba[idx] * 100) if len(proba) > idx else 0.0
        reason = "Classified by the AI model."

        # --- Safeguard: handle NaN / invalid label by replacing with 'genuine' ---
        # This prevents showing 'nan' as a prediction in the UI.
        if raw_label is None or (isinstance(raw_label, float) and math.isnan(raw_label)) or str(raw_label).strip() == "" or str(raw_label).lower() == "nan":
            prediction = "genuine"
            reason = "Model returned invalid label — replaced with 'genuine'."
            confidence = 80.0
        else:
            prediction = str(raw_label)

    # 5) Casual chat heuristic (only if not already fraud/misinformation)
    if prediction not in ("fraudulent", "misinformation"):
        if confidence < 65 or any(word in text_lower for word in casual_keywords):
            prediction = "normal_chat"
            reason = "Low confidence or casual keywords detected: flagged as normal chat."

    # 6) Generative explanation (placeholder)
    explanation = generate_explanation(text, prediction)

    # 7) Return structured result
    return {
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "reason": reason,
        "explanation": explanation,
        "ts": datetime.utcnow().isoformat()
    }

# -------------------------
# Tabs: Analyze | History | Learn
# -------------------------
tabs = st.tabs(["🔎 Analyze", "📋 Flagged Records", "📊 Analytics", "🕒 History", "📚 Learn"])

# ----- TAB: Analyze -----
with tabs[0]:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Analyze a statement")
        example_list = [
            "Share your OTP now",
            "Telangana is in America",
            "Vaccines are safe and effective",
            "Hey are you free tonight?"
        ]
        example_choice = st.selectbox("Try an example:", [""] + example_list)
        user_input = example_choice if example_choice else st.text_area("Enter a statement to analyze:", height=120)

        # Batch upload
        uploaded = st.file_uploader("Or upload a CSV (one column 'Text') for batch analysis", type=["csv"])
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                if "Text" not in df_upload.columns:
                    st.warning("CSV must contain a column named 'Text'.")
                else:
                    if st.button("Run batch analysis"):
                        results = []
                        for t in df_upload["Text"].astype(str).tolist():
                            res = analyze_text(t)
                            results.append(res)
                            # add to session history
                            st.session_state.history.append((res["text"], res["prediction"], res["confidence"], res["ts"]))
                        st.success("Batch analysis complete — results shown below.")
                        st.dataframe(pd.DataFrame(results)[["text", "prediction", "confidence", "reason"]])
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    with right:
        st.subheader("Quick controls & metrics")
        st.write("Model: TF-IDF + classifier")
        st.metric("Session analyses", len(st.session_state.history))
        # small legend
        st.markdown("**Legend**: fraud = scam, misinformation = likely false, normal_chat = casual")

    # single analyze action (below columns so button spans)
    if st.button("Analyze statement"):
        if not user_input or str(user_input).strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            result = analyze_text(user_input)
            # display nicely
            st.markdown(f"### Prediction: **{result['prediction']}**")
            st.progress(min(int(result["confidence"]), 100))
            st.caption(f"Confidence: {result['confidence']:.2f}%")
            st.info(result["reason"])
            st.success(result["explanation"])

            # save to session history
            st.session_state.history.append((result["text"], result["prediction"], result["confidence"], result["ts"]))
            st.session_state.counts.update([result["prediction"]])

            # log flagged misinformation to DB (only high confidence)
            if result["prediction"] == "misinformation" and result["confidence"] >= 70:
                c.execute("INSERT INTO flagged (text,prediction,confidence,ts) VALUES (?,?,?,?)",
                          (result["text"], result["prediction"], result["confidence"], result["ts"]))
                conn.commit()
                st.warning("⚠️ Misinformation flagged and stored for review.")
            # --- Fraudulent Handling Section (Replace your current block with this) ---
            if result["prediction"] == "fraudulent":
               # Store frauds in DB
               c.execute("INSERT INTO flagged (text,prediction,confidence,ts) VALUES (?,?,?,?)",
                        (result["text"], result["prediction"], result["confidence"], result["ts"]))
               conn.commit()
               st.error("🚩 Fraudulent message logged.")

               # --- Persistent Report Button ---
               # Initialize state flag if not exists
               if "show_report_contacts" not in st.session_state:
                   st.session_state.show_report_contacts = True

               # Show the button (unique key to prevent conflicts)
               if st.button("📢 Report to Cyber Crime", key=f"report_{result['ts']}"):
                   st.session_state.show_report_contacts = True

               # Display contacts persistently once pressed
               if st.session_state.show_report_contacts:
                   st.info("☎ **Toll-free Number To Report Hyderabad Cyber Crime Helpline:** 1930")
                   st.info("📧 **Hyderabad Cyber Crime Email:** dcpcybercrimeshyd[at]@gmail.com")

            # ----- TAB: Flagged Records -----
with tabs[1]:
    st.subheader("Flagged records (DB)")
    rows = c.execute("SELECT id, text, prediction, confidence, ts FROM flagged ORDER BY id DESC").fetchall()
    if rows:
        df_flagged = pd.DataFrame(rows, columns=["id", "text", "prediction", "confidence", "ts"])
        st.dataframe(df_flagged)
        csv_bytes = df_flagged.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download flagged records as CSV", csv_bytes, file_name="flagged_records.csv")
    else:
        st.info("No flagged records yet.")

# ----- TAB: Analytics -----
with tabs[2]:
    st.subheader("Session analytics")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history, columns=["text", "prediction", "confidence", "ts"])
        counts = df_hist["prediction"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X("Category", sort="-y"),
            y="Count",
            color="Category"
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("#### Recent entries")
        st.dataframe(df_hist.tail(25))
    else:
        st.info("No session history yet — analyze some statements first.")

# ----- TAB: History -----
with tabs[3]:
    st.subheader("Session history")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history, columns=["text", "prediction", "confidence", "ts"])
        st.dataframe(df_history)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.counts = Counter()
            st.success("Session history cleared.")
    else:
        st.info("No history yet. Use Analyze tab to run checks.")

# ----- TAB: Learn -----
with tabs[4]:
    st.subheader("Learn: Tips & Resources")
    st.markdown("""
    **Why something might be flagged**
    - **Fraud/Scams**: Requests for OTP, CVV, account info, or links to unknown sites.
    - **Misinformation**: Unsupported factual claims, geography errors, health claims without evidence, Fake news.
    - **Low confidence**: Short or ambiguous messages often need manual review.

    **Quick Safety Tips**
    - Never share OTP, CVV, bank details, or passwords.
    - Verify suspicious claims with official sources (government sites, WHO, reputable news).
    - When in doubt, don't forward — check it or else Report to the Cyber Crime toll-free number:1930.

    **Resources**
    - [WHO mythbusters](https://www.who.int/)
    - [Fact-checking organizations (example)](https://www.poynter.org/)
    """)

# -------------------------
# Footer / small instructions
# -------------------------
st.markdown("---")
st.caption("Note: Never Share Your Personal Details to Anyone Report to Cyber Crime")
