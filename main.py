import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# Try importing audio modules
try:
    import speech_recognition as sr
    import pyttsx3
    AUDIO_ENABLED = True
except:
    AUDIO_ENABLED = False

# Set your Gemini API key here
GEMINI_API_KEY = "AIzaSyCbiUUxGB-4cpjUZ-P7O6bJwijdftvYEec"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Helper
def call_gemini_api(prompt, temperature=0.7):
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=700,
            )
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Theme toggle
theme = st.sidebar.radio("Choose Theme", ["🌞 Light", "🌚 Dark"])
if theme == "🌚 Dark":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Header: logo + title
col1, col2 = st.columns([1, 4])
with col1:
    st.image("ChatGPT Image Jun 6, 2025, 10_32_22 AM.png", width=125)
with col2:
    st.markdown("## Spark 5.0 Ultra MAX Mode AI")
    st.caption("By Aaradhya Pratish Vanakhade")

# Sidebar
st.sidebar.title("Select AI Tool Category")
option = st.sidebar.selectbox("Choose category", [
    "Text Analysis", "Voice & Speech AI", "Recommendation Systems",
    "Language Learning Tools", "Automation Tools",
    "Healthcare AI", "Educational Tools", "Creative Tools"
])

# --- Tools ---
def sentiment_analysis():
    st.header("Sentiment Analysis")
    text = st.text_area("Paste text to analyze sentiment:")
    if st.button("Analyze Sentiment"):
        prompt = f"Analyze the sentiment (Positive/Neutral/Negative):\n\n{text}"
        st.write("Sentiment:", call_gemini_api(prompt))

def spam_detector():
    st.header("Spam / Fake News Detector")
    text = st.text_area("Paste text:")
    if st.button("Check"):
        prompt = f"Is this spam or fake news? Explain:\n\n{text}"
        st.write(call_gemini_api(prompt))

def plagiarism_checker():
    st.header("Plagiarism Checker / Summarizer")
    text = st.text_area("Paste text:")
    if st.button("Summarize"):
        prompt = f"Summarize this:\n\n{text}"
        st.write("Summary:", call_gemini_api(prompt))

def speech_to_text():
    st.header("🎤 Speech-to-Text")
    if not AUDIO_ENABLED:
        st.error("Audio features are not supported on this deployment.")
        return
    if st.button("Start Recording"):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            st.info("Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, phrase_time_limit=10)
        try:
            st.success("You said:")
            st.write(recognizer.recognize_google(audio))
        except Exception as e:
            st.error(str(e))

def text_to_speech():
    st.header("🔊 Text-to-Speech")
    if not AUDIO_ENABLED:
        st.error("Audio features are not supported on this deployment.")
        return
    text = st.text_area("Enter text:")
    if st.button("Speak"):
        if text.strip():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            st.success("Spoken!")
        else:
            st.warning("Please enter text.")

def recommend_movies():
    st.header("Movie Recommender")
    genre = st.text_input("Preferred genre:")
    if st.button("Recommend"):
        prompt = f"Recommend 5 {genre} movies with descriptions."
        st.write(call_gemini_api(prompt))

def language_tutor():
    st.header("Language Tutor")
    text = st.text_area("Write a sentence:")
    if st.button("Correct"):
        prompt = f"Correct the grammar:\n\n{text}"
        st.write("Corrected:", call_gemini_api(prompt))

def translator():
    st.header("Translator")
    text = st.text_area("Text to translate to French:")
    if st.button("Translate"):
        prompt = f"Translate to French:\n\n{text}"
        st.write("Translation:", call_gemini_api(prompt))

def email_generator():
    st.header("Email / Content Generator")
    topic = st.text_input("Enter topic:")
    if st.button("Generate"):
        prompt = f"Write an engaging email about {topic}."
        st.write(call_gemini_api(prompt))

def symptom_checker():
    st.header("Symptom Checker")
    symptoms = st.text_area("Enter symptoms:")
    if st.button("Check"):
        prompt = f"What are the possible causes of: {symptoms}. Disclaimer: not medical advice."
        st.write(call_gemini_api(prompt))
        st.warning("⚠️ This is not a medical diagnosis.")

def quiz_generator():
    st.header("Quiz Generator")
    file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    num = st.slider("No. of questions", 1, 10, 5)

    def extract_text(file):
        if file.type == "application/pdf":
            reader = PdfReader(file)
            return "\n".join([p.extract_text() for p in reader.pages])
        return file.read().decode("utf-8")

    if file:
        text = extract_text(file)
        st.text_area("Preview:", text[:1000] + "..." if len(text) > 1000 else text)
        if st.button("Generate Quiz"):
            prompt = f"Create {num} multiple-choice questions from:\n\n{text}"
            st.text_area("Quiz:", call_gemini_api(prompt), height=300)

def story_generator():
    st.header("Story / Poem Generator")
    topic = st.text_input("Topic:")
    if st.button("Generate"):
        prompt = f"Write a creative story or poem about: {topic}"
        st.write(call_gemini_api(prompt))

# --- Router ---
if option == "Text Analysis":
    tab1, tab2, tab3 = st.tabs(["Sentiment", "Spam", "Plagiarism"])
    with tab1: sentiment_analysis()
    with tab2: spam_detector()
    with tab3: plagiarism_checker()

elif option == "Voice & Speech AI":
    tab1, tab2 = st.tabs(["Speech-to-Text", "Text-to-Speech"])
    with tab1: speech_to_text()
    with tab2: text_to_speech()

elif option == "Recommendation Systems":
    recommend_movies()

elif option == "Language Learning Tools":
    tab1, tab2 = st.tabs(["Grammar", "Translate"])
    with tab1: language_tutor()
    with tab2: translator()

elif option == "Automation Tools":
    email_generator()

elif option == "Healthcare AI":
    symptom_checker()

elif option == "Educational Tools":
    quiz_generator()

elif option == "Creative Tools":
    story_generator()
