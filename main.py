import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# Set your Gemini API key here
GEMINI_API_KEY = "AIzaSyCbiUUxGB-4cpjUZ-P7O6bJwijdftvYEec"

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# --- Helper function ---
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

# --- Text Analysis Tools ---
def sentiment_analysis():
    st.header("Sentiment Analysis")
    text = st.text_area("Paste text to analyze sentiment:")
    if st.button("Analyze Sentiment"):
        prompt = f"Analyze the sentiment of this text and respond with Positive, Neutral, or Negative:\n\n{text}"
        result = call_gemini_api(prompt)
        st.write("Sentiment:", result)

def spam_detector():
    st.header("Spam / Fake News Detector")
    text = st.text_area("Paste text to detect spam/fake news:")
    if st.button("Check Spam/Fake News"):
        prompt = f"Detect if this text is spam or fake news. Respond with Yes or No and explain briefly:\n\n{text}"
        result = call_gemini_api(prompt)
        st.write(result)

def plagiarism_checker():
    st.header("Plagiarism Checker / Text Summarizer")
    text = st.text_area("Paste text to summarize or check plagiarism:")
    if st.button("Summarize Text"):
        prompt = f"Summarize the following text briefly:\n\n{text}"
        summary = call_gemini_api(prompt)
        st.write("Summary:", summary)

# --- Voice & Speech AI (placeholders) ---
def speech_to_text():
    st.header("Speech-to-Text Transcription")
    st.info("Upload audio file feature coming soon!")

def text_to_speech():
    st.header("Text-to-Speech Reader")
    st.info("Upload Text-to-Speech Reader feature coming soon!")

# --- Recommendation Systems ---
def recommend_movies():
    st.header("Movie Recommender")
    genre = st.text_input("Enter preferred genre:")
    if st.button("Get Recommendations"):
        prompt = f"Recommend 5 {genre} movies with short descriptions."
        recs = call_gemini_api(prompt)
        st.write(recs)

# --- Language Learning Tools ---
def language_tutor():
    st.header("AI Language Tutor with Grammar Correction")
    text = st.text_area("Write a sentence or paragraph to correct:")
    if st.button("Correct Grammar"):
        prompt = f"Correct the grammar and improve this text:\n\n{text}"
        corrected = call_gemini_api(prompt)
        st.write("Corrected Text:", corrected)

def translator():
    st.header("Real-time Translator")
    text = st.text_area("Enter text to translate to French:")
    if st.button("Translate"):
        prompt = f"Translate this text to French:\n\n{text}"
        translation = call_gemini_api(prompt)
        st.write("Translation:", translation)

# --- Automation Tools ---
def email_generator():
    st.header("Email / Social Media Content Generator")
    topic = st.text_input("Enter topic or subject:")
    if st.button("Generate Content"):
        prompt = f"Write an engaging email about {topic}."
        email_text = call_gemini_api(prompt)
        st.write(email_text)

# --- Healthcare AI ---
def symptom_checker():
    st.header("Symptom Checker Chatbot")
    symptoms = st.text_area("Enter your symptoms:")
    if st.button("Check Symptoms"):
        prompt = f"Based on these symptoms, what could be the possible causes? {symptoms}. Note: This is for informational purposes only and not medical advice."
        advice = call_gemini_api(prompt)
        st.write(advice)
        st.warning("⚠️ This is for informational purposes only. Please consult a healthcare professional for medical advice.")

# --- Educational Tools ---
def quiz_generator():
    st.header("Quiz Generator")
    uploaded_file = st.file_uploader("Upload your study material (TXT or PDF)", type=["txt", "pdf"])
    num_questions = st.slider("Number of questions", 1, 10, 5)
    
    def extract_text_from_pdf(file):
        try:
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
        
        with st.expander("Extracted Text Preview"):
            st.write(text[:1000] + "..." if len(text) > 1000 else text)
        
        if st.button("Generate Quiz"):
            prompt = f"Generate {num_questions} multiple choice questions with 4 options each based on this text:\n\n{text}"
            quiz = call_gemini_api(prompt)
            st.text_area("Generated Quiz Questions:", quiz, height=300)

# --- Creative Tools ---
def story_generator():
    st.header("AI Story / Poetry Generator")
    topic = st.text_input("Enter story or poem topic:")
    if st.button("Generate Story/Poem"):
        prompt = f"Write a creative story or poem about {topic}."
        story = call_gemini_api(prompt)
        st.write(story)

# --- Main app layout ---
st.title("✨ Spark 5.0 Ultra MAX Mode AI By Aaradhya Pratish Vanakhade")

st.sidebar.title("Select AI Tool Category")

option = st.sidebar.selectbox("Choose category", [
    "Text Analysis",
    "Voice & Speech AI",
    "Recommendation Systems",
    "Language Learning Tools",
    "Automation Tools",
    "Healthcare AI",
    "Educational Tools",
    "Creative Tools"
])

if option == "Text Analysis":
    tabs = st.tabs(["Sentiment Analysis", "Spam Detector", "Plagiarism Checker"])
    with tabs[0]:
        sentiment_analysis()
    with tabs[1]:
        spam_detector()
    with tabs[2]:
        plagiarism_checker()

elif option == "Voice & Speech AI":
    tabs = st.tabs(["Speech-to-Text", "Text-to-Speech"])
    with tabs[0]:
        speech_to_text()
    with tabs[1]:
        text_to_speech()

elif option == "Recommendation Systems":
    recommend_movies()

elif option == "Language Learning Tools":
    tabs = st.tabs(["Language Tutor", "Translator"])
    with tabs[0]:
        language_tutor()
    with tabs[1]:
        translator()

elif option == "Automation Tools":
    email_generator()

elif option == "Healthcare AI":
    symptom_checker()

elif option == "Educational Tools":
    quiz_generator()

elif option == "Creative Tools":
    story_generator()
