import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import speech_recognition as sr
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
    text = st.text_area("Paste text to analyze sentiment:", key="sentiment_text")
    if st.button("Analyze Sentiment", key="analyze_sentiment"):
        prompt = f"Analyze the sentiment of this text and respond with Positive, Neutral, or Negative:\n\n{text}"
        result = call_gemini_api(prompt)
        st.write("Sentiment:", result)

def spam_detector():
    st.header("Spam / Fake News Detector")
    text = st.text_area("Paste text to detect spam/fake news:", key="spam_text")
    if st.button("Check Spam/Fake News", key="check_spam"):
        prompt = f"Detect if this text is spam or fake news. Respond with Yes or No and explain briefly:\n\n{text}"
        result = call_gemini_api(prompt)
        st.write(result)

def plagiarism_checker():
    st.header("Plagiarism Checker / Text Summarizer")
    text = st.text_area("Paste text to summarize or check plagiarism:", key="plagiarism_text")
    if st.button("Summarize Text", key="summarize_text"):
        prompt = f"Summarize the following text briefly:\n\n{text}"
        summary = call_gemini_api(prompt)
        st.write("Summary:", summary)

# --- Voice & Speech AI ---
def speech_to_text():
    st.header("🎤 Real Speech-to-Text Transcription")
    st.write("Click the button and speak. Your speech will be converted to text.")

    if st.button("Start Recording", key="start_recording"):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            st.info("Listening... Please speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, phrase_time_limit=10)

        try:
            text = recognizer.recognize_google(audio)
            st.success("Transcription:")
            st.write(text)
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")

# --- Recommendation Systems ---
def recommend_movies():
    st.header("Movie Recommender")
    genre = st.text_input("Enter preferred genre:", key="movie_genre")
    if st.button("Get Recommendations", key="get_recommendations"):
        prompt = f"Recommend 5 {genre} movies with short descriptions."
        recs = call_gemini_api(prompt)
        st.write(recs)

# --- Language Learning Tools ---
def language_tutor():
    st.header("AI Language Tutor with Grammar Correction")
    text = st.text_area("Write a sentence or paragraph to correct:", key="language_text")
    if st.button("Correct Grammar", key="correct_grammar"):
        prompt = f"Correct the grammar and improve this text:\n\n{text}"
        corrected = call_gemini_api(prompt)
        st.write("Corrected Text:", corrected)

def translator():
    st.header("Real-time Translator")
    text = st.text_area("Enter text to translate to French:", key="translate_text")
    if st.button("Translate", key="translate_button"):
        prompt = f"Translate this text to French:\n\n{text}"
        translation = call_gemini_api(prompt)
        st.write("Translation:", translation)

# --- Automation Tools ---
def email_generator():
    st.header("Email / Social Media Content Generator")
    topic = st.text_input("Enter topic or subject:", key="email_topic")
    if st.button("Generate Content", key="generate_email"):
        prompt = f"Write an engaging email about {topic}."
        email_text = call_gemini_api(prompt)
        st.write(email_text)

# --- Healthcare AI ---
def symptom_checker():
    st.header("Symptom Checker Chatbot")
    symptoms = st.text_area("Enter your symptoms:", key="symptom_text")
    if st.button("Check Symptoms", key="check_symptoms"):
        prompt = f"Based on these symptoms, what could be the possible causes? {symptoms}. Note: This is for informational purposes only and not medical advice."
        advice = call_gemini_api(prompt)
        st.write(advice)
        st.warning("⚠️ This is for informational purposes only. Please consult a healthcare professional for medical advice.")

# --- Educational Tools ---
def quiz_generator():
    st.header("Quiz Generator")
    uploaded_file = st.file_uploader("Upload your study material (TXT or PDF)", type=["txt", "pdf"], key="quiz_uploader")
    num_questions = st.slider("Number of questions", 1, 10, 5, key="num_questions")
    
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
        
        if st.button("Generate Quiz", key="generate_quiz"):
            prompt = f"Generate {num_questions} multiple choice questions with 4 options each based on this text:\n\n{text}"
            quiz = call_gemini_api(prompt)
            st.text_area("Generated Quiz Questions:", quiz, height=300, key="quiz_output")

# --- Creative Tools ---
def story_generator():
    st.header("AI Story / Poetry Generator")
    topic = st.text_input("Enter story or poem topic:", key="story_topic")
    if st.button("Generate Story/Poem", key="generate_story"):
        prompt = f"Write a creative story or poem about {topic}."
        story = call_gemini_api(prompt)
        st.write(story)

# --- Main app layout ---
col1, col2 = st.columns([1, 4])

with col1:
    st.image("ChatGPT Image Jun 6, 2025, 10_32_22 AM.png", width=125)

with col2:
    st.markdown("## Spark 5.0 Ultra MAX Mode AI")
    st.caption("By Aaradhya Pratish Vanakhade")

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
    tabs = st.tabs(["Speech-to-Text"])
    with tabs[0]:
        speech_to_text()

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
