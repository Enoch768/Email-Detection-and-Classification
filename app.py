import pickle
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup
import re
import string
# Title and description

st.markdown(
    """
    <div style="text-align:center">
        <h1>Email Classification App</h1>
        <p>This app classifies emails into one of four categories: Fraudulent, Harassment, Normal, or Suspicious. 
        Using pre-trained machine learning models such as Logistic Regression, Naive Bayes, Random Forest, Decision Tree, and Gradient Boosting, 
        the app predicts the nature of the email based on its content.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Choose Model for Predictions")
st.sidebar.markdown(
    """
    This sidebar allows you to choose the model for making predictions.
    Different models may provide different results based on the same input data.
    """
)

# Use selectbox instead of radio for model selection
selected_model = st.sidebar.selectbox("Select Model", ('Logistic Regression', 'Naive Bayes', 'Random Forest', 'Decision Tree', 'GBC Model'))
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Preprocess email text
def preprocess_email_text(email_text):
    email_text = str(email_text)
    email_text = re.sub(r'^\s*([\w-]+:).*$', '', email_text, flags=re.MULTILINE)
    email_text = re.sub(r'----.*forwarded.*by.*----.*$', '', email_text, flags=re.IGNORECASE | re.DOTALL)
    email_text = re.sub(r'----.*original.*message.*----.*$', '', email_text, flags=re.IGNORECASE | re.DOTALL)
    email_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', email_text)
    email_text = email_text.lower()
    soup = BeautifulSoup(email_text, "html.parser")
    email_text = soup.get_text()
    email_text = re.sub(r'http\S+|www\S+|https\S+', '', email_text, flags=re.MULTILINE)
    email_text = re.sub(r'\d+', '', email_text)
    email_text = email_text.translate(str.maketrans('', '', string.punctuation))
    email_text = ' '.join(email_text.split())
    tokens = word_tokenize(email_text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
option = st.selectbox("Choose input method", ("Upload email file", "Copy and paste email text"))

email_text = ""

if option == "Upload email file":
    uploaded_file = st.file_uploader("Choose an email file", type=["eml", "txt"])
    if uploaded_file is not None:
        email_text = uploaded_file.read().decode("utf-8")

elif option == "Copy and paste email text":
    email_text = st.text_area("Paste your email text here")
# Label encoder classes
label_encoder_classes = {0: 'Fraudulent', 1: 'Harassment', 2: 'Normal', 3: 'Suspicious'}


if st.button('Classify'):
    if email_text:
        preprocessed_text = preprocess_email_text(email_text)
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        vectorized_text = vectorizer.transform([preprocessed_text])
        if selected_model == "Logistic Regression":
            model = pickle.load(open("logistic_regression_model.pkl", 'rb'))
        elif selected_model == "Naive Bayes":
            model = pickle.load(open("naive_bayes_model.pkl", 'rb'))
        elif selected_model == "Decision Tree":
            model = pickle.load(open("dt_model.pkl", 'rb'))
        elif selected_model == "GBC Model":
            model = pickle.load(open("gb_model.pkl", 'rb'))
        elif selected_model == "Random Forest":
            model = pickle.load(open("rf_model.pkl", 'rb'))
        
        predicted_class = model.predict(vectorized_text)[0]
        class_name = label_encoder_classes[predicted_class]
        
        # Enhanced output
        if class_name == 'Normal':
            st.success(f"The email is classified as: **{class_name}**")
            st.balloons()
        elif class_name == 'Fraudulent':
            st.error(f"The email is classified as: **{class_name}**")
        elif class_name == 'Harassment':
            st.warning(f"The email is classified as: **{class_name}**")
        elif class_name == 'Suspicious':
            st.info(f"The email is classified as: **{class_name}**")

    else:
        st.write("Please provide email text for classification.")



