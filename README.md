# CodeAlpha_Chatbot-for-FAQs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure you have NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample FAQs and responses
faq_data = {
    "What is your product?": "Our product is a smart home assistant that helps automate daily tasks.",
    "How does it work?": "It connects to your home devices and allows you to control them via voice commands.",
    "Is it compatible with all devices?": "Our product supports most smart home devices like lights, thermostats, and cameras.",
    "What is the price?": "The price varies depending on the model. Visit our website for the latest pricing.",
    "Where can I buy it?": "You can purchase it from our official website or leading e-commerce platforms."
}

# Preprocessing: Tokenization and stopword removal
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(tokens)

# Prepare data for similarity comparison
questions = list(faq_data.keys())
processed_questions = [preprocess_text(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

# Chatbot function
def chatbot_response(user_input):
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    
    similarities = cosine_similarity(input_vector, question_vectors)
    best_match_index = np.argmax(similarities)

    if similarities[0][best_match_index] > 0.3:  # Confidence threshold
        return faq_data[questions[best_match_index]]
    else:
        return "I'm sorry, I don't have an answer to that. Can you try rephrasing?"

# Chat loop
print("Chatbot: Hello! Ask me anything about our product.")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = chatbot_response(user_query)
    print(f"Chatbot: {response}")
