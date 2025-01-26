import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import pickle
import os
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict

st.set_page_config(
    page_title="AgroVision",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Web-Enabled Chatbot Class
class WebEnabledChatbot:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3600  # Cache results for 1 hour
        self.ddgs = DDGS()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,?!]', '', text)
        return text.strip()

    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search the web using DuckDuckGo API"""
        enhanced_query = f"agriculture farming {query}"
        try:
            results = list(self.ddgs.text(enhanced_query, max_results=num_results))
            return results if results else []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
            if main_content:
                return self.clean_text(main_content.get_text())
            return self.clean_text(soup.get_text())
        except Exception as e:
            print(f"Extraction error for {url}: {e}")
            return ""

    def get_relevant_snippet(self, content: str, query: str, max_length: int = 250) -> str:
        """Extract most relevant snippet from content"""
        sentences = re.split(r'[.!?]+', content)
        relevant_sentences = []
        query_terms = query.lower().split()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                score = sum(1 for term in query_terms if term in sentence.lower())
                if score > 0:
                    relevant_sentences.append((sentence, score))
        
        if relevant_sentences:
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            snippet = relevant_sentences[0][0]
            if len(snippet) > max_length:
                snippet = snippet[:max_length] + "..."
            return snippet
        return ""

    def format_response(self, search_results: List[Dict], query: str) -> str:
        """Format search results into a coherent response"""
        if not search_results:
            return simple_chatbot(query)  # Fallback to simple chatbot if no web results
        
        response_parts = ["Here's what I found from reliable sources:\n"]
        
        for result in search_results:
            snippet = result.get('body', '')
            if snippet:
                relevant_info = self.get_relevant_snippet(snippet, query)
                if relevant_info:
                    response_parts.append(f"• {relevant_info}")
        
        response_parts.append("\nℹ️ Information compiled from agricultural resources and farming databases.")
        
        return "\n".join(response_parts)

    def get_response(self, user_input: str) -> str:
        """Generate response based on web search results"""
        if user_input in self.cache:
            cache_time, response = self.cache[user_input]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                return response
        
        search_results = self.search_web(user_input)
        response = self.format_response(search_results, user_input)
        self.cache[user_input] = (datetime.now(), response)
        return response

def simple_chatbot(user_input: str) -> str:
    """Fallback chatbot for basic queries"""
    user_input = user_input.lower()
    
    responses = {
        "farming": "Farming is the practice of cultivating soil, growing crops, and raising animals for food, fiber, and other products.",
        "crop": "Crops are plants cultivated for food, fiber, or other uses. Common crops include rice, wheat, and corn.",
        "disease": "Plant diseases can be caused by pathogens like fungi, bacteria, and viruses. Early detection is key!",
        "help": "How can I assist you? You can ask about farming practices, crop recommendations, or plant diseases.",
        "pest": "Pest management is essential for healthy crops. Use integrated pest management (IPM) techniques.",
        "sustainability": "Sustainable farming practices help preserve the environment and ensure long-term productivity.",
        "technology": "Smart farming technologies like drones and IoT sensors can optimize crop monitoring and irrigation.",
        "weather": "For accurate weather updates, please check local weather services or apps tailored for agricultural needs."
    }

    for key, response in responses.items():
        if key in user_input:
            return response

    return "I'm sorry, I don't understand that. Can you ask something else?"

# Enhanced Custom CSS
def load_custom_css():
    st.markdown("""
        <style>
        /* Global Styles */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f0f5f0;
            color: #2c3e50;
        }
        
        /* Main Header */
        .main-header {
            background: linear-gradient(135deg, #1eb563, #128547);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .main-title {
            color: white;
            font-size: 3.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
        }
        
        /* Cards */
        .custom-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Add the rest of your CSS styles here */
        </style>
    """, unsafe_allow_html=True)

# Database initialization
def init_db():
    conn = sqlite3.connect('agrovision.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, email TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  title TEXT,
                  content TEXT,
                  category TEXT,
                  timestamp DATETIME,
                  FOREIGN KEY (username) REFERENCES users(username))''')
    c.execute('''CREATE TABLE IF NOT EXISTS comments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  post_id INTEGER,
                  username TEXT,
                  content TEXT,
                  timestamp DATETIME,
                  FOREIGN KEY (post_id) REFERENCES posts(id),
                  FOREIGN KEY (username) REFERENCES users(username))''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Load models and resources
@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model('cnn_model.keras')

@st.cache_resource
def load_crop_model():
    return pickle.load(open('model.pkl', 'rb'))

@st.cache_resource
def load_scalers():
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    return sc, mx

@st.cache_resource
def load_class_names():
    with open('class_names.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_disease_solutions():
    with open('disease_solutions.json', 'r') as f:
        return json.load(f)

# Load all resources
plant_model = load_plant_model()
crop_model = load_crop_model()
scaler, minmax_scaler = load_scalers()
class_names = load_class_names()
disease_solutions = load_disease_solutions()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email):
    conn = sqlite3.connect('agrovision.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", 
                 (username, hash_password(password), email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('agrovision.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

# Forum functions
def create_post(username, title, content, category):
    conn = sqlite3.connect('agrovision.db')
    c = conn.cursor()
    c.execute("INSERT INTO posts (username, title, content, category, timestamp) VALUES (?, ?, ?, ?, ?)",
             (username, title, content, category, datetime.now()))
    conn.commit()
    conn.close()

def get_posts():
    conn = sqlite3.connect('agrovision.db')
    posts = pd.read_sql_query("SELECT * FROM posts ORDER BY timestamp DESC", conn)
    conn.close()
    return posts

def add_comment(post_id, username, content):
    conn = sqlite3.connect('agrovision.db')
    c = conn.cursor()
    c.execute("INSERT INTO comments (post_id, username, content, timestamp) VALUES (?, ?, ?, ?)",
             (post_id, username, content, datetime.now()))
    conn.commit()
    conn.close()

def get_comments(post_id):
    conn = sqlite3.connect('agrovision.db')
    comments = pd.read_sql_query("SELECT * FROM comments WHERE post_id=? ORDER BY timestamp",
                               conn, params=(post_id,))
    conn.close()
    return comments

# Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'web_chatbot' not in st.session_state:
    st.session_state.web_chatbot = WebEnabledChatbot()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Authentication UI
def show_login_page():
    with st.sidebar:
        st.markdown('<h2 class="section-header">👤 Account</h2>', unsafe_allow_html=True)
        
        if not st.session_state.logged_in:
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    submit = st.form_submit_button("Login")
                    if submit:
                        if verify_user(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
            
            with tab2:
                with st.form("register_form"):
                    new_username = st.text_input("Username", key="register_username")
                    new_password = st.text_input("Password", type="password", key="register_password")
                    email = st.text_input("Email")
                    submit = st.form_submit_button("Register")
                    if submit:
                        if create_user(new_username, new_password, email):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Username already exists")
        
        else:
            st.markdown(f"### Welcome, {st.session_state.username}! 👋")
            if st.button("Logout", key="logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.rerun()

# Chatbot Tab
def chatbot_tab():
    st.markdown('<h2 class="section-header">🤖 AgroVision AI Assistant</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    💡 Try asking about:
    - Current crop prices and market trends
    - Latest farming techniques and technologies
    - Weather impacts on agriculture
    - Pest control methods
    - Sustainable farming practices
    """)

    user_input = st.text_input("Ask your farming question:", key="chat_input", 
                              placeholder="e.g., What are the best practices for organic farming?")
    
    if st.button("Get Answer"):
        if user_input:
            with st.spinner("Searching agricultural resources..."):
                response = st.session_state.web_chatbot.get_response(user_input)
                st.session_state.chat_history.append({"user": user_input, "bot": response})
    
    if st.session_state.chat_history:
        st.markdown("### Recent Conversations")
        for chat in reversed(st.session_state.chat_history):
            with st.container():
                st.markdown("---")
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown("👤")
                with col2:
                    st.markdown(f"**Question:** {chat['user']}")
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown("🤖")
                with col2:
                    st.markdown(f"**Answer:** {chat['bot']}")

# Main Application
def main():
    load_custom_css()
    show_login_page()
    
    # Main navigation
    tabs = st.tabs(["🏠 Home", "🌾 Crop Recommendation", "🔍 Disease Detection", "💬 Discussion", "🤖 Chatbot"])
    
    # Home Tab
    with tabs[0]:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-title">🌿 AgroVision</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="color: white;">Your Smart Farming Assistant</h3>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Features")
            st.markdown("""
            - Intelligent Crop Recommendations
            - Plant Disease Detection
            - Community Discussion Forum
            - Expert Farming Advice
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🌟 Benefits")
            st.markdown("""
            - Increase Crop Yield
            - Early Disease Detection
            - Community Knowledge Sharing
            - Data-Driven Decisions
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display main image
        try:
            img = Image.open("homepageimg.jpeg")
            img_resized = img.resize((int(img.width * 0.4), int(img.height * 0.4)))
            st.image(img_resized, caption="Smart Farming Solutions", use_container_width=True)
        except Exception as e:
            st.warning("Homepage image not found. Please ensure 'homepageimg.jpeg' is in the application directory.")
    
    # Crop Recommendation Tab
    with tabs[1]:
        st.markdown('<h2 class="section-header">🌱 Crop Recommendation System</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                N = st.number_input("Nitrogen (N)", min_value=0, max_value=1000, value=50)
                P = st.number_input("Phosphorus (P)", min_value=0, max_value=1000, value=50)
                K = st.number_input("Potassium (K)", min_value=0, max_value=1000, value=50)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=1000.0, value=60.0)
            
            with col2:
                temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
                ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0)

            if st.button("Get Recommendation"):
                feature_list = [N, P, K, temp, humidity, ph, rainfall]
                single_pred = np.array(feature_list).reshape(1, -1)
                mx_features = minmax_scaler.transform(single_pred)
                sc_mx_features = scaler.transform(mx_features)
                
                prediction = crop_model.predict(sc_mx_features)
                crop_dict = {
                    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
                }
                crop = crop_dict.get(prediction[0], "Unknown Crop")
                st.success(f"Based on the provided conditions, the recommended crop is: **{crop}**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Disease Detection Tab
    with tabs[2]:
        st.markdown('<h2 class="section-header">🔍 Plant Disease Detection</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a leaf image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                img_array = preprocess_image(image)
                prediction = plant_model.predict(img_array)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_label = class_names[predicted_class[0]]
                confidence = np.max(prediction) * 100
                
                st.markdown("### Analysis Results")
                st.success(f"Detected Condition: **{predicted_label}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
                
                if predicted_label in disease_solutions:
                    solution = disease_solutions[predicted_label]
                    st.markdown("### Treatment Recommendations")
                    st.write(f"**Solution:** {solution['solution']}")
                    st.write(f"**Prevention:** {solution['prevention']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Discussion Forum Tab
    with tabs[3]:
        st.markdown('<h2 class="section-header">💬 Community Discussion</h2>', unsafe_allow_html=True)
        
        if not st.session_state.logged_in:
            st.warning("Please login to participate in discussions")
        else:
            # Create new post
            with st.expander("Create New Post"):
                st.markdown('<div class="form-container">', unsafe_allow_html=True)
                post_title = st.text_input("Title")
                category = st.selectbox("Category", ["General", "Crop Issues", "Disease Help", "Best Practices", "Market Discussion"])
                post_content = st.text_area("Content")
                if st.button("Submit Post"):
                    if post_title and post_content:
                        create_post(st.session_state.username, post_title, post_content, category)
                        st.success("Post created successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill in all fields")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display posts
            st.markdown("### Recent Discussions")
            posts = get_posts()
            
            for _, post in posts.iterrows():
                st.markdown('<div class="forum-post">', unsafe_allow_html=True)
                st.markdown(f'<div class="post-header">', unsafe_allow_html=True)
                st.markdown(f'<span class="post-title">{post["title"]}</span>', unsafe_allow_html=True)
                st.markdown(f'<p class="post-meta">Posted by {post["username"]} in {post["category"]} • {post["timestamp"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.write(post['content'])
                
                # Comments section
                comments = get_comments(post['id'])
                with st.expander(f"Comments ({len(comments)})"):
                    for _, comment in comments.iterrows():
                        st.markdown('<div class="comment">', unsafe_allow_html=True)
                        st.markdown(f"**{comment['username']}**: {comment['content']}")
                        st.markdown(f"<small>{comment['timestamp']}</small>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    new_comment = st.text_area("Add a comment", key=f"comment_{post['id']}")
                    if st.button("Submit Comment", key=f"submit_{post['id']}"):
                        if new_comment:
                            add_comment(post['id'], st.session_state.username, new_comment)
                            st.success("Comment added!")
                            st.rerun()
                        else:
                            st.error("Please enter a comment")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Chatbot Tab
    with tabs[4]:
        chatbot_tab()

if __name__ == "__main__":
    main()