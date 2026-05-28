import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os
import nltk
import tensorflow as tf
import zipfile

# Set page config with premium aesthetics
st.set_page_config(
    page_title="DeepEmotion AI - BiLSTM Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize NLTK Tokenizers
@st.cache_resource(show_spinner="Initializing NLTK tokenizer...")
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

setup_nltk()
from nltk.tokenize import word_tokenize

# Define categories, emojis, and harmonious colors
categories = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
emojis = {
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'guilt': '😳',
    'joy': '😊',
    'sadness': '😢',
    'shame': '🫣'
}
colors_map = {
    'anger': '#EF4444',     # Vibrant red
    'disgust': '#10B981',   # Emerald green
    'fear': '#F59E0B',     # Amber yellow
    'guilt': '#8B5CF6',    # Purple
    'joy': '#EC4899',      # Pink
    'sadness': '#3B82F6',  # Bright blue
    'shame': '#6B7280'     # Cool gray
}

# ----------------- Custom Styling -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* Core fonts */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Title style */
    .title-gradient {
        background: linear-gradient(135deg, #EC4899 0%, #8B5CF6 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #6B7280;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Glassmorphic elements */
    .premium-card {
        border-radius: 16px;
        padding: 24px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
    }
    
    [data-theme="dark"] .premium-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.8);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .emotion-hero {
        text-align: center;
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 20px;
        font-weight: 600;
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.3);
    }
    
    .highlight-box {
        background: #F1F5F9;
        border-left: 5px solid #6366F1;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin: 15px 0;
        color: #334155;
    }
    [data-theme="dark"] .highlight-box {
        background: #1E293B;
        color: #E2E8F0;
    }
    .highlight-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 12px;
    }
    [data-theme="dark"] .highlight-title {
        color: #818CF8;
    }
    .highlight-item {
        font-size: 0.95rem;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px dashed rgba(0,0,0,0.05);
        padding-bottom: 6px;
    }
    [data-theme="dark"] .highlight-item {
        border-bottom: 1px dashed rgba(255,255,255,0.05);
    }
    .highlight-item:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .highlight-label {
        font-weight: 600;
    }
    .highlight-value {
        font-family: monospace;
        background: rgba(99, 102, 241, 0.1);
        color: #4F46E5;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    [data-theme="dark"] .highlight-value {
        background: rgba(129, 140, 248, 0.2);
        color: #A5B4FC;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Resource Loading -----------------

@st.cache_resource(show_spinner="Loading pre-trained BiLSTM model...")
def load_emotion_model():
    model_path = "bilstm_emotion_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in current directory. Please make sure the trained model file is in the workspace.")
        return None
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource(show_spinner="Loading GloVe Word Embeddings...")
def load_glove_vectors():
    txt_path = "glove.6B.50d.txt"
    zip_path = "glove.6B.50d.zip"
    
    embeddings_index = {}
    
    if os.path.exists(txt_path):
        with open(txt_path, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            txt_filename = "glove.6B.50d.txt"
            if txt_filename in z.namelist():
                with z.open(txt_filename) as f:
                    for line in f:
                        line_decoded = line.decode('utf-8').rstrip()
                        values = line_decoded.rsplit(' ')
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
            else:
                st.error(f"'{txt_filename}' not found inside '{zip_path}'.")
                return None
    else:
        st.error("Neither 'glove.6B.50d.txt' nor 'glove.6B.50d.zip' was found in the workspace.")
        return None
        
    return embeddings_index

@st.cache_data(show_spinner="Loading training samples from ISEAR dataset...")
def load_isear_dataset():
    dataset_path = "isear.csv"
    if not os.path.exists(dataset_path):
        return None
    try:
        df = pd.read_csv(dataset_path, header=None)
        # Clean [ No response.] fields
        df.drop(df[df[1] == '[ No response.]'].index, inplace=True)
        df.columns = ["Emotion", "Text"]
        df["Emotion"] = df["Emotion"].str.lower().str.strip()
        df["Text"] = df["Text"].str.strip()
        return df
    except Exception:
        return None

# Load resources
model = load_emotion_model()
embeddings_index = load_glove_vectors()
isear_df = load_isear_dataset()

# Check if model or embeddings loaded successfully
if model is None or embeddings_index is None:
    st.stop()

# ----------------- Preprocessing & Prediction Pipeline -----------------

def preprocess_and_predict(text):
    max_sentence_length = 100
    embedding_dim = 50
    
    # 1. Tokenize using NLTK
    tokens = word_tokenize(text)
    
    # 2. Pad / Truncate sentence to exactly max_sentence_length
    if len(tokens) < max_sentence_length:
        padded_tokens = tokens + ['<pad>'] * (max_sentence_length - len(tokens))
    else:
        padded_tokens = tokens[:max_sentence_length]
        
    # 3. Create embedded sequence using GloVe word vectors
    sentence_embeddings = []
    for word in padded_tokens:
        word_lower = word.lower()
        if word_lower in embeddings_index:
            sentence_embeddings.append(embeddings_index[word_lower])
        else:
            sentence_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            
    # Convert to standard 3D array of shape (1, 100, 50) for BiLSTM inference
    X_input = np.array([sentence_embeddings], dtype=np.float32)
    
    # Run prediction
    prediction = model.predict(X_input)[0]
    return prediction

# ----------------- UI Layout & Content -----------------

# Header Section
col_logo, col_title = st.columns([1, 12])
with col_title:
    st.markdown('<div class="title-gradient">DeepEmotion AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A High-Performance Bidirectional LSTM Recurrent Neural Network for Human Emotion Sequence Classification</div>', unsafe_allow_html=True)

# Main Application Columns
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown('### 🧠 Analyze Text Emotion')
    st.markdown("Enter a sentence below to witness the neural network decode the speaker's emotional state in real-time.")
    
    user_input = st.text_area(
        "📝 Your Sentence:",
        value="",
        height=120,
        placeholder="Type a sentence here (e.g. 'I was so happy when I heard the news!')...",
        key="emotion_input"
    )
    
    analyze_btn = st.button("🚀 Analyze Emotional State", use_container_width=True)

    if user_input:
        with st.spinner("Model inference in progress..."):
            probs = preprocess_and_predict(user_input)
            
        max_idx = np.argmax(probs)
        pred_emotion = categories[max_idx]
        pred_confidence = probs[max_idx]
        
        st.markdown("---")
        
        # Dominant Emotion Hero Card
        hero_color = colors_map[pred_emotion]
        st.markdown(
            f"""
            <div class="emotion-hero" style="background: linear-gradient(135deg, {hero_color} 0%, #1E1B4B 100%);">
                <span style='font-size: 3.5rem;'>{emojis[pred_emotion]}</span>
                <h2 style='color: white; margin: 5px 0 0 0; text-transform: uppercase; font-size: 2rem; font-weight: 800;'>{pred_emotion}</h2>
                <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 1.1rem;'>
                    Classified with {pred_confidence*100:.1f}% Model Confidence
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Plotly chart for premium metrics visualization
        chart_df = pd.DataFrame({
            'Emotion': [cat.capitalize() for cat in categories],
            'Probability': [float(p) for p in probs],
            'Color': [colors_map[cat] for cat in categories]
        }).sort_values('Probability', ascending=True)

        fig = px.bar(
            chart_df,
            x='Probability',
            y='Emotion',
            orientation='h',
            text=chart_df['Probability'].apply(lambda x: f"{x*100:.1f}%"),
            color='Emotion',
            color_discrete_map={cat.capitalize(): col for cat, col in colors_map.items()},
            title="Categorical Probability Distribution"
        )
        
        fig.update_layout(
            showlegend=False,
            height=340,
            margin=dict(l=0, r=40, t=40, b=0),
            xaxis=dict(title='Confidence', range=[0, 1.05], tickformat='.0%'),
            yaxis=dict(title=''),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(family='Outfit', size=16, color='#6B7280')
        )
        fig.update_traces(
            textposition='outside', 
            textfont=dict(size=12, weight='bold'),
            cliponaxis=False
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with right_col:
    # Sidebar/Right-column Panel info
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Model Diagnostics")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric(label="Model Architecture", value="BiLSTM")
    with col_stat2:
        st.metric(label="Embedding Layer", value="GloVe 50d")
        
    st.markdown(
        """
        <div class="highlight-box">
            <div class="highlight-title">Pipeline Configuration</div>
            <div class="highlight-item">
                <span class="highlight-label">Sequence Padding</span>
                <span class="highlight-value">100 words</span>
            </div>
            <div class="highlight-item">
                <span class="highlight-label">Word Vectors</span>
                <span class="highlight-value">glove.6B.50d</span>
            </div>
            <div class="highlight-item">
                <span class="highlight-label">Vocabulary Size</span>
                <span class="highlight-value">400k keys</span>
            </div>
            <div class="highlight-item">
                <span class="highlight-label">BiLSTM Hidden Units</span>
                <span class="highlight-value">100</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ISEAR Dataset Explorer
    if isear_df is not None:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 📂 ISEAR Dataset Explorer")
        st.markdown("Explore actual sentences from the model's training dataset.")
        
        explore_emotion = st.selectbox(
            "Select Emotion to browse samples:", 
            [cat.capitalize() for cat in categories], 
            key="dataset_select"
        )
        
        # Get random sample of selected emotion
        filtered_df = isear_df[isear_df['Emotion'] == explore_emotion.lower()]
        
        if not filtered_df.empty:
            if st.button("🎲 Draw Another Sample", key="draw_sample_btn"):
                st.session_state.sample_index = np.random.randint(0, len(filtered_df))
            elif "sample_index" not in st.session_state or st.session_state.get("prev_emotion") != explore_emotion:
                st.session_state.sample_index = np.random.randint(0, len(filtered_df))
                st.session_state.prev_emotion = explore_emotion
                
            sample_text = filtered_df.iloc[st.session_state.sample_index]['Text']
            
            st.markdown(
                f"""
                <blockquote style="font-style: italic; font-size: 1.05rem; padding: 10px 20px; margin: 10px 0; border-left: 4px solid {colors_map[explore_emotion.lower()]}; background: rgba(0,0,0,0.02); border-radius: 4px;">
                    "{sample_text}"
                </blockquote>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"<small>Showing sample {st.session_state.sample_index + 1} of {len(filtered_df)}</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Embedding Vocabulary Explorer
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Embedding Lookup")
    st.markdown("Check if a specific word exists in GloVe vocabulary and inspect its values.")
    
    lookup_word = st.text_input("Enter a single word:", "ambition", key="lookup_input")
    
    if lookup_word:
        clean_word = lookup_word.strip().lower()
        if clean_word in embeddings_index:
            vector = embeddings_index[clean_word]
            st.success(f"✓ **'{clean_word}'** found in GloVe vocabulary!")
            st.markdown(f"**First 8 components of the 50d vector:**")
            
            # Format vector beautifully
            vector_str = "  ".join([f"`{val:.4f}`" for val in vector[:8]])
            st.markdown(vector_str)
        else:
            st.warning(f"✗ **'{clean_word}'** not found in GloVe. It will be represented as a zero-vector (`0.0`).")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; font-size: 0.85rem; padding: 10px 0;">
        DeepEmotion AI Frontend • Powered by Streamlit, Keras, TensorFlow and GloVe Word Embeddings.
    </div>
    """,
    unsafe_allow_html=True
)
