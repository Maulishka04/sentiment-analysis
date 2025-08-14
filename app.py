import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.markdown("""
<style>
/* Add transition effects */
div[data-testid="stVerticalBlock"] {
    transition: all 0.5s ease-in-out;
}

/* Animate pie chart and bar chart */
svg {
    transition: transform 0.5s ease;
}

svg:hover {
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# Custom CSS styling
st.markdown("""
    <style>
    /* Remove extra padding at the top */
    .block-container {
        padding-top: 1rem !important;
    }
    .main {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [role="tab"] {
        background-color: #f0f2f6;
        padding: 10px;
        margin-right: 5px;
        border-radius: 10px 10px 0 0;
        font-weight: bold;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Use custom Google Font (Poppins)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# Load saved models and vectorizer
log_model = joblib.load('logistic_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
svm_model = joblib.load('svm_model.pkl')
#rf_model = joblib.load('random_forest_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

model_dict = {
    "Logistic Regression": log_model,
    "XGBoost": xgb_model,
    "Naive Bayes": nb_model,
    "SVM": svm_model,
    #"Random Forest": rf_model
}
#load model accuracies
with open("model_accuracies.json", "r") as f:
    model_accuracies = json.load(f)

# Load the dataset with cleaned text and sentiment labels
df = pd.read_csv("cleaned_reviews.csv")  # Or the file where you saved cleaned text
df = df[['Sentiment', 'Cleaned Text']]  # Only keep necessary columns

# Encode sentiment labels
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df = df.dropna(subset=['Cleaned Text'])  # Removes rows where Cleaned Text is NaN

# Prepare feature matrix and labels again
X = tfidf.transform(df['Cleaned Text'])  # Reuse the vectorizer
y = df['Sentiment']

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Encode sentiment labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

st.title("Sentiment Analysis Dashboard")
st.markdown("""
### 📊 Dataset Overview
This dashboard is built using the **Amazon Fine Food Reviews** dataset, which contains over 500,000 customer reviews from Amazon's food product category.  
The data includes text reviews and corresponding ratings, which have been labeled as:

- 🟢 **Positive**: Ratings of 4 or 5  
- 🟠 **Neutral**: Rating of 3  
- 🔴 **Negative**: Ratings of 1 or 2  

We’ve cleaned and preprocessed the reviews to build sentiment classification models and visualize trends.
""")

st.markdown("### 🧾 Sample Reviews from the Dataset")
st.dataframe(df.sample(5), use_container_width=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Accuracies", "🧠 Analyze Review", "☁️ Word Clouds", "📈 Sentiment Distribution"])

# TAB 1: Model Accuracy
with tab1:
    st.markdown("""
        <div style='background-color:#ffffff; padding: 2rem 1.5rem; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-top: 1rem;'>
            <h2 style='text-align: center; color: #4CAF50;'>📊 Model Accuracy Comparison</h2>
            <p style='text-align: center; color: grey; font-size: 16px;'>
                Visual comparison of all machine learning models trained on the Amazon Fine Food Reviews dataset
            </p>
            <hr style='border: 1px solid #f0f0f0;'/>
    """, unsafe_allow_html=True)

    # Define color for each model
    model_colors = {
        "Logistic Regression": "#A3BE8C",
        "Naive Bayes": "#88C0D0",
        "SVM": "#B48EAD",
        #"Random Forest": "#D08770",
        "XGBoost": "#EBCB8B"
    }

    fig, ax = plt.subplots(figsize=(6, 3))

    for i, (model, accuracy) in enumerate(zip(accuracy_df['Model'], accuracy_df['Accuracy'])):
        color = model_colors.get(model, "#888888")
        bar = ax.barh(i, accuracy, color=color)
        ax.text(accuracy + 0.005, i, f'{accuracy:.4f}', va='center', fontsize=10)

    ax.set_yticks(range(len(accuracy_df)))
    ax.set_yticklabels(accuracy_df['Model'])
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracies", fontsize=14)
    ax.invert_yaxis()
    
    st.pyplot(fig, use_container_width=False)

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Prepare comparison table
    model_metrics = []

    for name, model in model_dict.items():
        preds = model.predict(X_test)

        # 🔐 Safe decoding only if predictions are numeric (int or np.int64, etc.)
        if np.issubdtype(preds.dtype, np.integer):
            preds = le.inverse_transform(preds)
        
        model_metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, preds, average='weighted', zero_division=0)
        })

    metrics_df = pd.DataFrame(model_metrics)
    metrics_df = metrics_df.sort_values(by="Accuracy", ascending=False)

    # Style and display
    st.markdown("### 📋 Detailed Model Metrics")
    st.markdown("""
<div style='color: #555; font-size:15px; margin-top: -1rem; margin-bottom: 1.5rem;'>
🧠 The table below shows <strong>overall performance</strong> of each model across all sentiment classes (Positive, Neutral, Negative), 
using weighted average of metrics.
</div>
""", unsafe_allow_html=True)
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}"
}).background_gradient(cmap='YlGn'), use_container_width=True)

    from sklearn.metrics import classification_report

    # Loop through each model and show per-class metrics
    st.markdown("### 🧪 Class-wise Performance for Each Model")

    for name, model in model_dict.items():
        preds = model.predict(X_test)
    
        # Decode predictions to match y_test labels (which are strings)
        if isinstance(preds[0], (int, float, np.integer)):
            preds = le.inverse_transform(preds)

        st.markdown(f"""
    <div style='margin-top:2rem;'>
        <h5 style='color:#4CAF50;'>{name}</h5>
    </div>
    """, unsafe_allow_html=True)

        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: Review Analyzer
with tab2:
    st.markdown("""
        <div style='background-color:#ffffff; padding: 2rem 1.5rem; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-top: 1rem;'>
            <h2 style='text-align: center; color: #4CAF50;'>📝 Analyze a Review</h2>
            <p style='text-align: center; color: grey; font-size: 16px;'>
                Choose a model and type your review to predict sentiment.
            </p>
            <hr style='border: 1px solid #f0f0f0;'/>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3], gap="large")

    with col1:
        st.markdown("### 🔽 Select Model")
        model_choice = st.selectbox("", list(model_dict.keys()), key="model_sel")

    with col2:
        st.markdown("### ✍️ Enter your review")
        user_input = st.text_area("", height=150, key="review_text", placeholder="Write your product review here...")

    analyze_btn_col = st.columns([2, 1, 2])[1]
    with analyze_btn_col:
        analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

    if analyze:
        if user_input.strip():
            selected_model = model_dict[model_choice]
            cleaned_input = user_input.lower()
            vectorized_input = tfidf.transform([cleaned_input])
            prediction = selected_model.predict(vectorized_input)[0]

            sentiment_colors = {
                "Positive": "#c8f7c5",
                "Negative": "#f8c6c6",
                "Neutral":  "#f9f7b9"
            }

            st.markdown(
                f"""
                <div style="
                    margin-top: 2rem;
                    padding: 1.5rem;
                    border-radius: 12px;
                    background-color: {sentiment_colors[prediction]};
                    color: #222;
                    font-size: 18px;
                    font-weight: 600;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                ">
                    🎯 Predicted Sentiment: <span style="font-size:22px;">{prediction}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ Please enter a review before analyzing.")

    import random

    st.markdown("---")
    st.markdown("## 🎲 Show Me a Random Review")
    st.markdown("Get a randomly selected review from the dataset and see its predicted sentiment.")

    # Random review button
    if st.button("🔁 Show me a random review"):
        random_row = df.sample(1).iloc[0]
        random_text = random_row['Cleaned Text']
        actual_sentiment = random_row['Sentiment']
    
        # Predict with selected model
        selected_model = model_dict[model_choice]
        vectorized = tfidf.transform([random_text])
        predicted_sentiment = selected_model.predict(vectorized)[0]

        st.markdown(f"**📝 Review:** _{random_text}_")
    
        sentiment_colors = {
            "Positive": "#90ee90",
            "Negative": "#ff6b6b",
            "Neutral":  "#f3e979"
        }

        st.markdown(f"""
            <div style="
                margin-top: 1rem;
                padding: 1.2rem;
                border-radius: 10px;
                background-color: {sentiment_colors.get(predicted_sentiment, '#ddd')};
                font-weight: bold;
                text-align: center;
            ">
            🎯 Predicted Sentiment: {predicted_sentiment}  
            🧾 Actual Sentiment: {actual_sentiment}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


# TAB 3: Word Clouds
with tab3:
    st.markdown("""
        <div style='background-color:#ffffff; padding: 2rem 1.5rem; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-top: 1rem;'>
            <h2 style='text-align: center; color: #4CAF50;'>☁️ Word Clouds by Sentiment</h2>
            <p style='text-align: center; color: grey; font-size: 16px;'>
                Explore the most frequent words used in Positive, Negative, and Neutral reviews.
            </p>
            <hr style='border: 1px solid #f0f0f0;'/>
    """, unsafe_allow_html=True)

    def generate_wordcloud(data, title):
        text = data.dropna().astype(str)
        wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            max_font_size=40,
            scale=3,
            random_state=1,
            collocations=False,
            stopwords={'food', 'product', 'flavor', 'taste', 'amazon'}
        ).generate(" ".join(text))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        return fig

    # Positive Word Cloud
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h4 style="color: #2ecc71;">🌟 Positive Reviews</h4>
        </div>
    """, unsafe_allow_html=True)
    fig1 = generate_wordcloud(df[df['Sentiment'] == 'Positive']['Cleaned Text'], "Positive Word Cloud")
    st.pyplot(fig1)

    # Negative Word Cloud
    st.markdown("""
        <div style="margin-top: 3rem; margin-bottom: 2rem;">
            <h4 style="color: #e74c3c;">😠 Negative Reviews</h4>
        </div>
    """, unsafe_allow_html=True)
    fig2 = generate_wordcloud(df[df['Sentiment'] == 'Negative']['Cleaned Text'], "Negative Word Cloud")
    st.pyplot(fig2)

    # Neutral Word Cloud
    st.markdown("""
        <div style="margin-top: 3rem; margin-bottom: 2rem;">
            <h4 style="color: #f1c40f;">😐 Neutral Reviews</h4>
        </div>
    """, unsafe_allow_html=True)
    fig3 = generate_wordcloud(df[df['Sentiment'] == 'Neutral']['Cleaned Text'], "Neutral Word Cloud")
    st.pyplot(fig3)

    st.markdown("</div>", unsafe_allow_html=True)

# TAB 4: Sentiment Distribution
with tab4:
    st.markdown("""
        <div style='background-color:#ffffff; padding: 2rem 1.5rem; border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-top: 1rem;'>
            <h2 style='text-align: center; color: #4CAF50;'>📈 Sentiment Distribution</h2>
            <p style='text-align: center; color: grey; font-size: 16px;'>
                Visual representation of the distribution of sentiments across the dataset.
            </p>
            <hr style='border: 1px solid #f0f0f0;'/>
    """, unsafe_allow_html=True)

    sentiment_counts = df['Sentiment'].value_counts()
    col1, col2 = st.columns([1, 1])
    with col1:
        # Pie Chart
        st.markdown("### 🥧 Pie Chart")
        fig1, ax1 = plt.subplots(figsize=(5, 5), dpi=100)  # fixed size
        wedges, texts, autotexts = ax1.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=['#4CAF50', '#9E9E9E', '#F44336'],
            textprops={'fontsize': 12}
        )
        ax1.axis('equal')
        st.pyplot(fig1, use_container_width=False)
    with col2:
        # Bar Chart
        st.markdown("### 📊 Bar Chart")
        fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=100)  # fixed size
        bars = ax2.bar(
            sentiment_counts.index,
            sentiment_counts.values,
            color=['#4CAF50', '#9E9E9E', '#F44336'],
            edgecolor='black'
        )
        ax2.set_ylabel("Number of Reviews", fontsize=12)
        ax2.set_title("Sentiment Count", fontsize=14, pad=15)
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11)
        st.pyplot(fig2, use_container_width=False)

    total = sentiment_counts.sum()
    pos = sentiment_counts.get("Positive", 0)
    neg = sentiment_counts.get("Negative", 0)
    neu = sentiment_counts.get("Neutral", 0)

    st.markdown(f"""
<div style="margin-top:1rem; text-align:center; font-size:16px; color:#555;">
    ✅ Out of <strong>{total:,}</strong> total reviews:
    <ul style="list-style:none; padding:0;">
        <li>🟢 <strong>{pos:,}</strong> are Positive</li>
        <li>🔴 <strong>{neg:,}</strong> are Negative</li>
        <li>🟠 <strong>{neu:,}</strong> are Neutral</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)