from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from stable_diffusion_pytorch import model_loader, pipeline
from PIL import Image
import pandas as pd
import logging
from textblob import TextBlob
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
mbti_model = joblib.load('/Users/yinshuodi/Desktop/code/my_flask_app/models/mbti_linear_svc_model.pkl')
vectorizer = joblib.load('/Users/yinshuodi/Desktop/code/my_flask_app/models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('/Users/yinshuodi/Desktop/code/my_flask_app/models/label_encoder.pkl')

# Load brand to MBTI mapping
brand_mbti_mapping = pd.read_csv('/Users/yinshuodi/Desktop/code/data/brand_mbti_mapping.csv')

# Preload Stable Diffusion model
models = model_loader.preload_models('cpu')
data = pd.read_csv('/Users/yinshuodi/Desktop/code/data/final_dataset_with_mbti_predictions.csv')

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Extract topics using LDA
def extract_topics(text, n_topics=1, n_words=3):
    count_vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
    X_counts = count_vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_counts)
    feature_names = count_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))
    return topics

# Convert sentiment to color
def sentiment_to_color(polarity, subjectivity, base_color):
    base_rgb = to_rgb(base_color)
    hsv_color = rgb_to_hsv(base_rgb)
    hsv_color[1] = max(0, min(1, hsv_color[1] + polarity))
    hsv_color[2] = max(0, min(1, hsv_color[2] + subjectivity))
    final_rgb = hsv_to_rgb(hsv_color)
    final_hex = "#{:02x}{:02x}{:02x}".format(int(final_rgb[0]*255), int(final_rgb[1]*255), int(final_rgb[2]*255))
    return final_hex

# Recommend brand and color
def recommend_brand(user_input, model, vectorizer, brand_mapping, label_encoder):
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_mbti_num = model.predict(user_input_vectorized)[0]
    predicted_mbti = label_encoder.inverse_transform([predicted_mbti_num])[0]

    if predicted_mbti not in brand_mapping.columns:
        raise ValueError(f"Predicted MBTI type {predicted_mbti} is not in the brand mapping columns.")
    
    recommended_brands = brand_mapping[['brand', predicted_mbti]].sort_values(by=predicted_mbti, ascending=False)['brand'].values
    
    mbti_color_mapping = {
        'INTP': 'black',
        'INFP': 'white',
        'INTJ': 'blue',
        'ENTJ': 'red',
        'ENTP': 'orange',
        'INFJ': 'purple',
        'ENFJ': 'green',
        'ENFP': 'yellow',
        'ISTJ': 'brown',
        'ISFJ': 'beige',
        'ESTJ': 'gray',
        'ESFJ': 'pink',
        'ISTP': 'teal',
        'ISFP': 'lavender',
        'ESTP': 'navy',
        'ESFP': 'coral'
    }
    base_color = mbti_color_mapping.get(predicted_mbti, 'grey')
    return predicted_mbti, recommended_brands, base_color

# Route to handle results
@app.route('/result', methods=['POST'])
def result():
    try:
        user_input = request.form['user_input']
        predicted_mbti, recommended_brands, base_color = recommend_brand(user_input, mbti_model, vectorizer, brand_mbti_mapping, label_encoder)
        
        sentiment_polarity, sentiment_subjectivity = analyze_sentiment(user_input)
        topics = extract_topics(user_input)
        
        # Calculate final color using sentiment analysis results and base color
        final_color = sentiment_to_color(sentiment_polarity, sentiment_subjectivity, base_color)
        
        # Debug output to confirm generated color
        print(f"Generated Color: {final_color}")
        
        # Pass final color to the template
        return render_template('result.html', 
                               user_input=user_input, 
                               mbti_type=predicted_mbti, 
                               recommended_brands=recommended_brands, 
                               recommended_colors=base_color, 
                               sentiment_polarity=sentiment_polarity, 
                               sentiment_subjectivity=sentiment_subjectivity, 
                               topics=topics, 
                               generated_color=final_color)  # 这里传递 final_color

    except Exception as e:
        logging.error(f"Error in result route: {e}")
        return "Error"

# Route to handle LDA analysis
@app.route('/lda', methods=['POST'])
def lda():
    try:
        mbti_type = request.form['mbti_type']
        count_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        X_counts = count_vectorizer.fit_transform(data[data['MBTI'] == mbti_type]['cleaned_text'])
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X_counts)
        
        def display_topics(model, feature_names, no_top_words):
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
            return topics
        
        topics = display_topics(lda, count_vectorizer.get_feature_names_out(), 10)
        return render_template('lda.html', mbti_type=mbti_type, topics=topics)
    except Exception as e:
        logging.error(f"Error in LDA route: {e}")
        return "Error"

# Route to generate images
@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        prompt = request.form['prompt']
        prompts = [prompt]
        logging.info(f"Generating image with prompt: {prompt}")
        image = pipeline.generate(prompts=prompts, uncond_prompts=None, input_images=[], strength=0.8, do_cfg=True, cfg_scale=7.5, height=512, width=512, sampler='k_lms', n_inference_steps=20, seed=None, models=models, device='cpu', idle_device='cpu')[0]
        logging.info("Image generation successful")

        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        img_data = base64.b64encode(img_io.getvalue()).decode()
        
        return render_template('image_result.html', img_data=img_data)
    
    except Exception as e:
        logging.error(f"Error in generate_image route: {e}")
        return render_template('image_result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
