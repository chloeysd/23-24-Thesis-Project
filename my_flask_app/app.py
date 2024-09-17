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
import csv
from datetime import datetime

#initialize Flask app,ref:https://python-adv-web-apps.readthedocs.io/en/latest/flask.html
app = Flask(__name__)

#load the four SVC models
svc_model_EI = joblib.load('saved_models/svc_model_EI.pkl')
svc_model_NS = joblib.load('saved_models/svc_model_NS.pkl')
svc_model_FT = joblib.load('saved_models/svc_model_FT.pkl')
svc_model_PJ = joblib.load('saved_models/svc_model_PJ.pkl')
#load the vectorizer
vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')
#load brand to MBTI mapping
brand_mbti_mapping = pd.read_csv('data/brand_mbti_mapping.csv')

#preload Stable Diffusion model,ref:https://github.com/kjsman/stable-diffusion-pytorch
models = model_loader.preload_models('cpu')
data = pd.read_csv('data/final_dataset.csv')

#set up basic logging,ref:https://stackoverflow.com/questions/38537905/set-logging-levels
logging.basicConfig(level=logging.INFO)

#route for the home page,ref:https://www.geeksforgeeks.org/flask-rendering-templates/
@app.route('/')
def index():
    return render_template('index.html')

#sentiment analysis function
def analyze_sentiment(text):
    #creates a TextBlob object from the input text
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

#extract topics using LDA,ref:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def extract_topics(text, n_topics=1, n_words=3):
    #initialize CountVectorizer to convert the text into a term-frequency matrix
    count_vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
    #transform the input text into a term-frequency matrix 
    X_counts = count_vectorizer.fit_transform([text])
    #create an instance of LLDA to identify topics
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    #fit the LDA model to the term-frequency matrix 
    lda.fit(X_counts)
    #get the feature names from the CountVectorizer, which correspond to the terms used
    feature_names = count_vectorizer.get_feature_names_out()
    #initialize an empty list
    topics = []
    #iterate over the topics in the LDA model
    for topic_idx, topic in enumerate(lda.components_):
        #for each topic, get the top words based on their weights in descending order
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))
    return topics

#convert sentiment to color,ref:https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.hsv_to_rgb.html
def sentiment_to_color(polarity, subjectivity, base_color):
    #convert the base color from hex to RGB format
    base_rgb = to_rgb(base_color)
    #convert the RGB color to HSV format
    hsv_color = rgb_to_hsv(base_rgb)
    # adjust the saturation by adding the polarity value
    hsv_color[1] = max(0, min(1, hsv_color[1] + polarity))
    #adjust the brightness by adding the subjectivity value
    hsv_color[2] = max(0, min(1, hsv_color[2] + subjectivity))
    #convert the adjusted HSV color back to RGB format
    final_rgb = hsv_to_rgb(hsv_color)
    #convert the final RGB values to hexadecimal format for color representation
    final_hex = "#{:02x}{:02x}{:02x}".format(int(final_rgb[0]*255), int(final_rgb[1]*255), int(final_rgb[2]*255))
    return final_hex

#recommend brand and color
def recommend_brand(user_input, model_EI, model_NS, model_FT, model_PJ, vectorizer, brand_mapping):
    #perform MBTI prediction using the four models
    predicted_mbti = predict_mbti(user_input, vectorizer, model_EI, model_NS, model_FT, model_PJ)
    #recommend a brand based on the predicted MBTI type
    if predicted_mbti not in brand_mapping.columns:
        #raise an error if the MBTI type is not found in the brand mapping
        raise ValueError(f"Predicted MBTI type {predicted_mbti} is not in the brand mapping columns.")
    #sort the brands based on the predicted MBTI type in descending order and return the brand names
    recommended_brands = brand_mapping[['brand', predicted_mbti]].sort_values(by=predicted_mbti, ascending=False)['brand'].values
    #define a mapping of MBTI types to base colors
    mbti_color_mapping = {'INTP': 'black','INFP': 'white','INTJ': 'blue','ENTJ': 'red','ENTP': 'orange','INFJ': 'purple','ENFJ': 'green','ENFP': 'yellow','ISTJ': 'brown','ISFJ': 'beige','ESTJ': 'gray','ESFJ': 'pink','ISTP': 'teal','ISFP': 'lavender','ESTP': 'navy','ESFP': 'coral'}
    #retrieve the base color based on the predicted MBTI type, default to 'grey' if not found
    base_color = mbti_color_mapping.get(predicted_mbti, 'grey')
    return predicted_mbti, recommended_brands, base_color

#predict MBTI based on four models
def predict_mbti(user_input, vectorizer, model_EI, model_NS, model_FT, model_PJ):
    #vectorize the user input using the provided vectorizer,ref:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    user_input_vectorized = vectorizer.transform([user_input])
    #use each of the four models to make predictions on the vectorized user input
    predicted_EI = model_EI.predict(user_input_vectorized)[0]
    predicted_NS = model_NS.predict(user_input_vectorized)[0]
    predicted_FT = model_FT.predict(user_input_vectorized)[0]
    predicted_PJ = model_PJ.predict(user_input_vectorized)[0]
    #combine the four dimensions to form a complete MBTI type
    predicted_mbti = predicted_EI + predicted_NS + predicted_FT + predicted_PJ
    return predicted_mbti

#this function processes user input from a form, predicts the MBTI type, analyzes the sentiment, extracts topics, adjusts a color based on sentiment, and renders the results to a template for display
@app.route('/result', methods=['POST'])
def result():
    #retrieve the user input from the form submission,ref:https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request.form
    user_input = request.form['user_input']
    #use the four models to predict the MBTI type and recommend brands and a base color
    predicted_mbti, recommended_brands, base_color = recommend_brand(user_input, svc_model_EI, svc_model_NS, svc_model_FT, svc_model_PJ, vectorizer, brand_mbti_mapping)
    #analyze the sentiment of the user input
    sentiment_polarity, sentiment_subjectivity = analyze_sentiment(user_input)
    #extract topics from the user input text using LDA
    topics = extract_topics(user_input)
    #compute the final color by adjusting the base color using sentiment polarity and subjectivity
    final_color = sentiment_to_color(sentiment_polarity, sentiment_subjectivity, base_color)
    print(f"Generated Color: {final_color}")
    #render the 'result.html' template and pass the necessary values to the template
    #ref:https://www.geeksforgeeks.org/flask-rendering-templates/
    return render_template('result.html', user_input=user_input, mbti_type=predicted_mbti, recommended_brands=recommended_brands, recommended_colors=final_color, sentiment_polarity=sentiment_polarity, sentiment_subjectivity=sentiment_subjectivity, topics=topics, generated_color=final_color)

#route to LDA analysis
@app.route('/lda', methods=['POST'])
def lda():
    #retrieve the MBTI type from the form submission,ref:https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request.form
    mbti_type = request.form['mbti_type']
    #filter the dataset to only include rows where the 'MBTI' column matches the specified MBTI type
    filtered_data = data[data['MBTI'] == mbti_type]['cleaned_text']
    #further filter the data by keeping only non-empty strings,ref:https://www.w3schools.com/python/ref_func_isinstance.asp
    filtered_data = filtered_data[filtered_data.apply(lambda x: isinstance(x, str) and x.strip() != '')] 
    #convert the filtered text data into a term-frequency matrix using CountVectorizer,ref:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    count_vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X_counts = count_vectorizer.fit_transform(filtered_data)
    #initialize and fit the LDA model to the term-frequency matrix
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X_counts)
        
    #function to display the topics found by the LDA model
    def display_topics(model, feature_names, no_top_words):
        topics = []
        #for each topic, get the top words with the highest weights
        for topic_idx, topic in enumerate(model.components_):
            topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        return topics
    #extract and display the top 10 words for each topic
    topics = display_topics(lda, count_vectorizer.get_feature_names_out(), 10)
    #render the 'lda.html' template and pass the MBTI type and topics to the template
    #ref:https://www.geeksforgeeks.org/flask-rendering-templates/
    return render_template('lda.html', mbti_type=mbti_type, topics=topics)
    
#route to generate images
@app.route('/generate_image', methods=['POST'])
def generate_image():
    #retrieve the prompt from the form submission,ref:https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request.form
    prompt = request.form['prompt']
    #create a list of prompts with the retrieved prompt
    prompts = [prompt]
    #log the prompt for debugging,ref:https://blog.csdn.net/weixin_39278265/article/details/115203933
    logging.info(f"Generating image with prompt: {prompt}")
    #generate the image using the stable diffusion pipeline with specified parameters,ref:https://huggingface.co/docs/diffusers/index
    image = pipeline.generate(prompts=prompts, uncond_prompts=None, input_images=[], strength=0.8, do_cfg=True, cfg_scale=7.5, height=512, width=512, sampler='k_lms', n_inference_steps=20, seed=None, models=models, device='cpu', idle_device='cpu')[0]
    logging.info("Image generation successful")
    #create a bytes buffer to save the generated image as a PNG,ref:https://imageio.readthedocs.io/en/stable/examples.html
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    #reset the buffer's pointer to the beginning,ref:https://stackoverflow.com/questions/66246256/image-seek-doesnt-return-image-object
    img_io.seek(0)
    #encode the image in base64 for rendering in the HTML template,ref:https://www.geeksforgeeks.org/base64-b64encode-in-python/
    img_data = base64.b64encode(img_io.getvalue()).decode()
    return render_template('image_result.html', img_data=img_data)

@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    #retrieve the user's rating from the form submission,ref:https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request.form
    rating = request.form['rating']
    #log the rating for debugging or tracking purposes,ref:https://blog.csdn.net/weixin_39278265/article/details/115203933
    logging.info(f"User submitted a rating: {rating} stars")
    #get the current time and format it as a string,ref:https://www.programiz.com/python-programming/datetime/strftime
    submission_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
    #save to csv
    with open('ratings.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        #write a new row in the CSV file with the submission time and rating
        writer.writerow([submission_time, rating]) 
        return render_template('thank_you.html', rating=rating)

if __name__ == '__main__':
    #uf this script is run directly, start the Flask application
    app.run(debug=True, host='127.0.0.1', port=8000)