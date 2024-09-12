# Personalized Fashion Recommendation System

## Project Overview

As such, this project integrates natural language processing and machine learning into an image generation model to develop a personalized fashion recommendation system. It analyzes comments by users on their social media platforms, precisely Twitter, to make a prediction of the MBTI personality types and thus present them with personalized fashion recommendations. The outcome of this is a web application using Flask where users input text and get results on MBTI type predictions, brand recommendations, color suggestions, sentiment analysis, and keyword extraction. Users can also generate fashion images powered by AI, using this information.

## Project Structure

### 1. `01_Twitter_Comment_Scrape.ipynb`
This notebook is responsible for scraping comments from Twitter. Using the Twitter API, it collects user comments from multiple fashion brands, which serve as the foundation for data analysis and model training.

### 2. `02_MBTI_Model_Training.ipynb`
In this notebook, the scraped text data is processed and used to train various machine learning models for predicting MBTI types. The best-performing model is selected for use in later stages of the project.

### 3. `03_MBTI_Prediction_and_Visualization.ipynb`
This notebook applies the best-trained MBTI model to the scraped dataset to predict the MBTI type of each commenter. It also includes visualizations such as MBTI type distributions, word clouds, and BERT-based semantic analysis.

### 4. `04_network_sentiment_lda_and_brand_analysis.ipynb`
This notebook performs further analysis by categorizing data by brand. It includes social network analysis, sentiment analysis, and LDA topic modeling to analyze user interactions and emotions regarding different brands.

### 5. `05_mbti_brand_recommendation.ipynb`
This notebook establishes the mapping between MBTI types and fashion brands. It integrates the predicted MBTI type from the text with brand recommendations, sentiment scores, and keyword extraction. These outputs set the stage for the final web application.

### 6. `06_stable_diffusion`
This folder contains the components for generating fashion images based on user input keywords using the Stable Diffusion model.

### 7. `my_flask_app`
This directory contains the Flask web application that integrates all system functionalities. Users can input any text on the homepage to receive:
- MBTI type prediction
- Brand recommendations
- Color suggestions
- Sentiment analysis
- Keyword extraction
- Network analysis and sentiment LDA visualization

Users can then input specific keywords to generate AI-powered fashion images tailored to their preferences.

### 8. `data`
This folder contains datasets used throughout the project. 
  
### 9. `lib`
This folder is used for any additional libraries or custom modules that the project may use during execution.

### Additional Files
- **`label_encoder.pkl`**: A saved model that encodes and decodes MBTI labels.
- **`mbti_linear_svc_model.pkl`**: The saved MBTI prediction model (Linear SVC).
- **`tfidf_vectorizer.pkl`**: The saved TF-IDF vectorizer used for text feature extraction.
- **`mbti_brand_network_white.html`** & **`mbti_brand_sentiment_network.html`**: HTML files that visualize the network and sentiment analyses related to the brand and MBTI mapping.

## Dependencies
- Flask
- Scikit-learn
- Transformers (BERT)
- Stable Diffusion
- Pandas
- Numpy
- TextBlob
- Matplotlib
- Joblib

## Usage Instructions
1. Run the Flask application inside the `my_flask_app` folder: app.py
2. Open your browser and visit http://127.0.0.1:5000/.
3. Enter any text on the homepage to receive an MBTI prediction, brand recommendations, color suggestions, sentiment analysis, and keyword extraction.
4. Optionally, enter specific keywords to generate personalized fashion images using AI.
