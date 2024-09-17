# Personalized Fashion Recommendation System

## Project Overview

As such, this project integrates natural language processing and machine learning into an image generation model to develop a personalized fashion recommendation system. It analyzes comments by users on their social media platforms, precisely Twitter, to make a prediction of the MBTI personality types and thus present them with personalized fashion recommendations. The outcome of this is a web application using Flask where users input text and get results on MBTI type predictions, brand recommendations, color suggestions, sentiment analysis, and keyword extraction. Users can also generate fashion images powered by AI, using this information.

## Project Structure

### 1. `01_Twitter_Comment_Scrape.ipynb`
This notebook is responsible for scraping comments from Twitter. Using the Twitter API, it collects user comments from multiple fashion brands, which serve as the foundation for data analysis and model training.

### 2. `02_MBTI_Model_Training.ipynb`
In this notebook, the scraped text data is processed and used to train various machine learning models for predicting MBTI types. The best-performing model is selected for use in later stages of the project.

### 3. `03_MBTI_Prediction_and_Visualization.ipynb`
This notebook focuses on analyzing and visualizing MBTI type distributions from two datasets. It calculates the percentage distribution of each MBTI type in both datasets and compares them using bar charts, illustrating the differences in MBTI distributions between the two.

### 4. `04_MBTI_Classification_and_BERT_Embeddings_Analysis.ipynb`
This notebook focuses on applying pre-trained MBTI classification models to predict MBTI types based on text data. It uses both TF-IDF features and BERT embeddings to generate a combined feature set, which is then used to make predictions for each MBTI dimension (EI, NS, FT, PJ). The final MBTI types are saved in a new dataset.

### 5. `05_Network_Sentiment_Lda_Analysis.ipynb`
This notebook performs sentiment analysis and LDA (Latent Dirichlet Allocation) topic modeling on MBTI-related text data, visualizing sentiment polarity distribution by MBTI type and analyzing topic distributions. ​

### 6. `06_MBTI_Brand_Mapping.ipynb`
This notebook predicts MBTI types based on user input and recommends brands associated with those MBTI types, while also performing sentiment analysis and topic extraction from the input text

### 7. `07_stable_diffusion`
This folder contains the components for generating fashion images based on user input keywords using the Stable Diffusion model.

### 8. `my_flask_app`
This directory contains the Flask web application that integrates all system functionalities. Users can input any text on the homepage to receive:
- MBTI type prediction
- Brand recommendations
- Color suggestions
- Sentiment analysis
- Keyword extraction
- Network analysis and sentiment LDA visualization

Users can then input specific keywords to generate AI-powered fashion images tailored to their preferences.

### 9. `data`
This folder contains datasets used throughout the project. 
  
### 10. `lib`
This folder is used for any additional libraries or custom modules that the project may use during execution.

### 11. `saved_models`
This folder is containing four support vector classifier (SVC) models for the MBTI dimensions (EI, FT, NS, PJ) and a TF-IDF vectorizer used for text preprocessing.

### Additional Files
- **`label_encoder.pkl`**: A saved model that encodes and decodes MBTI labels.
- **`mbti_linear_svc_model.pkl`**: The saved MBTI prediction model (Linear SVC).
- **`tfidf_vectorizer.pkl`**: The saved TF-IDF vectorizer used for text feature extraction.
- **`mbti_brand_network.html`** & **`mbti_brand_sentiment_network.html`**: HTML files that visualize the network and sentiment analyses related to the brand and MBTI mapping.
- **`ratings.csv`**: The saved user feedback.

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
2. Open your browser and visit http://127.0.0.1:8000/.
3. Enter any text on the homepage to receive an MBTI prediction, brand recommendations, color suggestions, sentiment analysis, and keyword extraction.
4. Optionally, enter specific keywords to generate personalized fashion images using AI.
