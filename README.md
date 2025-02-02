# Text-classifciation
Sustainable Development Goals (SDGs) are a global framework designed to address critical challenges such as poverty, education, and climate change. This project focuses on developing a multi-label text classification model to automatically identify SDGs relevant to textual content. The
system combines keyword-based and machine learning approaches to achieve robust and interpretable predictions.

Heres the below code 

import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import hamming_loss, f1_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# Load the dataset
data = pd.read_excel("sdg_data.xlsx")

# Initialize required tools
lem = WordNetLemmatizer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')

# Define SDG keywords
sdg_keywords = {
    'SDG 4': ['education', 'literacy', 'learning', 'schooling', 'teacher', 'curriculum', 'school'],
    'SDG 2': ['hunger', 'malnutrition', 'food security', 'agriculture', 'nutrition', 'farming',
              'crop', 'livestock', 'sustainable farming', 'zero hunger'],
    'SDG 3': ['health', 'well-being', 'pollution', 'disease', 'hospitals', 'medical', 'access to healthcare',
              'vaccination', 'pandemic', 'mental health', 'maternal health', 'child mortality'],
    'SDG 13': ['climate change', 'global warming', 'carbon emissions', 'climate action',
               'environmental protection', 'sustainability', 'renewable energy', 'greenhouse gases']
}

# Clean and preprocess text
def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)
    tokens = row.split()
    cleaned = [lem.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# Map SDGs to binary features
def sdg_features(text):
    features = {key: 0 for key in sdg_keywords}
    words = set(text.split())
    for sdg, keywords in sdg_keywords.items():
        if any(keyword in words for keyword in keywords):
            features[sdg] = 1
    return list(features.values())

# Apply text cleaning
data['Data'] = data['Data'].apply(lambda x: clean_row(x) if pd.notnull(x) else '')

# Handle multiple SDG labels in the dataset
data['SDG'] = data['SDG'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['SDG'])

# Transform text data using CountVectorizer
Vector = CountVectorizer(max_features=1000, lowercase=False, ngram_range=(1, 2))
x = data['Data']
vec_data = Vector.fit_transform(x)
dictionary_features = np.array([sdg_features(text) for text in x])

# Combine and normalize features using MinMaxScaler
# Weight SDG dictionary features more heavily
weighted_dictionary_features = dictionary_features * 2  # Adjust weight as needed
combined_features = np.hstack([vec_data.toarray(), weighted_dictionary_features])
scaler = MinMaxScaler()
combined_features = scaler.fit_transform(combined_features)

# Split data into training and testing sets
train_data, test_data, train_label, test_label = train_test_split(
    combined_features, y, test_size=0.2, random_state=0
)

# Train a model with probability thresholding
clf = OneVsRestClassifier(MultinomialNB())
clf.fit(train_data, train_label)

# Predict probabilities
y_pred_prob = clf.predict_proba(test_data)

# Define a threshold for multilabel classification
threshold = 0.5
y_pred = (y_pred_prob >= threshold).astype(int)

# Evaluate the model using multilabel metrics
print("Hamming Loss:", hamming_loss(test_label, y_pred))
print("Micro F1 Score:", f1_score(test_label, y_pred, average='micro'))
print("Macro F1 Score:", f1_score(test_label, y_pred, average='macro'))
print("Average Precision:", average_precision_score(test_label, y_pred, average='micro'))

# Function to predict and identify multiple SDGs for a new input
def clean_and_predict(text, threshold=0.5):
    # Clean the input text
    cleaned_text = clean_row(text)
    
    # Vectorize the cleaned text
    vec_news = Vector.transform([cleaned_text])
    news_dict_features = np.array(sdg_features(cleaned_text)).reshape(1, -1) * 2  # Apply same weight
    combined_news_features = np.hstack([vec_news.toarray(), news_dict_features])
    combined_news_features = scaler.transform(combined_news_features)
    
    # Predict using the model
    pred_prob = clf.predict_proba(combined_news_features)
    pred = (pred_prob >= threshold).astype(int)
    predicted_labels_model = mlb.inverse_transform(pred)
    
    # Identify SDGs using the dictionary approach
    dictionary_based_sdgs = [
        sdg for sdg, keywords in sdg_keywords.items()
        if any(keyword in cleaned_text.split() for keyword in keywords)
    ]
    
    # Combine model-based and dictionary-based SDGs
    combined_sdgs = set(
        [item for sublist in predicted_labels_model for item in sublist] +
        dictionary_based_sdgs
    )
    return list(combined_sdgs)

# Example usage
txt = 'More than 36 million people are living with HIV globally with hunger'
predicted_sdgs = clean_and_predict(txt)
if isinstance(predicted_sdgs, list) and predicted_sdgs:  # If the result is a non-empty list
    print(f"Identified SDGs: {', '.join(predicted_sdgs)}")
else:
    print("No SDGs are identified for the text.")
