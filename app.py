import requests
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from collections import Counter

# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

# Function to extract webpage text
def get_webpage_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract all text from the page
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    else:
        print("Error fetching the webpage.")
        return None

# Function to extract keywords using spaCy
def extract_keywords_spacy(text):
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return keywords

# Function to extract keywords using TF-IDF
def extract_keywords_tfidf(text, n=5):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform([text])
    feature_names = tfidf.get_feature_names_out()
    sorted_items = tfidf_matrix.toarray().argsort()[0][::-1]
    keywords = [feature_names[i] for i in sorted_items[:n]]
    return keywords

# Function to extract keywords using YAKE
def extract_keywords_yake(text, n=5):
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=n)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def get_combined_keywords(spacy_keywords, tfidf_keywords, yake_keywords, top_n=5):
    # Flatten all keywords into one list
    all_keywords = spacy_keywords + tfidf_keywords + yake_keywords
    # Count the frequency of each keyword
    keyword_counts = Counter(all_keywords)
    # Sort by frequency and select top_n keywords
    most_common_keywords = [keyword for keyword, _ in keyword_counts.most_common(top_n)]
    return most_common_keywords

# Fetch and process the webpage text
url = "https://www.redshelf.com/"
webpage_text = get_webpage_text(url)
if webpage_text:
    # Extract keywords with different methods
    spacy_keywords = extract_keywords_spacy(webpage_text)
    tfidf_keywords = extract_keywords_tfidf(webpage_text)
    yake_keywords = extract_keywords_yake(webpage_text)
 
    top_keywords = get_combined_keywords(spacy_keywords, tfidf_keywords, yake_keywords, top_n=5)

    # Print extracted keywords
    #print("SpaCy Extracted Keywords:", spacy_keywords)
    print("TF-IDF Extracted Keywords:", tfidf_keywords)
    print("YAKE Extracted Keywords:", yake_keywords)
    print("Top 5 Keywords Representing the Website:", top_keywords)