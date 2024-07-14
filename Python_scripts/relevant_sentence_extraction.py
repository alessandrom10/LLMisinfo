from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import psutil
import requests
from bs4 import BeautifulSoup
import spacy
from urllib.parse import urlparse
import re
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import yaml

def load_config(filename='config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
dataset_path = config['dataset_path']
driver_path = config['chromedriver_path']
max_results = config['max_results']
max_sentences = config['max_sentences']
language = config['language']
url_blacklist = config['url_blacklist']
tag_blacklist = config['tag_blacklist']
windowed = config['windowed']
window_size = config['window_size']

def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def filter_urls(urls):
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
        
    def is_allowed_domain(url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        return not any(excluded in domain for excluded in url_blacklist)
    
    filtered_urls = [
        url for url in urls
        if is_valid_url(url) and is_allowed_domain(url)
    ]
    return filtered_urls

def save_soup_to_file(soup, filename):
    # Open the file in write mode
    with open(filename, 'w', encoding='utf-8') as file:
        # Write the string representation of the soup object to the file
        file.write(str(soup))

def get_all_text(url):
    # Fetch the web page content
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page, status code: {response.status_code}")
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    for element in soup(tag_blacklist):
        element.decompose()

    #save_soup_to_file(soup, 'soup.txt')
    text = soup.get_text(separator=' ')
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def google_search(query, driver_path, keep_browser_open=False):
    # Configure Chrome options
    chrome_options = Options()
    #chrome_options.add_argument("--headless")  # Run Chrome in headless mode for faster execution
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-extensions")

    # Set up Chrome service
    service = Service(driver_path)
    
    # Create a new instance of the Chrome driver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Open Google
        driver.get('https://www.google.com')
        
        # Wait for the search box to be present and visible
        wait = WebDriverWait(driver, 10)
        try:
            accept_cookies_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[text()="Accetta tutto"]')))
            accept_cookies_button.click()
        except Exception as e:
            print("Cookie consent dialog not found or already accepted.")
        search_box = wait.until(EC.element_to_be_clickable((By.NAME, 'q')))
        
        # Enter the query into the search box
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        # Wait for the results to load
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.yuRUbf')))
        
        # Find all search result links
        results = driver.find_elements(By.CSS_SELECTOR, "div.g div.yuRUbf a")
        print("Search results found:",len(results))
        search_results = []
        for result in results:
            try:
                search_results.append(result.get_attribute("href"))
            except Exception as e:
                search_results.append("NULL")
        search_results = filter_urls(search_results)
        return search_results

    finally:
        if not keep_browser_open:
            # Close the browser
            driver.close()
            driver.quit()
            kill_process_and_children(service.process.pid)
        
# Function to get sentence embeddings
def get_embeddings(sentences, model):
    return [model(sentence)[0][0] for sentence in sentences]

def remove_questions(sentences):
    return [sentence for sentence in sentences if "?" not in sentence]

# Function to calculate cosine similarity
def cosine_similarity_score(query_embedding, sentence_embeddings):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    sentence_embeddings = np.array(sentence_embeddings)
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    return similarities[0]

def extract_relevant_sentences(content, query, similarity_model, top_n=5, windowed=False, window_size=5):
    # Tokenize document into sentences
    doc = nlp(content)
    sentences = [sent.text for sent in doc.sents]
    sentences = remove_questions(sentences)

    # Get embeddings for document sentences and query
    query_embedding = get_embeddings([query], similarity_model)[0]

    filtered_sentences = []
    sentence_embeddings = []
    for sentence in sentences:
        try:
            embedding = get_embeddings([sentence], similarity_model)[0]
            filtered_sentences.append(sentence)
            sentence_embeddings.append(embedding)
        except Exception as e:
            continue        
    # If no valid sentences remain, return empty results
    if not filtered_sentences:
        return [], [], []
    
    # Calculate similarities
    similarities = cosine_similarity_score(query_embedding, sentence_embeddings)
    # Rank sentences by similarity
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_sentences = [filtered_sentences[idx] for idx in ranked_indices]
    ranked_similarities = [similarities[idx] for idx in ranked_indices]

    top_indices = ranked_indices[:top_n]
    top_similarities = ranked_similarities[:top_n]
    if windowed:
        # for each index in top_indices, get the corresponfing sentence and a window of window_size sentences around it
        top_sentences = []
        for i in top_indices:
            start = max(0, i-(window_size//2))
            end = min(len(ranked_sentences), i+(window_size//2)+1)
            top_sentences.append(" ".join(filtered_sentences[start:end]))
    else:
        top_sentences = ranked_sentences[:top_n]
        
    return top_sentences, top_indices, top_similarities




df = pd.read_csv(dataset_path+"\\english_claim_review.csv")
#authors_url = df["author.url"].unique()
url_blacklist = np.concatenate((df["author.url"].unique(), url_blacklist))

if language == "en":
    nlp = spacy.load("en_core_web_sm")
elif language == "it":
    nlp = spacy.load("it_core_news_sm") 
elif language == "es":
    nlp = spacy.load("es_core_news_sm")
else:
    raise Exception("ERROR: LANGUAGE NOT CORRECT - should be set as either 'en', 'it' or 'es'")

# Load transformer model for sentence similarity
similarity_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")    
# that's the sentence transformer
print(similarity_model)
query = df.loc[0,"claimReviewed"]
date = pd.to_datetime(df.loc[0, 'datePublished']).date()
if date != None:
    query_search = query + " before:" + str(date)
else:
    query_search = query

print("QUERY:",query,"\nDATE:",date)
results = google_search(query_search, driver_path, keep_browser_open=False)
print(results,"\n\n")

n_good_results = 0
for result in results:
    print("GETTING page:", result)
    if result!="NULL":
        try:
            text = get_all_text(result)
            print("----------------------------------------------------------------")
            top_s, top_i, top_simil = extract_relevant_sentences(text, query, similarity_model, top_n=max_sentences, windowed=windowed, window_size=window_size)
            print("\n".join([f"{top_i[i]}, {top_simil[i]}: {top_s[i]}" for i in range(len(top_i))]))
            n_good_results += 1
            if n_good_results==max_results:
                break
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print()
    else:
        print("NULL result\n")
print("NUMBER OF GOOD RETRIEVED RESULTS:", n_good_results)

    