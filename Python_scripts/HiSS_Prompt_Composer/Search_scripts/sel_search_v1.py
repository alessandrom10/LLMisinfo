# This script performs research over the internet, mainly for HiSS prompting
# I don't fully know what parts of it works

import os
import re
import numpy as np
import pandas as pd
import psutil
import requests
import yaml
import spacy
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from urllib.parse import urlparse

# Load the configuration variables from the configuration file
def load_config(filename='my_config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
driver_path = config['chromedriver_path']
chromium_path = config['chromium_path']
max_results = config['max_results']
max_sentences = config['max_sentences']
language = config['language']
url_blacklist = config['url_blacklist']
tag_blacklist = config['tag_blacklist']
type_blacklist = config['type_blacklist']
windowed = config['windowed']
window_size = config['window_size']

# Load the spacy model corresponding to our language - this is used for sentence tokenization
if language == "en":
    nlp = spacy.load("en_core_web_sm")
elif language == "it":
    nlp = spacy.load("it_core_news_sm") 
elif language == "es":
    nlp = spacy.load("es_core_news_sm")
else:
    raise Exception("ERROR: LANGUAGE NOT CORRECT - should be set as either 'en', 'it' or 'es'")

# Load the transformer model for sentence similarity
similarity_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")   

'''def google_search(query: str) -> str:
    """
    Get the google search results given a query
    
    Args:
        query: the query that will be used to perform the search
    Returns:
        A sting containing the list of website given by the search results along with a snippet of their content.
    """
    return "Google: "+query+"\n"+\
            "1. nytimes.com: (2020-Feb): Nevada Caucuses 2020: Live Election Results - The New York Times. Feb 22, 2020 ...\n"+\
            "2. nbcnews.com: Sanders wins Nevada Democratic caucuses with wave of young and ....\n"+\
            "3. nytimes.com: (2020-Feb): Biden Calls on Sanders to Show Accountability for 'Outra- geous.. Feb 16, 2020\n"+\
            "4. vox.com: (2020-Feb): Nevada caucus results: 3 winners and 2 losers - Vox. Feb 22, 2020\n"+\
            "5. washingtonpost.com: 2020 Nevada Caucuses Live Election Results | The Washing- ton Post. Feb 24, 2020 ... 2020 Nevada Democratic presidential caucuses; Bernie Sanders, 6,788, 46.8; Joe Biden, 2,927, 20.2; Pete Buttigieg, 2,073 ....\n"+\
            "6. theintercept.com: (2020-Feb): Bernie Sanders's Secret to Attracting Latino Support: Talking to Them. Feb 20, 2020\n"+\
            "7. pbs.org: (2020-Feb): Bloomberg qualifies for debate, Sanders leads ahead of Nevada 8. wikipedia.org: 2020 Nevada Democratic presidential caucuses - Wikipedia.\n"+\
            "9. politico.com: Iowa Election Results 2020 | Live Map Updates | Voting by County\n"+\
            "10. tufts.edu: (2020-Feb): Exclusive Analysis: In Nevada, young people once again force\n"'''

def google_search(query: str, date: str = "") -> str:
    """
    Get the google search results given a query and a date (date is optional)
    
    Args:
        query: the query that will be used to perform the search
        date: results after this date will be excluded (optional)
    Returns:
        formatted_results: A string containing the list of websites given by the search results along with a snippet of their content.
    """
    formatted_results = "Google: " + query + " \n"
    if date != "":
        date = pd.to_datetime(date).date()
        query += " before:" + str(date)
        #print("Query with date:", query)

    search_results = get_search_results(query)
    n_good_results = 0
    for result in search_results:
        print("GETTING page:", result)
        if result!="NULL":
            try:
                text = get_all_text(result)
                top_s, _, _ = extract_relevant_sentences(text, query)#, top_n=5, windowed=windowed, window_size=window_size)
                n_good_results += 1
                formatted_results += str(n_good_results) + ". " + get_domain(result) +": " + " \n ".join(top_s) +" \n"
                #if we have reached the maximum number of results, we stop
                if n_good_results==max_results:
                    break
            except Exception as e:
                print("Failed to fetch page:", result)
                print(e)
                continue
    return formatted_results
        
def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def get_domain(url: str) -> str:
    """
    Extracts the domain name from a given URL.
    
    Args:
        url (str): The URL from which to extract the domain.
    
    Returns:
        str: The domain name.
    """
    # Parse the URL
    parsed_url = urlparse(url)
    # Extract the netloc (network location part)
    domain = parsed_url.netloc
    # Remove 'www.' if present
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

def filter_urls(urls):
    """
    Filters a list of URLs based on validity and allowed domains.

    Args:
        urls : A list of URLs to be filtered.

    Returns:
        list: A filtered list of URLs that are valid and belong to allowed domains.
    """

    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and not url.endswith(".pdf")
        except ValueError:
            return False
        
    def is_allowed_domain(url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        return not any(excluded in domain for excluded in url_blacklist)
    
    def is_allowed_type(url):
        return not any(url.endswith(excluded) for excluded in type_blacklist)
    
    filtered_urls = [
        url for url in urls
        if is_valid_url(url) and is_allowed_domain(url) and is_allowed_type(url)
    ]
    
    # Remove duplicate results from the same domain
    filtered_domains = [get_domain(url) for url in filtered_urls]
    filtered_urls = [
        filtered_urls[i] for i in range(len(filtered_urls))
        if get_domain(filtered_urls[i]) not in filtered_domains[:i]
    ]
    return filtered_urls

def save_soup_to_file(soup, filename):
    """
    Save a BeautifulSoup object to a file.

    Args:
        soup : The BeautifulSoup object to save.
        filename : The name of the file to save the BeautifulSoup object to.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(str(soup))

def get_all_text(url):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page, status code: {response.status_code}")
    # Check the content type to determine if it's a webpage or a downloadable file
    content_type = response.headers.get('Content-Type')
    if 'text/html' not in content_type:
        raise Exception(f"URL does not point to an HTML page, content type: {content_type}")

    
    soup = BeautifulSoup(response.text, 'html.parser')
    for element in soup(tag_blacklist):
        element.decompose()

    text = soup.get_text(separator=' ')
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def get_search_results(query, keep_browser_open=False):
    """
    Get the search results for a given query. Uses Selenium to scrape the search results from Google.
    
    Args:
        query: The query to search for.
        keep_browser_open: Whether to keep the browser open after scraping the search results.
    
    Returns:
        list: The URLs of the search results.
    """
    chrome_options = Options()
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-search-engine-choice-screen")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--silent")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    if os.name == 'nt':
        os.environ['WDM_LOG_LEVEL'] = '0'
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    else:
        chrome_options.add_argument("--log-path=/dev/null")

    service = Service(driver_path)
    
    driver = webdriver.Chrome(service = service, options = chrome_options)
    
    try:
        driver.get('https://www.google.com')
        
        wait = WebDriverWait(driver, 10)
        try:
            #accept_cookies_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[text()="Accetta tutto"]')))
            accept_cookies_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[text()="Accept all"]')))
            
            accept_cookies_button.click()
        except Exception as e:
            print("Cookie consent dialog not found or already accepted.")
        search_box = wait.until(EC.element_to_be_clickable((By.NAME, 'q')))
        
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.yuRUbf')))
        
        results = driver.find_elements(By.CSS_SELECTOR, "div.g div.yuRUbf a")
        #print("Search results found:", len(results))
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
            driver.close()
            driver.quit()
            kill_process_and_children(service.process.pid)

def get_embeddings(sentences, model):
    return [model(sentence)[0][0] for sentence in sentences]

def remove_questions(sentences):
    return [sentence for sentence in sentences if "?" not in sentence]

def cosine_similarity_score(query_embedding, sentence_embeddings):
    """
    Calculate the cosine similarity between a query embedding and a list of sentence embeddings.
    
    Args:
        query_embedding: The query embedding.
        sentence_embeddings: The list of sentence embeddings.
    
    Returns:
        list: The cosine similarity scores between the query and the sentences.
    """
    query_embedding = np.array(query_embedding).reshape(1, -1)
    sentence_embeddings = np.array(sentence_embeddings)
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    return similarities[0]

def extract_relevant_sentences(content, query):#, top_n=5, windowed=False, window_size=5):
    """
    Extract the most relevant sentences from a given text based on a query.
    
    Args:
        content: The text from which to extract the sentences.
        query: The query to use for extracting the sentences.
    
    Returns:
        list: The top sentences extracted from the text.
        list: The indices of the top sentences in the original text.
        list: The cosine similarity scores of the top sentences
    """
    doc = nlp(content)
    sentences = [sent.text for sent in doc.sents]
    sentences = remove_questions(sentences)

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
    if not filtered_sentences:
        return [], [], []
    
    similarities = cosine_similarity_score(query_embedding, sentence_embeddings)
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_sentences = [filtered_sentences[idx] for idx in ranked_indices]
    ranked_similarities = [similarities[idx] for idx in ranked_indices]

    top_indices = ranked_indices[:max_sentences]
    top_similarities = ranked_similarities[:max_sentences]
    if windowed:
        top_sentences = []
        for i in top_indices:
            start = max(0, i-(window_size//2))
            end = min(len(ranked_sentences), i+(window_size//2)+1)
            top_sentences.append(" ".join(filtered_sentences[start:end]))
    else:
        top_sentences = ranked_sentences[:max_sentences]
        
    return top_sentences, top_indices, top_similarities

#google_search("Bernie Sanders wins Nevada Democratic caucuses", date="2024-03-03")