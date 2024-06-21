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
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def google_search(query, keep_browser_open=False):
    # Path to the ChromeDriver executable
    driver_path = 'C:\\Users\\flash\\Documents\\UNI\\MAG-ANNOII\\MDP\\chromedriver-win64\\chromedriver.exe'  # Sostituisci con il percorso effettivo

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
        #search_box = driver.find_element(By.NAME, 'q')
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
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[jscontroller="SC7lYd"]')))
        
        # Extract the search results
        results = driver.find_elements(By.CSS_SELECTOR, 'div[jscontroller="SC7lYd"]')
        print("Search results found:",len(results))
        search_results = []
        
        for result in results[:5]:
            #print(result)
            try:
                title = result.find_element(By.TAG_NAME, 'h3').text
                link = result.find_element(By.TAG_NAME, 'a').get_attribute('href')
                snippet = result.find_element(By.CSS_SELECTOR, 'div[class="VwiC3b yXK7lf lVm3ye r025kc hJNv6b Hdw6tb"]').text
                search_results.append({'title': title, 'link': link, 'snippet': snippet})
            except Exception as e:
                # If any element is not found, skip the result
                #continue
                search_results.append({'title': "NULL", 'link': "NULL", 'snippet': "NULL"})
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

# Function to calculate cosine similarity
def cosine_similarity_score(query_embedding, sentence_embeddings):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    sentence_embeddings = np.array(sentence_embeddings)
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    return similarities[0]

# Main processing function
def extract_relevant_sentences(content, query, similarity_model, top_n=5):
    # Tokenize document into sentences
    doc = nlp(content)
    sentences = [sent.text for sent in doc.sents]
    
    # Get embeddings for document sentences and query
    query_embedding = get_embeddings([query], similarity_model)[0]
    sentence_embeddings = get_embeddings(sentences, similarity_model)
    
    # Calculate similarities
    similarities = cosine_similarity_score(query_embedding, sentence_embeddings)
    
    # Rank sentences by similarity
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_sentences = [sentences[idx] for idx in ranked_indices]
    # Extract top N sentences
    top_sentences = ranked_sentences[:top_n]
    top_indices = ranked_indices[:top_n]
    return top_sentences, top_indices

language = "en"
if language == "en":
    nlp = spacy.load("en_core_web_sm")
#elif language == "it":
    #nlp = italian sentence tokenizer 
#else:
    #nlp = spanish sentence tokenizer 

# Load transformer model for sentence similarity
similarity_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")    
query = "Il governo ha cancellato i 330 milioni di euro di supporto per gli affitti"
results = google_search(query, keep_browser_open=False)
doc = nlp(query)
# Extract keywords and entities
#keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
#entities = [(entity.text, entity.label_) for entity in doc.ents]
#print("Keywords:", keywords)
#print("Entities:", entities)
#keywords = ['governo', 'cancellato', '330', 'milioni', 'euro', 'supporto', 'affitti']
for result in results:
    #print(f"Title: {result['title']}\nLink: {result['link']}\n")
    print(f"Title: {result['title']}\nLink: {result['link']}\nSnippet: {result['snippet']}\n")


for result in results:
    #try:
    print("GETTING ", result['link'], "page")
    if result['link']!="NULL":
        page = requests.get(result['link'])
        page_soup = BeautifulSoup(page.text, 'html.parser')
        # Extract main content
        # We should customize this based on the website structure
        paragraphs = page_soup.find_all('p')
        paragraphs += page_soup.find_all('span')
        content = "\n ".join([para.text for para in paragraphs])
        # Check for relevance (e.g., keyword presence)
        print("\n----------------------------------------------------------------\n")
        top_s, top_i = extract_relevant_sentences(content, query, similarity_model)
        print("\n".join([f"{top_i[i]}: {top_s[i]}" for i in range(len(top_i))]))
        #if all(keyword in content for keyword in keywords):
        #    print(f"Relevant content from {result['link']}:\n", content, "\n")
        #else:
        #    print(content)
        #except Exception as e:
        #    print(f"Failed to scrape {result['link']}: {e}")
    
    