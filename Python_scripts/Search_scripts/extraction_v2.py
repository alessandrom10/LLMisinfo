import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sel_search import *

# Download necessary NLTK data
nltk.download('punkt')

def scrape_website(url): # Function to scrape the website content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def preprocess_text(text): # Function to preprocess the text into sentences
    sentences = sent_tokenize(text)
    return [sentence.strip() for sentence in sentences if len(sentence.split()) > 5]

def get_embeddings(texts, model, tokenizer): # Function to get embeddings for the text with the fine-tuned model
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def calculate_relevance_score(query_embedding, sentence_embedding, query_tokens, sentence_tokens):
    cos_sim = cosine_similarity(query_embedding, sentence_embedding)[0][0]
    
    # Calculate token overlap
    query_set = set(query_tokens)
    sentence_set = set(sentence_tokens)
    overlap = len(query_set.intersection(sentence_set)) / len(query_set)
    
    # Combine cosine similarity and token overlap
    relevance_score = 0.7 * cos_sim + 0.3 * overlap
    return relevance_score

def extract_relevant_sentences(query, url, top_k=3):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Scrape and preprocess the website content
    content = scrape_website(url)
    sentences = preprocess_text(content)

    # Get embeddings for query and sentences
    query_embedding = get_embeddings([query], model, tokenizer)
    sentence_embeddings = get_embeddings(sentences, model, tokenizer)

    # Calculate relevance scores
    query_tokens = query.lower().split()
    relevance_scores = []
    for i, sentence in enumerate(sentences):
        sentence_tokens = sentence.lower().split()
        score = calculate_relevance_score(
            query_embedding, 
            sentence_embeddings[i].unsqueeze(0), 
            query_tokens, 
            sentence_tokens
        )
        relevance_scores.append(score)

    # Get top-k relevant sentences
    top_indices = np.argsort(relevance_scores)[-top_k:][::-1]
    return [sentences[i] for i in top_indices]

# Example usage
query = "What are the main features of Python?"
url = "https://www.python.org/about/"
relevant_sentences = extract_relevant_sentences(query, url)

print(f"Query: {query}")
print("Relevant sentences:")
for i, sentence in enumerate(relevant_sentences, 1):
    print(f"{i}. {sentence}")