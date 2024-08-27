import requests
from bs4 import BeautifulSoup

def fetch_search_results(query, num_results=3):
    # Replace 'YOUR_SEARCH_ENGINE_URL' with the actual search engine URL and query parameter
    search_url = f'https://www.google.com/search?q={query}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the main search results. This depends on the structure of the search engine's results page.
        # For Google, the results are usually within 'div' tags with class 'g'
        search_results = soup.find_all('div', limit=num_results)
        
        for i, result in enumerate(search_results, start=1):
            title = result.find('h3')
            if title:
                title = title.text
            else:
                title = "No title found"
            
            snippet = result.find('div', class_='IsZvec')
            if snippet:
                snippet = snippet.text
            else:
                snippet = "No snippet found"
            
            print(f"Result {i}:")
            print(f"Title: {title}")
            print(f"Snippet: {snippet}\n")
    else:
        print("Failed to retrieve search results")

# Example usage
fetch_search_results('photosynthesis')
