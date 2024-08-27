import requests
from bs4 import BeautifulSoup

def search_arxiv_citation(query):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 3, 
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    response = requests.get(base_url, params=params)
    feed = BeautifulSoup(response.content, features="html.parser")


    for entry in feed.find_all('entry'):
        # Extract necessary metadata
        arxiv_id = entry.id.text.split('/')[-1]
        title = entry.title.text.replace('\n', ' ')
        author_list = [author.find('name').text for author in entry.find_all('author')]
        authors = ' and '.join(author_list)
        published_date = entry.published.text.split('-')[0]

        # Format and print the BibTeX citation
        bibtex = f"""
                @article{{{arxiv_id},
                    title = {{{title}}},
                    author = {{{authors}}},
                    year = {{{published_date}}},
                    eprint = {{{arxiv_id}}},
                    archivePrefix = {{arXiv}},
                    primaryClass = {{cs.CL}}
                }}
                """
        print(bibtex)

def search_arxiv(query):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 3,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    response = requests.get(base_url, params=params)
    feed = BeautifulSoup(response.content, features="html.parser")

    summaries = []
    for entry in feed.find_all('entry'):
        title = entry.title.text
        summary = entry.summary.text
        summaries.append((title, summary))

    return summaries


query = "Effects of creatine supplementation"

summaries = search_arxiv(query)

print(summaries)