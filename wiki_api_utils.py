import wikipedia

def search_wikipedia_short(query):
    search_results = wikipedia.search(query)
    if search_results:
        main_topic = search_results[0]

        try:
            page_summary = wikipedia.page(main_topic).content
        except wikipedia.exceptions.PageError as e:
            page_summary = wikipedia.page(e.options[0]).content

        page_summary = page_summary.split("\n\n")[0]
        return page_summary
    else:
        return None
    

def search_wikipedia(query):
    search_results = wikipedia.search(query)
    if search_results:
        main_topic = search_results[0]
        page_summary = wikipedia.page(main_topic).content
        return page_summary
    else:
        return None
    
def get_topics(query):
    search_results = wikipedia.search(query)
    if search_results:
        return search_results
    else:
        return None