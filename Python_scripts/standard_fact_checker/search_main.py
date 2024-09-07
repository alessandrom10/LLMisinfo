import sel_search

def main():
    query = input("Enter your search query: ")
    date = input("Enter the date (YYYY-MM-DD): ")
    
    # Perform Google search using sel_search module
    results = sel_search.google_search(query, date)

    print(results)
    
if __name__ == "__main__":
    main()