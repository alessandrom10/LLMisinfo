import requests

def search_duckduckgo(query):
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    response = requests.get(url)
    data = response.json()
    return data

query = "photosynthesis"
result = search_duckduckgo(query)

# Print the JSON structure, just main fields
print(result.keys())

# Extract and print the main information
main_info = result['Abstract']
print(main_info)

# print also the abstract text and show the image
abstract_text = result['AbstractText']
print(abstract_text)

# show the image
# image_url = result['Image']
# print(image_url)