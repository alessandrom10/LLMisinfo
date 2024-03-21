import requests
import json

# list query 
query_string = {
	"action": "query",
	"format": "json",
	"list": "search",
	"formatversion": "2",
	"srsearch": "Photosyntesis"
}

response = requests.get("https://en.wikipedia.org/w/api.php", params=query_string)


# if the search query is not found, the API will suggest a new search query
new_search = response.json()['query']["searchinfo"]['suggestion']

query_string["srsearch"] = new_search
del query_string["list"]


query_string["prop"] = "description"

response = requests.get("https://en.wikipedia.org/w/api.php", params=query_string)

print(response.json()['query']["search"][0]["snippet"])



