# Description: This script is the main script for the standard fact checker. It uses the Hugging Face API to load the model and make it assess the claim. 
# The user's input is a claim, a date and an author. The model's output is the final assessment of the claim. The search results are also printed.
# Up to now, supported languages are English and Italian. (Spanish TODO)
# importing sys
import sys
sys.path.insert(0, './Python_scripts/search_scripts')
from sel_search_v2 import *
import yaml
from huggingface_hub import InferenceClient
import os
import re
import ast

def extract_query(llm_output: str) -> str:
    """
    Extracts the query from the LLM output if present. (the llm is expected to output the query in the format "Query: <query>")
    Args:
        llm_output: The output of the LLM.
    Returns:
        The extracted query is returned as a string. If no query is found, an empty string is returned.
    """
    # define a pattern that matches strings that starts with "Query: <query>"  and ends with a period
    pattern = r"[Qq]uery: *['\"]*(.*)['\"]*"
    # search for the pattern in the LLM output
    match = re.search(pattern, llm_output)
    if match:
        # extract <query> from the matched string
        query = match.group(1)
        return query
    # if no match is found, return an empty string
    return ""

def parse_kshot(filename = 'standard_kshot.txt'):
    """
    Parses the k-shot examples from the file and returns them as a list of dictionaries.
    Args:
        filename: The name of the file containing the k-shot examples.
    Returns:
        A list of dictionaries, representing messages that will be inputed to the llm. Each dictionary contains the keys 'role' and 'content'.
    """
    messages = []
    snippet = ""
    user = False
    assistant = False
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            #print(line)
            if line.startswith("<user>"):
                if assistant:
                    messages.append({"role":"assistant", "content":snippet})
                    snippet = ""
                user = True
                assistant = False
            elif line.startswith("<assistant>"):
                if user:
                    messages.append({"role":"user", "content":snippet})
                    snippet = ""
                user = False
                assistant = True
            elif line=="\n":
                if user:
                    messages.append({"role":"user", "content":snippet})
                elif assistant:
                    messages.append({"role":"assistant", "content":snippet})
                snippet = ""
                user = False
                assistant = False
            else:
                snippet += line
    #for m in messages:
    #    print(m)
    return messages
    
def load_config(filename='my_config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load the configuration variables from the config file
config = load_config()
max_searches = config['max_searches']
model_name = config['model_name']
model_id = config['model_id']
tools = config['tools']
temperature = config['temperature']
max_tokens = config['max_tokens']
language = config['language']

# Set the standard system prompt based on the language and the path to the corresponding k-shot examples
if language == "en":
    kshot_examples_path = config['kshot_examples_path']
    standard_system_prompt = config['standard_system_prompt_1']
    if max_searches == 1:
        standard_system_prompt += "you must make only one google search. "
    else:
        standard_system_prompt += "you can make up to "+str(max_searches)+" google searches. "
    standard_system_prompt += config['standard_system_prompt_2']
elif language == "it":
    kshot_examples_path = config['italian_kshot_examples_path']
    standard_system_prompt = config['italian_system_prompt_1']
    if max_searches == 1:
        standard_system_prompt += "puoi fare solo una ricerca su google. "
    else:
        standard_system_prompt += "puoi fare fino a "+str(max_searches)+" ricerche su google. "
    standard_system_prompt += config['italian_system_prompt_2']

# Load the Hugging Face API token from the environment variables
try:
    API_TOKEN = os.getenv("HF_TOKEN")
except:
    print("Please set the HF_TOKEN environment variable to your Hugging Face API token.")
    exit(1)

# Load the model from the Hugging Face Hub
print("Loading model "+model_name+"...\n")
client = InferenceClient(model=model_id, token=API_TOKEN)

def generate_output(user_input):
    """
    Uses the huggingface API to load the model and make it assess the claim.
    
    Args:
        user_input: A dictionary containing the user's input, with the keys 'claim', 'author' and 'date'.
                    'author' and 'date' are optional, and can be empty strings.
    Returns:
        The model's final assessment of the claim as a string. The search results are also printed.
    """
    messages = [{"role": "system", "content": standard_system_prompt}]
    print("<system> "+standard_system_prompt)
    messages += parse_kshot(kshot_examples_path)
    print("<user and assistant> K-shot examples loaded.")
    formatted_claim = "Claim: "+user_input["claim"]+". " 
    if user_input["author"]!="":
        if language == "en":
            formatted_claim += "Made by "+user_input["author"]+". "
        elif language == "it":
            formatted_claim += "Di "+user_input["author"]+". "
    if user_input["date"]!="":
        if language == "en":
            formatted_claim += "Date: "+user_input["date"]+". "
        elif language == "it":
            formatted_claim += "Data: "+user_input["date"]+". "
    messages.append({"role": "user", "content": formatted_claim})  
    print("<user> "+formatted_claim)    
    num_searches = 0
    response = client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature).choices[0].message.content
    print("<assistant> "+response)
    
    query = extract_query(response)
    while query!="": 
        num_searches += 1
        search_results = google_search(query, date=user_input["date"])
        search_results = "Search number " + str(num_searches) + ":\n" + search_results
        messages.append({"role": "user", "content": search_results})
        print("<user> "+search_results)
        response = client.chat_completion(messages=messages,max_tokens=max_tokens, temperature=temperature).choices[0].message.content
        print("<assistant> "+response)
        query = extract_query(response)
    return response


def main():
    """
    Main function of the script that runs the fact checker for a single claim inputed from the terminal.
    """
    print("Welcome to the Fact Checker! The language model "+model_name+" will verify your claim with the help of google search results.")
    # Get input from the user
    claim = input("Enter the claim to fact-check: ")
    date = input("Enter the date of the claim (optional): ")
    author = input("Enter the author of the claim (optional): ")
    user_input = {"claim": claim, "date": date, "author": author}
    # Generate output from the model
    output = generate_output(user_input)
    # Print the model's output
    #print("Model Output:\n" + str(output))

if __name__ == "__main__":
    main()