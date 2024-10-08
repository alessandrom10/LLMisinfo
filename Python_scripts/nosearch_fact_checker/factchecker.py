# Description: This script is the main script for the HiSS (Hierarchical Step-by-Step prompting style) fact checker. It uses the Hugging Face API to load the model and make it assess the claim. 
# The user's input is a claim, a date and an author. The model's output is the final assessment of the claim. The search results are also printed.
# Up to now, supported languages are English, Italian and Spanish.

#from Python_scripts.Search_scripts.sel_search import *
from huggingface_hub import InferenceClient
import os
import re
import time
import yaml

def load_config(filename = 'my_config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load the configuration variables
my_config = load_config("my_config.yaml")
model_name = my_config['model_name']
model_id = my_config['model_id']
temperature = my_config['temperature']
max_tokens = my_config['max_tokens']
language = my_config['language']
kshot_path_en = my_config['kshot_path_no_search_en']
kshot_path_it = my_config['kshot_path_no_search_it']
kshot_path_es = my_config['kshot_path_no_search_es']

if language == "en":
    print("English language selected.")
    kshots = load_config(kshot_path_en)["start_messages"]
elif language == "it":
    print("Lingua italiana selezionata.")
    kshots = load_config(kshot_path_it)["start_messages"]
elif language == "es":
    print("Lengua española seleccionada.")
    kshots = load_config(kshot_path_es)["start_messages"]

# Load the Hugging Face API token from the environment variables
try:
    API_TOKEN = os.getenv("HF_TOKEN")
except:
    print("Please set the HF_TOKEN environment variable to your Hugging Face API token.")
    exit(1)

# Load the model from the Hugging Face Hub
print("Loading model " + model_id + "...\n")
client = InferenceClient(model = model_id, token = API_TOKEN)   
    
def chat_completion(messages):
    """
    Defines the chat completion such that it waits in case the api is busy or overloaded.
    """
    for i in range(100):
        try:
            response = client.chat_completion(messages = messages, max_tokens = 1000, temperature = temperature)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error is: \n {e}\n End of error")
            print("API is busy or overloaded. Waiting and retrying...")
            time.sleep(5*60)


def generate_output(user_input):
    """
    Uses the huggingface API to load the model and make it assess the claim.
    
    Args:
        user_input: A dictionary containing the user's input, with the keys 'claim', 'date' and 'author'.
    Returns:
        The model's final assessment of the claim as a string. The search results are also printed.
    """
    messages = list(kshots) #  We set it to the few shot shamples
    print("<" + messages[0]['role'] + "> " + messages[0]['content']) # We check if the file has been correctly loaded
    print("<user and assistant> Loaded few-shot examples.")
    formatted_claim = "Claim: " + user_input["claim"] + ". " # We create a string with "Claim: <claim>"
    if user_input["author"] != "": # If an author is provided we add an "Author: <author>" part to the formatted claim
        if language == "en":
            formatted_claim += "Made by " + user_input["author"] + "."
        elif language == "it":
            formatted_claim += "Di " + user_input["author"] + "."
        elif language == "es":
            formatted_claim += "Hecho por " + user_input["author"] + "."
    if user_input["date"] != "": # Same with the date
        if language == "en":
            formatted_claim += " Date: " + user_input["date"] + "."
        elif language == "it":
            formatted_claim += " Data: " + user_input["date"] + "."
        elif language == "es":
            formatted_claim += " Fecha: " + user_input["date"] + "."
    messages.append({"role": "user", "content": formatted_claim}) # We add the fact that user has said this formatted claim to the messages array of dictionaries
    print("<user> " + formatted_claim) # We print it to show it's working 
    # Get the answer from the llm, it should stop when it thinks it's the user that needs to add information
    response = chat_completion(messages = messages)
    print("<assistant> " + response)
    return response

def main():
    """
    Main function for testing the script on a single claim inputed from the terminal.
    """
    print("Welcome to the Fact Checker! The language model " + model_name + " will verify your claim with the help of google search results.")

    # Get input from the user
    claim = input("Enter the claim to fact-check: ")
    date = input("Enter the date of the claim (optional): ")
    author = input("Enter the author of the claim (optional): ")
    user_input = {"claim": claim, "date": date, "author": author, "domain": ""}

    # Generate output from the model
    output = generate_output(user_input)

    # Print the model's output
    #print("Model Output:\n" + str(output))

if __name__ == "__main__":
    main()
