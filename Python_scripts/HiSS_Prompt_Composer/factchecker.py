# Description: This script is the main script for the HiSS (Hierarchical Step-by-Step prompting style) fact checker. It uses the Hugging Face API to load the model and make it assess the claim. 
# The user's input is a claim, a date and an author. The model's output is the final assessment of the claim. The search results are also printed.
# Up to now, supported languages are English and Italian. (Spanish TODO)
import sys
sys.path.insert(0, './Python_scripts/search_scripts')
from Search_scripts.sel_search_v2 import *
import yaml
from huggingface_hub import InferenceClient
import os
import re

def extract_question(llm_output: str, language:str) -> str:
    """
    The llm is expected to output the question in the format "Question: <question>". This function extracts the question from the llm output.
    Args:
        llm_output: The output of the llm.
    Returns:
        The extracted question is returned as a string. If no question is found, an empty string is returned.
    """
    # define a pattern that matches strings that includes "Question:"
    llm_output = llm_output.lower()
    if language == "en":
        pattern = r"question: (.+)"
    elif language == "it":
        pattern = r"domanda: (.+)"
    # search for the pattern in the LLM output
    match = re.search(pattern, llm_output)
    if match:
        # extract the matched string
        question = match.group(1)
        print("Question posed by the LLM: " + question)
        return question
    # if no match is found, return an empty string
    return ""

def extract_yes_no(llm_output: str, language: str) -> str:
    """
    The llm is expected to output the confidence it has in answering the question in the format "YES" or "NO". This function extracts the confidence from the llm output.
    Args:
        llm_output: The output of the llm.
    Returns:
        The extracted confidence is returned as a string. If no confidence is found, an empty string is returned.
    """
    # uppercase the output to make it case-insensitive
    llm_output = llm_output.upper()
    # define a pattern that matches strings that include "YES" or "NO"
    if language == "en":
        pattern = r"(YES|NO)"
    elif language == "it":
        pattern = r"(S[ÌI]|NO)"
    # search for the pattern in the LLM output
    match = re.search(pattern, llm_output)
    if match:
        # extract the matched string
        confidence = match.group(1)
        if confidence == "SÌ" or confidence == "SI":
            confidence = "YES"
        return confidence
    # if no match is found, return an empty string
    return ""
    
def load_config(filename='my_config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load the configuration variables
config = load_config()
max_searches = config['max_searches']
model_name = config['model_name']
model_id = config['model_id']
tools = config['tools']
temperature = config['temperature']
max_tokens = config['max_tokens']
language = config['language']

if language == "en":
    confident_message = {"role": "user", "content": "Tell me if you are confident to answer the question or not. Answer with 'yes' or 'no':"}
    hiss_config = load_config("Prompts/hiss_kshot.yaml")
elif language == "it":
    confident_message = {"role": "user", "content": "Dimmi se sei sicuro di poter rispondere alla domanda o no. Rispondi con 'sì' o 'no':"}
    hiss_config = load_config("Prompts/hiss_kshot_it.yaml")

start_messages = hiss_config["hiss_messages"]

# Load the Hugging Face API token from the environment variables
try:
    API_TOKEN = os.getenv("HF_TOKEN")
    print(API_TOKEN)
except:
    print("Please set the HF_TOKEN environment variable to your Hugging Face API token.")
    exit(1)

# Load the model from the Hugging Face Hub
print("Loading model "+model_id+"...\n")
client = InferenceClient(model=model_id, token=API_TOKEN)

def generate_output(user_input):
    """
    Uses the huggingface API to load the model and make it assess the claim.
    
    Args:
        user_input: A dictionary containing the user's input, with the keys 'claim' and 'date'.
    Returns:
        The model's final assessment of the claim as a string. The search results are also printed.
    """
    messages = start_messages
    print("<"+messages[0]['role']+"> "+messages[0]['content'])
    print("<user and assistant> Loaded few-shot examples.")
    formatted_claim = "Claim: "+user_input["claim"]+". " 
    if user_input["author"]!="":
        if language == "en":
            formatted_claim += "Made by "+user_input["author"]+"."
        elif language == "it":
            formatted_claim += "Di "+user_input["author"]+"."
    if user_input["date"]!="":
        if language == "en":
            formatted_claim += " Date: "+user_input["date"]+"."
        elif language == "it":
            formatted_claim += " Data: "+user_input["date"]+"."
    messages.append({"role": "user", "content": formatted_claim})
    print("<user> "+formatted_claim) 
    response = client.chat_completion(messages=messages, max_tokens=1000, temperature=temperature).choices[0].message.content
    print("<assistant> "+response)
    
    question = extract_question(response)
    while question != "":
        messages.append({"role": "assistant", "content": response})
        messages.append(confident_message)
        print("<user> "+confident_message["content"])
        response = client.chat_completion(messages=messages, max_tokens=1000, temperature=temperature).choices[0].message.content
        print("<assistant> "+response)
        confidence = extract_yes_no(response)
        if confidence != "":
            if confidence == "YES": # If the model is confident, it will answer the question so
                messages.append({"role": "assistant", "content": response}) 
            elif confidence == "NO":
                search_results = google_search(question, date=user_input["date"])
                #search_results = "This is an example"
                if language == "en":
                    messages.append({"role": "user", "content": "Answer: "+search_results})
                elif language == "it":
                    messages.append({"role": "user", "content": "Risposta: "+search_results})
                print("<user> "+messages[-1]["content"])
                response = client.chat_completion(messages=messages, max_tokens=1000, temperature=temperature).choices[0].message.content
                print("<assistant> "+response)
        else:
            print("The model did not tell whether it is confident or not on answering the question.")
            break
        question = extract_question(response)
    return response


def main():
    """
    Main function for testing the script on a single claim inputed from the terminal.
    """
    print("Welcome to the Fact Checker! The language model "+model_name+" will verify your claim with the help of google search results.")
    # Get input from the user
    claim = input("Enter the claim to fact-check: ")
    date = input("Enter the date of the claim (optional): ")
    author = input("Enter the author of the claim (optional): ")
    user_input = {"claim": claim, "date": date, "author": author}
    # Generate output from the model
    output = generate_output(user_input)

if __name__ == "__main__":
    main()