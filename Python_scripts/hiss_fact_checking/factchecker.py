import yaml
from sel_search import *
from huggingface_hub import InferenceClient
import os
import re

def extract_question(llm_output: str) -> str:
    # define a pattern that matches strings that includes "Question:"
    pattern = r"Question: (.+)"
    # search for the pattern in the LLM output
    match = re.search(pattern, llm_output)
    if match:
        # extract the matched string
        question = match.group(1)
        print("Question posed by the LLM: " + question)
        return question
    # if no match is found, return an empty string
    return ""

def extract_yes_no(llm_output: str) -> str:
    # uppercase the output to make it case-insensitive
    llm_output = llm_output.upper()
    # define a pattern that matches strings that include "YES" or "NO"
    pattern = r"(YES|NO)"
    # search for the pattern in the LLM output
    match = re.search(pattern, llm_output)
    if match:
        # extract the matched string
        confidence = match.group(1)
        return confidence
    # if no match is found, return an empty string
    return ""
    
def load_config(filename='config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
max_searches = config['max_searches']
model_name = config['model_name']
model_id = config['model_id']
tools = config['tools']
temperature = config['temperature']
max_tokens = config['max_tokens']

try:
    API_TOKEN = os.getenv("HF_TOKEN")
    print(API_TOKEN)
except:
    print("Please set the HF_TOKEN environment variable to your Hugging Face API token.")
    exit(1)

def load_model_and_generate_output(user_input):
    """
    Uses the huggingface API to load the model and make it assess the claim.
    
    Args:
        user_input: A dictionary containing the user's input, with the keys 'claim' and 'date'.
    Returns:
        The model's final assessment of the claim as a string. The search results are also printed.
    """
    confident_message = {"role": "user", "content": "Tell me if you are confident to answer the question or not. Answer with 'yes' or 'no':"}
    hiss_config = load_config("Prompts/hiss_kshot.yaml")
    messages = hiss_config["hiss_messages"]
    print("Loaded HISS messages, including the system prompt and few-shot examples.")
    formatted_claim = "Claim: "+user_input["claim"]+". " 
    if user_input["author"]!="":
        formatted_claim += "Made by "+user_input["author"]+"."
    print("FORMATTED CLAIM: "+formatted_claim)
    messages.append({"role": "user", "content": formatted_claim}) 
    # Load the model from the Hugging Face Hub
    client = InferenceClient(model=model_id, token=API_TOKEN)
    print("Loaded the model from the Hugging Face Hub.")
    response = client.chat_completion(messages=messages, max_tokens=1000, temperature=temperature).choices[0].message.content
    print("LLM RESPONSE:"+response)
    question = extract_question(response)
    while question != "":
        messages.append({"role": "assistant", "content": response})
        messages.append(confident_message)
        response = client.chat_completion(messages=messages, max_tokens=1000, temperature=temperature).choices[0].message.content
        print("LLM RESPONSE:"+response)
        confidence = extract_yes_no(response)
        if confidence != "":
            if confidence == "YES": # If the model is confident, it will answer the question so
                messages.append({"role": "assistant", "content": response}) 
            elif confidence == "NO":
                search_results = google_search(question, date=user_input["date"])
                print(search_results)
                messages.append({"role": "user", "content": "Answer: "+search_results})
                response = client.chat_completion(messages=messages, max_tokens=1000, temperature=temperature).choices[0].message.content
                print("LLM RESPONSE:"+response)
        else:
            print("The model did not tell whether it is confident or not on answering the question.")
            break
        question = extract_question(response)
    return response


def main():
    print("Welcome to the Fact Checker! The language model "+model_name+" will verify your claim with the help of google search results.")
    # Get input from the user
    claim = input("Enter the claim to fact-check: ")
    date = input("Enter the date of the claim (optional): ")
    author = input("Enter the author of the claim (optional): ")
    user_input = {"claim": claim, "date": date, "author": author}
    # Generate output from the model
    output = load_model_and_generate_output(user_input)
    # Print the model's output
    #print("Model Output:\n" + str(output))

if __name__ == "__main__":
    main()