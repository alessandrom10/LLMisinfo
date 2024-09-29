import sys
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
kshot_path_en = my_config['kshot_path_en'] # To change
kshot_path_it = my_config['kshot_path_it']
#kshot_path_es = my_config['kshot_path_es']

if language == "en":
    print("English language selected.")
    kshots = load_config(kshot_path_en)["hiss_messages"]
elif language == "it":
    print("Lingua italiana selezionata.")
    kshots = load_config(kshot_path_it)["hiss_messages"]
elif language == "es":
    print("Lengua española seleccionada.")
    kshots = load_config(kshot_path_es)["hiss_messages"]
print("Loaded HiSS configuration: ", kshots[0])

# Load the Hugging Face API token from the environment variables
try:
    API_TOKEN = os.getenv("HF_TOKEN")
except:
    print("Please set the HF_TOKEN environment variable to your Hugging Face API token.")
    exit(1)

# Load the model from the Hugging Face Hub
print("Loading model " + model_id + "...\n")
client = InferenceClient(model = model_id, token = API_TOKEN)

def extract_question(llm_output: str) -> str:
    """
    The llm is expected to output the question in the format "Question: <question>". This function extracts the question from the llm output.
    Args:
        llm_output: The output of the llm.
    Returns:
        The extracted question is returned as a string. If no question is found, an empty string is returned.
    """
    # Define a pattern that matches strings that includes "Question:"
    if language == "en":
        pattern = r"Question: (.+)"
    elif language == "it":
        pattern = r"Domanda: (.+)"
    elif language == "es":
        pattern = r"Pregunta: (.+)"
    # Search for the pattern in the LLM output, not caring about the upper/lower case
    match = re.search(pattern, llm_output, re.IGNORECASE)
    if match:
        # Extract the matched string (so the first question, regardeless of whether the llm is confident in answering it or not)
        question = match.group(1)
        print("Question posed by the LLM: " + question)
        return question
    # If no match is found, we return an empty string
    return ""

def extract_yes_no(llm_output: str) -> str:
    """
    The llm is expected to output the confidence it has in answering the question in the format "Yes" or "No" (because the whole output is made uppercase).
    This function extracts the confidence from the llm output.
    Args:
        llm_output: The output of the llm.
    Returns:
        The extracted confidence is returned as a string. If no confidence is found, an empty string is returned.
    """
    # Define a pattern that matches strings that include "Yes" or "No"
    if language == "en":
        pattern = r"(Yes|No)"
    elif language == "it":
        pattern = r"(S[ìi]|No)"
    elif language == "es":
        pattern = r"(Si|No)"
    # Search for the pattern in the LLM output in the first 5 characters, not caring about the upper/lower case
    match = re.search(pattern, llm_output[:5], re.IGNORECASE)
    if match:
        # If the extracted string is "Si" (which means yes in Italian), translate it to english and return that instead
        confidence = match.group(1).lower()
        if confidence == "sì" or confidence == "si":
            confidence = "yes"
        return confidence
    # If no match is found, return an empty string
    return ""
    

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
    
    """question = extract_question(response)
    while question != "":
        messages.append({"role": "assistant", "content": response})
        messages.append(confident_message) # We add to messages both what the assistant has given as output as well as the request if it's confident by user
        print("<user> " + confident_message["content"])
        # We ask the llm for an answer
        response = chat_completion(messages = messages)
        print("<assistant> " + response)
        confidence = extract_yes_no(response)
        if confidence != "":
            if confidence == "yes": # If the model is confident, it will also provide an answer to the question so:
                messages.append({"role": "assistant", "content": response}) 
            elif confidence == "no": # Otherwise we search on Google
                search_results = google_search(question, date = user_input["date"], claim_domain = user_input["domain"])
                if language == "en":
                    messages.append({"role": "user", "content": "Answer: " + search_results})
                elif language == "it":
                    messages.append({"role": "user", "content": "Risposta: " + search_results})
                elif language == "es":
                    messages.append({"role": "user", "content": "Respuesta: " + search_results})
                print("<user> " + messages[-1]["content"])
                response = chat_completion(messages = messages)
                print("<assistant> " + response)
        else:
            print("The model did not tell whether it is confident or not on answering the question.")
            break
        question = extract_question(response) # And we repeat this for every question the llm has dealt with (so until question == "")"""

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
    domain = ""
    user_input = {"claim": claim, "date": date, "author": author, "domain": domain}

    # Generate output from the model
    output = generate_output(user_input)

    # Print the model's output
    print("Model Output:\n" + str(output))

if __name__ == "__main__":
    main()
