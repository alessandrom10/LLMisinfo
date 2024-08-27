import yaml
from Python_scripts.standard_fact_checker.sel_search import *
from huggingface_hub import InferenceClient
import os
import re
import ast

def extract_function_call(llm_output: str) -> str:
    # define a pattern that matches strings that starts with "Query: <query>"  and ends with a period
    pattern = r"[Qq]uery: *['\"](.*)['\"]\."
    # search for the pattern in the LLM output
    match = re.search(pattern, llm_output)
    if match:
        # extract <query> from the matched string
        query = match.group(1)
        print("Query: " + query)
        return query
    # if no match is found, return an empty string
    return ""
    

'''def extract_function_call(llm_output: str) -> str:
    """
    Extracts the function call from the LLM's output if present, 
    handling mixed usage of single and double quotes.
    
    Args:
        llm_output (str): The output string from the LLM.
    
    Returns:
        str: The extracted function call string if present, otherwise an empty string.
    """
    # Define the flexible pattern to search for the function call
    pattern = r"(\{['\"]name['\"]: *['\"]google_search['\"], *['\"]parameters['\"]: *(\{.*\})\})|" + \
              r'(\{[\'"]name[\'"]: *[\'"]google_search[\'"], *[\'"]parameters[\'"]: *(\{.*\})\})'
    
    # Search for the pattern in the LLM output
    match = re.search(pattern, llm_output)
    
    if match:
        # Extract the matched group (2 or 4, depending on the quote style)
        function_call_str = match.group(1) if match.group(1) else match.group(3)
        return function_call_str
    
    # If no match is found, return an empty string
    return ""'''

def load_config(filename='config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
max_searches = config['max_searches']
model_name = config['model_name']
model_id = config['model_id']
tools = config['tools']

standard_system_prompt = config['standard_system_prompt_1']
if max_searches == 1:
    #standard_system_prompt += "you must use the function only once. "
    standard_system_prompt += "you must make only one google search. "
else:
    #standard_system_prompt += "you can use the function from one up to "+str(max_searches)+" times. "
    standard_system_prompt += "you can make google searches from one up to "+str(max_searches)+" times. "
standard_system_prompt += config['standard_system_prompt_2']
#standard_system_prompt += "\n"+str(tools[0])

print(standard_system_prompt)   

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
    messages = [
         {"role": "system", "content": standard_system_prompt},
         {"role": "user", "content": "Claim: "+user_input["claim"]+". Date:"+user_input["date"]}
    ]
    num_searches = 0
    # Load the model from the Hugging Face Hub
    client = InferenceClient(model=model_id, token=API_TOKEN)
    response = client.chat_completion(messages=messages, max_tokens=1000).choices[0].message.content
    print(response)

    '''fc = extract_function_call(response)
    while fc!="": 
        num_searches += 1
        tool_call = ast.literal_eval(fc)
        query = tool_call["parameters"]["query"]
        search_results = google_search(query, date=user_input["date"])
        search_results = "Search number " + str(num_searches) + ":\n" + search_results
        print(search_results)
        messages.append({"role": "user", "content": search_results})
        response = client.chat_completion(messages=messages,max_tokens=1000).choices[0].message.content
        print(response)
        fc = extract_function_call(response)'''
    
    query = extract_function_call(response)
    while fc!="": 
        num_searches += 1
        search_results = google_search(query, date=user_input["date"])
        search_results = "Search number " + str(num_searches) + ":\n" + search_results
        print(search_results)
        messages.append({"role": "user", "content": search_results})
        response = client.chat_completion(messages=messages,max_tokens=1000).choices[0].message.content
        print(response)
        query = extract_function_call(response)
    return response


def main():
    print("Welcome to the Fact Checker! The language model "+model_name+" will verify your claim with the help of google search results.")
    # Get input from the user
    claim = input("Enter the claim to fact-check: ")
    date = input("Enter the date of the claim (optional): ")
    user_input = {"claim": claim, "date": date}
    # Generate output from the model
    output = load_model_and_generate_output(user_input)
    # Print the model's output
    #print("Model Output:\n" + str(output))

if __name__ == "__main__":
    main()