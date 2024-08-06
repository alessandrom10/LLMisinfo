import yaml
from sel_search import *
from huggingface_hub import InferenceClient, login
import os
from transformers import Tool, CodeAgent, HfEngine

class GSerach(Tool):
    name = "google_search"
    description =" Get the google search results given a query, and an optional date to exclude results after that date."
    inputs = {
        "query": {
            "type": "text", 
            "description": "The query to search for on Google."
        },
        "date": {
            "type":"text",
            "description": "Only if user gave it, empty string otherwise. Results after this date will be excluded"
        }
    }
    output_type = "text"

    def forward(self, query: str, date: str) -> str:
        return google_search(query, date)


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
    standard_system_prompt += "you can use the function only once. "
else:
    standard_system_prompt += "you can use the function up to "+str(max_searches)+" times. "
standard_system_prompt += config['standard_system_prompt_2']

try:
    API_TOKEN = os.getenv("HF_TOKEN")
except:
    print("Please set the HF_TOKEN environment variable to your Hugging Face API token.")
    exit(1)

def load_model_and_generate_output(user_input):
    llm_engine = HfEngine(model_id)
    agent = CodeAgent(tools=[GSerach()], llm_engine=llm_engine)
    # Load the model from the Hugging Face Hub
    agent.run(standard_system_prompt+"\nClaim: "+user_input["claim"]+". Date:"+user_input["date"])

def main():
    login(API_TOKEN)
    print("Welcome to the Fact Checker! The language model "+model_name+" will verify your claim with the help of google search results.")
    # Get input from the user
    claim = input("Enter the claim to fact-check: ")
    date = input("Enter the date of the claim (optional): ")
    user_input = {"claim": claim, "date": date}
    # Generate output from the model
    #output = load_model_and_generate_output(user_input)
    load_model_and_generate_output(user_input)
    # Print the model's output
    #print("Model Output:\n" + str(output))

if __name__ == "__main__":
    main()