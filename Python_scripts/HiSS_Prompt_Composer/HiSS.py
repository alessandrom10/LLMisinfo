# Description: The original version of this script has been downloaded from here (https://github.com/jadeCurl/HiSS), but has since then been heavily modified
# It deals with the HiSS (Hierarchical Step-by-Step) prompting method to separate each claim into the subclaims that compose it,
# and for each of them it asks the llm if it is confident into proving its veracity,
# If the answer is "no", a standard answer "No, this is completely false" is returned
# and given to the model as an answer. At the very end it is asked to the llm to perform a final evaluation
# over the veracity of the whole statement considering the veracity of all of its subclaims
# This script fully works

from IPython.utils import io
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login
from Python_scripts.Search_scripts.sel_search import *
import os
import pandas as pd
import torch
import yaml

serpapi_key = os.getenv("SerpApi_KEY")
hf_key = os.getenv("HF_KEY")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_path = "./Datasets/claim_review_english_mapped.csv"
prompt_path = "./Prompts/hiss_kshot.yaml"

with open(prompt_path, 'r') as file:
  data = yaml.safe_load(file)

prompt = data["hiss_messages"]

login(token = hf_key)

bnb_config = BitsAndBytesConfig(
  load_in_4bit = True,
  bnb_4bit_use_double_quant = True,
  bnb_4bit_quant_type = "nf4",
  bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype = "auto",
  trust_remote_code = True,
  quantization_config = bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name, device = "cuda")
generator = pipeline("text-generation", model = model, tokenizer = tokenizer)

def promptf(claim, prompt, intermediate = "\nAnswer:", followup = "Intermediate Question", finalans= 'Based on'):
  '''
  parameters:
  claim - the claim to be fact-checked
  prompt - a list of two strings, the first one are the few shots for the HiSS prompting, while the second is the string "A fact checker will"
  intermediate - a string "\nAnswer:" to be put after the response of the llm when it answer no to the question if it has the confidence to answer the given question
  followup - a string "Intermediate question", that is actually never used it seems
  finalans - the string introducing the final predction
  '''

  system_prompt = generator.tokenizer.apply_chat_template(
    [prompt[0]], # Now it is a dictionary 
    tokenize = False, 
    add_generation_prompt = True
  )

  current_user_prompt = generator.tokenizer.apply_chat_template(
    [{
      "role" : "user",
      "content" : claim
    }],
    tokenize = False, 
    add_generation_prompt = True
  )

  separator = ", "

  ret_text = call_llama(system_prompt, current_user_prompt, stop = ' No.')
  print(f"Ret text is: {ret_text}")

  while 'Based on' not in ret_text and 'I would classify the claim as' not in ret_text: # Do this until you don't have the final answer (it has obtained all the answer to all the subqueries)
    question = extract_question(ret_text)
    print('Question:')
    print(question)
    print('External answer is:')
    external_answer = separator.join(get_answer(question)) # This function should retrieve the answer from the internet
    print(external_answer)
    current_user_prompt += ' ' + ret_text + intermediate + ' ' + external_answer + '.\n'
    ret_text = call_llama(system_prompt, current_user_prompt, stop = ' No.')

  returned_text = current_user_prompt + ret_text
  start_index = returned_text.index(finalans)
  end_of_phrase = returned_text.find('.', start_index) + 1
  returned_text = returned_text[:end_of_phrase]

  return returned_text

# This function that was added by me to the original script asks the current model to generate a response to the prompt received as input
def call_llama(system_message, user_message, stop=["\n"]):

  reasoner_messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
  ]

  prompt = generator.tokenizer.apply_chat_template(
    reasoner_messages, 
    tokenize = False, 
    add_generation_prompt = True
  )

  response = generator(
    prompt,
    num_return_sequences = 1,
    stop_sequence = stop,
    return_full_text = False,
    max_new_tokens = 2000
  )

  returned = response[0]['generated_text']
  return returned

# This function extracts the last sentence (and only that, not its answer) from the provided input
def extract_question(generated):
  generated = generated.split('Question: ')[-1].split('\nTell me if you are')[0]
  return generated

def get_answer(question):
  toret = google_search(question)
  return toret

df = pd.read_csv(dataset_path, encoding="utf-16", sep="\t", dtype={24: str})
start_row = 2 # This number refers to the claim's row in the excel file, so the claim in row 2 is actually the very first claim (of index 0)
df_subset = df.iloc[start_row - 2:]

for index, row in df_subset.iterrows():
  idx = row["id"]
  claim = row["claimReviewed"]
  label = row["reviewRating.alternateName"]
  
  print("Current id is: " + str(idx))
  ans = promptf(claim, prompt)

  print('\nOutput of the model is:\n')
  print(ans)
  print('\nThe label from the dataset is:')
  print(label)
