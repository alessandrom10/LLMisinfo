# This original version of this script has been downloaded from here (https://github.com/jadeCurl/HiSS), but has since then been heavily modified
# It deals with the HiSS (Hierarchical Step-by-Step) prompting method to separate each claim into the subclaims that compose it,
# and for each of them it asks the llm if it is confident into proving its veracity,
# If the answer is "no", then a fake call to the internet to search for the answer is performed, while actually the answer
# returned is allways "No, this is completely false", at the very end it is asked to the llm to perform a final evaluation
# over the veracity of the whole statement considering the veracity of all of its subclaims
# This script fully works, only the part about retrieving the answer from the internet doesn't, since that part has not been developed yet

from IPython.utils import io
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from huggingface_hub import login
#from serpapi import GoogleSearch
import os
import pandas as pd
import torch

serpapi_key = "" # get one from https://serpapi.com , first few requests are free!

hf_token = os.getenv("HF_TOKEN")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_path = "./Datasets/claim_review_english_mapped.csv"

prompt = ['''
Claim: "Emerson Moser, who was Crayola’s top crayon molder for almost 40 years, was colorblind."
A fact checker will decompose the claim into 4 subclaims that are easier to verify:
1.Emerson Moser was a crayon molder at Crayola.
2.Moser worked at Crayola for almost 40 years.
3.Moser was Crayola's top crayon molder.
4.Moser was colorblind.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is there any official record or documentation indicating that Emerson Moser worked as a crayon molder at Crayola?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there any official records or documentation confirming Emerson Moser's length of employment at Crayola?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Emerson Moser, who is retiring next week after 35 years, isn't colorblind in the sense that he can't see color at all. It's just that some ...
To verify subclaim 3, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there credible sources or publications that mention Emerson Moser as Crayola's top crayon molder?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
To verify subclaim 4, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there any credible sources or records indicating that Emerson Moser was colorblind?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
Question: Was Emerson Moser's colorblindness only confusing for certain colors?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Moser has had tritanomaly, a type of colorblindness that makes it difficult to distinguish between blue and green and between yellow and red.
Based on the answers to these questions, it is clear that among False, Mostly False, Mixture, Mostly True and True the claim can be classified as Mostly True.

Claim: "Bernie Sanders said 85 million Americans have no health insurance."
A fact checker will not split the claim since the original claim is easier to verify.
To verify the claim, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: How many Americans did Bernie Sanders claim had no health insurance?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: "We have 85 million Americans who have no health insurance," Sanders said Dec. 11 on CNN's State of the Union.
Question: How did Bernie Sanders define "no health insurance"?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Sanders spokesperson Mike Casca said the senator was referring to the number of uninsured and under-insured Americans and cited a report about those numbers for adults.
Question: How many Americans were uninsured or under-insured according to the Commonwealth Fund survey?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: The Commonwealth Fund survey found that 43% of working-age adults 19 to 64, or about 85 million Americans, were uninsured or inadequately insured.
Question: Is the statement "we have 85 million Americans who have no health insurance" partially accurate according to the information in the passage?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Bernie Sanders omitted that his figure included people who either have no health insurance or are under-insured.
Based on the answers to these questions, it is clear that among False, Mostly False, Mixture, Mostly True and True the claim is classified as Mostly True.

Claim: "JAG charges Nancy Pelosi with treason and seditious conspiracy."
A fact checker will decompose the claim into 2 subclaims that are easier to verify:
1. JAG has made a claim or accusation against Nancy Pelosi.
2. The specific charges or allegations made against Nancy Pelosi are treason and seditious conspiracy.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is it true that JAG has made a claim or accusation against Nancy Pelosi?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: There is no evidence to support this claim and a spokesperson for the U.S. Navy Judge Advocate General's Corps has stated that it is not true.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is it true that the specific charges or allegations made against Nancy Pelosi are treason and seditious conspiracy?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: There is no evidence to support this claim.
Question: Where is the source of the claim?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Real Raw News, a disclaimer stating that it contains "humor, parody and satire" and has a history of publishing fictitious stories.
Based on the answers to these questions, it is clear that among False, Mostly False, Mixture, Mostly True and True the claim is classified as False.

Claim: "Cheri Beasley “backs tax hikes — even on families making under $75,000."
A fact checker will decompose the claim into 2 subclaims that are easier to verify:
1.Cheri Beasley supports tax increases.
2.Cheri Beasley supports tax increases for families with an income under $75,000.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she might raise each question and look for an answer:
Question: Does Cheri Beasley support tax increases?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Beasley supports student loan bailouts for the wealthy.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she might raise each question and look for an answer:
Question: Does the ad accurately link Beasley's position on student loan debt forgiveness with her stance on tax hikes for families making under $75,000 per year?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: The ad makes a misleading connection between the two issues and does not accurately represent Beasley's position on tax hikes for families making under $75,000 per year.
Question: Has Cheri Beasley ever advocated for tax hikes specifically on families making under $75,000?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: No evidence found that Cheri Beasley has explicitly advocated for such a tax hike.
Based on the answers to these questions, it is clear that among False, Mostly False, Mixture, Mostly True and True the claim can be classified as Mostly True.


Claim: ''', '''A fact checker will''',
]

login(token = hf_token)

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

  system_prompt = prompt[0]
  current_user_prompt = claim + "\n" + prompt[1]

  ret_text = call_llama(system_prompt, current_user_prompt, stop = ' No.')

  while 'Based on' not in ret_text and 'I would classify the claim as' not in ret_text: # Do this until you don't have the final answer (it has obtained all the answer to all the subqueries)
    question = extract_question(ret_text)
    print('Question:')
    print(question)
    print('External answer is:')
    external_answer = get_answer(question) # This function should retrieve the answer from the internet
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
  """
  params = {
    "api_key": serpapi_key,
    "engine": "google",
    "q": question,
    "google_domain": "google.com",
    "gl": "us",
    "hl": "en"
  }
  with io.capture_output() as captured: # disables prints from GoogleSearch
    search = GoogleSearch(params)
    res = search.get_dict()

  if "organic_results" in res.keys():
    for idx in range(len(res["organic_results"])):
        if 'snippet' in res["organic_results"][idx].keys():
            if 'fact' not in res["organic_results"][idx]['link']:
              toret= res["organic_results"][idx]['snippet'] 
              break
        if (idx +1) == len(res["organic_results"]):
            toret = None
  else:
    toret = None
  return toret
  """
  toret = "No, this is completely false"
  return toret

df = pd.read_csv(dataset_path, encoding="utf-16", sep="\t", dtype={24: str})

for index, row in df.iterrows():
  idx = row["id"]
  claim = row["claimReviewed"]
  label = row["reviewRating.alternateName"]
  
  print("Current id is: " + str(idx))
  ans = promptf(claim, prompt)

  print('\nOutput of the model is:\n')
  print(ans)
  print('\nThe label from the dataset is:')
  print(label)
