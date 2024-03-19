from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature = 1., top_k = 10, max_tokens = 100)

llm = LLM(model="mistralai/Mistral-7B-v0.1", max_model_len = 16224)

# Ask a question
shots = "(1) What is the capital of France? \n The capital of France is Paris. [/end] (2) What is the biggest ocean on earth? \n The biggest ocean is the Pacific ocean [/end]"
question = "(3) Which is the tallest mountain on earth?"

# generate the answer
output = llm.generate(shots + question, sampling_params)

trimmed_answer = output[0].outputs[0].text.split("[/end] ")[0]

print("Question:", question)
print("Answer:", trimmed_answer)