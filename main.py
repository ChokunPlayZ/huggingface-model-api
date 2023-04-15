from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM, Conversation, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("Hobospider132/DialoGPT-Mahiru-Proto")
model = AutoModelForCausalLM.from_pretrained("Hobospider132/DialoGPT-Mahiru-Proto")

step = 0  # initialize step to 0

app = Flask('')

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    response = model.generate(
      input_ids=input_ids, 
      max_length=1000,
      pad_token_id=tokenizer.eos_token_id,  
      no_repeat_ngram_size=3,       
      do_sample=True, 
      top_k=100, 
      top_p=0.7,
      temperature=0.8,
      repetition_penalty = 1.3
    )
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    response_text = response_text.replace(prompt, "").strip() # Remove the prompt text from the response
    return response_text

@app.route('/')
def home():
    return "API is Online"

@app.route('/api/v1/talk', methods=['POST'])
def talk():
    data = request.json
    prompt = data["prompt"]
    response = generate_response(prompt)
    return {"response": response}

def run():
    app.run(host='0.0.0.0',port=9999)

print(generate_response("Hi,"))

run()
