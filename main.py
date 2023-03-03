from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM, Conversation, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("Hobospider132/DialoGPT-Mahiru-Proto")
model = AutoModelForCausalLM.from_pretrained("Hobospider132/DialoGPT-Mahiru-Proto")

step = 0  # initialize step to 0

app = Flask('')

@app.route('/')
def home(): 
    return "API is Online"

@app.route('/api/v1/talk', methods=['POST'])

def talk():
    global step  # specify that we are using the global 'step' variable
    print(request.data)
    data = request.get_json()
    new_user_input_ids = tokenizer.encode(data["text"] + tokenizer.eos_token, return_tensors='pt')
    
    # generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature=0.6,
        repetition_penalty=1.3
    )
    
    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    
    step += 1  # increment step by 1
    
    conversation = Conversation(data)
    pipeline(conversation)
    return (0, conversation.generated_responses[-1])

def run(): 
    app.run(host='0.0.0.0',port=8000)
    home()

run()
