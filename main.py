from flask import Flask, request
from transformers import Conversation, pipeline

app = Flask('')

@app.route('/')
def home(): 
    return "API is Online"

@app.route('/api/v1/talk', methods=['POST'])
def talk():
    print(request.data)
    conversation = Conversation(data)
    pipeline(conversation)
    return (0,conversation.generated_responses[-1])

def run(): 
    app.run(host='0.0.0.0',port=8080)

run()