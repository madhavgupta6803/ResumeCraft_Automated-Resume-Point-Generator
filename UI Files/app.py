import os
import chainlit as cl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

@cl.on_chat_start
def main():
    # Create the llm
    base_model_id = "bn22/Mistral-7B-Instruct-v0.1-sharded"
    adapter_id = "Pulkit506/mistral-7b-resume-iitb"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    llm = AutoModelForCausalLM.from_pretrained(
          base_model_id,  # Mistral, same as before
          quantization_config=bnb_config,  # Same quantization config as before
          device_map="auto",
          trust_remote_code=True,
        )
    llm.load_adapter(adapter_id)

    # Store the llm in the user session
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    cl.user_session.set("llm", llm)
      # Store the tokenizer in the user session
    cl.user_session.set("tokenizer", tokenizer) 
      # Set the chat title
    cl.chat_title("Resume Craft")
     # set the chat icon
    cl.chat_icon("https://icons8.com/icon/fyoqSj1xVmwr/roll-of-tickets")
    
  
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm = cl.user_session.get("llm")
      # Retrieve the tokenizer from the user session
    tokenizer = cl.user_session.get("tokenizer")
    

    msg = cl.Message(
        content="",
    )

    eval_prompt = f"[INST]{message.content}[/INST]"
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
      text = tokenizer.decode(llm.generate(**model_input, max_new_tokens=300, repetition_penalty=1.2, top_k=4, do_sample = True)[0], skip_special_tokens=True)

    processed_text = preprocess(text)
    for text in processed_text:
        await msg.stream_token(text)

    await msg.send()


def preprocess(text):
    text = text.replace("\n", "")
    text = text.replace("#", "")
    index = 0
    idx = []
    while(True):
      try:
          index = text.index("Answer",index+9)
          idx.append(index)
      except ValueError:
          break
    content = []

    for i in range(len(idx)-1):
      content.append(text[idx[i]+9:idx[i+1]-1])
    content.append(text[idx[len(idx)-1]+9:])
  
    points = []
    for i in content:
      point = i.split("',")
      points.append(point)
    
    cleaned_data = []
    for point in points:
      for text in point:
        if text[0] == "'":
          text = text[1:]
        text = text.replace("']","")
        text = text.replace(" '", "")
        cleaned_data.append(text)
    return cleaned_data





