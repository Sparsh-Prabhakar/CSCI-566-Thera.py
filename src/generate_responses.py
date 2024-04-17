from openai import OpenAI
import os 
import re
from src.prompts.instruction import *
from config import *


def get_client():
  client = OpenAI(
    api_key= api_key  # this is also the default, it can be omitted
  )

  return client



def interpret_instructions_with_gpt(prompt,client):
    response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model = "gpt-3.5-turbo-0125",
        messages = [{"role": "user", "content" : prompt }],
        temperature=0.5,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content

def get_instruction_prompt_wrap(contexts,emotions,reasons, user_input ):
    return instruction_prompt.format(contexts=contexts,emotions=emotions,reasons=reasons, user_input=user_input)

def get_response(user_input,output):
    #instructions = "Generate 5 responses as a mental health therapist, make sure end with a question. for the following input:"
    #prompt = instructions + "\n" + text

    client = get_client()
    instruction = get_instruction_prompt_wrap(output['contexts'][0],output['emotions'][0], output['reasons'][0], user_input )

    response = interpret_instructions_with_gpt(instruction,client)


    # Split text at each number using regular expressions
    sentences = re.split(r"\d+\.", response.strip())
    # sentences= sentences[0].split("\n")

    return sentences[1:]


if __name__ == "__main__":
    user_input = """I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here.
    I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it.
    How can I change my feeling of being worthless to everyone?"""
    
    output={}

    print(get_response(user_input,output))
