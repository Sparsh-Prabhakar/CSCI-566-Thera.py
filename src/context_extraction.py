from src.getGPT import *
from src.prompts.context_extraction import *
import pprint


def get_context_prompt_wrap(messages,):
    return context_prompt.format(input=messages)

def context_response_unwrap(message):
    if "Notes" in message and "Context" in message:
        notes_start = message[message.find("Notes"):].find(':') +2
        notes_end = message.find("Context") - 1

        context_start =  message[message.find("Context"):].find(':') + notes_end + 3
        
        return message[notes_start:notes_end].strip(), message[context_start:].strip()
    else:
        print("Recall the GPT function")
        return None, None

def extract_context_flow(message,model='gpt35turbotejas',n=1):
    prompt = get_context_prompt_wrap(message)
    responses = gpt(prompt,model=model,n=n)
    notes = []
    contexts = []

    for response in responses:
        note, context = context_response_unwrap(response)

        notes.append(note)
        contexts.append(context)

    return notes, contexts


def get_bucket_prompt_wrap(message, notes, context):
    return bucket_prompt.format(message=message,notes=notes,context=context)


def bucket_response_unwrap(message):
    if "Emotion" in message and "Reason" in message:
        emotion_start = message[message.find("Emotion"):].find(':') +2
        emotion_end = message.find("Reason") - 1

        reason_start =  message[message.find("Reason"):].find(':') + emotion_end + 3
        
        return message[emotion_start:emotion_end].strip(), message[reason_start:].strip()
    else:
        print("Recall the GPT function")
        return None, None

def bucket_flow(message, notes, context ,model='gpt35turbotejas',n=1):
    prompt = get_bucket_prompt_wrap(message, notes, context)
    responses = gpt(prompt,model=model,n=n)
    emotions = []
    reasons = []

    for response in responses:
        emotion, reason = bucket_response_unwrap(response)

        emotions.append(emotion)
        reasons.append(reason)

    return emotions, reasons


if __name__ == '__main__':
    message = "Hello"

    output = {}

    output['notes'], output['contexts'] = extract_context_flow(message)
    output['emotions'], output['reasons'] = bucket_flow(message, output['notes'][0], output['contexts'][0])

    pprint.pprint(output)