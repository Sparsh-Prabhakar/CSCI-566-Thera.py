context_prompt = """
Consider you are a therapy bot and a person is talking to you.
Your tasks are 
1. To understand what the person is saying and make notes. 
2. Derive a context of why the person might be saying that stuff.

Return me a response in following format - 
Notes : 
    <notes>

Context :
    <context>

Here is the message - 
    {input}
"""


buckets = ["Happiness", "Sadness", "Fear", "Disgust", "Anger", "Surprise"]

bucket_prompt = """
Consider you are a therapy bot and a person is talking to you.
You will be given the message, and notes from it and context of the message. 
Your task is to understand all of it and find the emotion behind the message. 
The emotion should be one of the following -> """+str(buckets)+"""

Return me a response in following format - 
Emotion: <one of the six emotions>
Reason:
    <Reason behind the mentioned emotion>

Heres the input - 
Message:
    {message}
Notes:
    {notes}
Context:
    {context}
"""