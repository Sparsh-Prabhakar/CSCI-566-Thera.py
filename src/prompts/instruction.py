instruction_prompt = """
    So far, for your context, this is what we have discussed: 
    {contexts}.
    Take the following emotion for setting the tone of the coversation: 
    {emotions}.
    Following is the reason why we are choosing to maintain the given emotion:
    {reasons}.
    Now, your task is to generate 5 responses as a mental health therapy bot, make sure end with a question. for the following input:
    {user_input}.

"""