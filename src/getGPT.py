import openai
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import backoff 

#Adding the Keyvault URL
keyVaultName = "https://openaiapigroup45.vault.azure.net/"

credential = DefaultAzureCredential()
client_vault = SecretClient(vault_url=keyVaultName, credential=credential)


completion_tokens = prompt_tokens = 0

openai.api_type = 'azure'
openai.api_version = "2024-02-15-preview"

# Pulling the keyvault api key for Azure OpenAI
api_key = client_vault.get_secret("ApiKey").value
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
print(api_key)


# Pulling the keyvault base url for Azure OpenAI
api_base = client_vault.get_secret("BaseURL").value
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.base_url = api_base

azure_client = openai.AzureOpenAI(azure_endpoint = api_base, api_key= api_key, api_version = "2024-02-15-preview")


def gpt(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

# @backoff.on_exception(backoff.expo, openai.OpenAIError)
def completions_with_backoff(**kwargs):
    return azure_client.chat.completions.create(**kwargs)

def chatgpt(messages, model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(#engine="gpt35turbotejas",
                                       model=model, 
                                       messages=messages, 
                                       temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-3.5-turbo"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


