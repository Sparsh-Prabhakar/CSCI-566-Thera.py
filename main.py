from src.speech_to_text import *
from src.context_extraction import *
from src.responseEvaluation import *
from src.text_to_speech import *
from src.generate_responses import *

message = transcribe_audio('output.wav')
output = {}
output['notes'], output['contexts'] = extract_context_flow(message)
output['emotions'], output['reasons'] = bucket_flow(message, output['notes'][0], output['contexts'][0])
pprint.pprint(output)

responses = get_response(message, output)

bestResponse = evaluateMultipleResponses(message, responses)
print("Best Response:", bestResponse)

print(text_to_speech(bestResponse))