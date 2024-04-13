import json
import math
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def evaluateMultipleResponses(context, responses):

    with open('data/config.json', 'r') as config_file:
        config_data = json.load(config_file)
    evaluation = config_data.get('evaluation', {})
    threshold = {
        "perplexity": evaluation.get('perplexity'),
        "bleu": evaluation.get('bleu'),
        "rogue": evaluation.get('rouge'),
        "f1": evaluation.get('f1')
    }

    validResponsesPerformance = []
    validResponsesRanks = []

    for response in responses:
        result = evaluateResponse(context, response)
        validResponse = False
        for criteria, valueRange in threshold.items():
            if result[criteria] in valueRange:
                validResponse = True
        if validResponse:
            validResponsesPerformance.append([result["perplexity"], result["bleu"], result["rogue"], result["f1"]])
        else:
            validResponsesPerformance.append(None)
    
    for performance in validResponsesPerformance:
        if performance is None:
            validResponsesRanks.append(None)
        else:
            ranked_performance = []
            for i in range(4):
                sub_elements = [p[i] for p in validResponsesPerformance if p is not None]
                if i==0:
                    rank_value = sorted(sub_elements).index(performance[i]) + 1
                else:
                    rank_value = sorted(sub_elements, reverse=True).index(performance[i]) + 1
                ranked_performance.append(rank_value)
            validResponsesRanks.append(ranked_performance)

    min_sum = float('inf')
    best_response_index = -1
    for index, ranks in enumerate(validResponsesRanks):
        if ranks is not None:
            sum_ranks = sum(ranks)
            if sum_ranks < min_sum:
                min_sum = sum_ranks
                best_response_index = index

    if best_response_index != -1:
        bestResponse = responses[best_response_index]
    else:
        print("recall the function that tejas creates")
        bestResponse = ""
    return bestResponse
        
def evaluateResponse(context, response):
    result = dict()

    df = pd.read_csv('data/mentalHealthConversations.csv')
    df['Context'] = df['Context'].str.replace('\n', ' ').str.strip()
    df['Response'] = df['Response'].str.replace('\n', ' ').str.strip()
    df['Response'] = df['Response'].apply(lambda x: '' if pd.isnull(x) else x.strip('"'))
    df['Combined'] = df['Context'] + " " + df['Response']
    
    result["perplexity"] = calculatePerplexity(response)
    result["bleu"] = calculateBLEUscore(df, context, response)
    rouge, f1 = calculateROUGEandF1score(df, context, response)
    result["rogue"] = rouge
    result["f1"] = f1

    return result

def calculatePerplexity(response):
    words = response.split()
    sample_text = "I am going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I am worthless and how I should not be here. I have never tried or contemplated suicide. I have always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone?"
    language_model = {word: 1/len(sample_text.split()) for word in sample_text.split()}
    log_sum = sum(math.log((language_model.get(word, 0) + 1) / (len(language_model) + 1 * len(language_model)), 2) for word in words)
    perplexity = 2 ** (-log_sum / len(words))
    return perplexity

def calculateBLEUscore(df, context, response):
    reference_texts = df['Combined'].tolist()
    test_combined = context + " " + response
    bleu_score = sentence_bleu(reference_texts, test_combined)
    return bleu_score

def calculateROUGEandF1score(df, context, response):
    rouge = Rouge()
    reference_texts = "\n".join(df['Combined'].tolist())
    test_combined = context + " " + response
    rouge_scores = rouge.get_scores(test_combined, reference_texts)
    rouge_1_p = []
    rouge_2_p = []
    rouge_l_p = []
    for score in rouge_scores:
        rouge_1_p.append(score['rouge-1']['p'])
        rouge_2_p.append(score['rouge-2']['p'])
        rouge_l_p.append(score['rouge-l']['p'])
    average_rouge_1_p = sum(rouge_1_p) / len(rouge_1_p)
    average_rouge_2_p = sum(rouge_2_p) / len(rouge_2_p)
    average_rouge_l_p = sum(rouge_l_p) / len(rouge_l_p)
    overall_rouge_p = (average_rouge_1_p + average_rouge_2_p + average_rouge_l_p) / 3
    rouge_1_f = []
    rouge_2_f = []
    rouge_l_f = []
    for score in rouge_scores:
        rouge_1_f.append(score['rouge-1']['f'])
        rouge_2_f.append(score['rouge-2']['f'])
        rouge_l_f.append(score['rouge-l']['f'])
    average_rouge_1_f = sum(rouge_1_f) / len(rouge_1_f)
    average_rouge_2_f = sum(rouge_2_f) / len(rouge_2_f)
    average_rouge_l_f = sum(rouge_l_f) / len(rouge_l_f)
    overall_rouge_f = (average_rouge_1_f + average_rouge_2_f + average_rouge_l_f) / 3
    return overall_rouge_p, overall_rouge_f



if __name__ == "__main__":

    context = "I am going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I am worthless and how I should not be here. I have never tried or contemplated suicide. I have always wanted to fix my issues, but I never get around to it. How can I change my feeling of being worthless to everyone?"
    responses = ["It is clear that you are struggling with some deep feelings of worthlessness. One step towards change is acknowledging your feelings, so well done for reaching out. Have you noticed any specific triggers or patterns that lead to these thoughts about worthlessness? Understanding triggers can help identify strategies to manage these feelings. Can you pinpoint any situations or thoughts that tend to make you feel more worthless?",
             "Feeling worthless can be incredibly overwhelming, and it's positive that you want to address these feelings. Have you considered seeking support from a mental health professional? They can provide tools and techniques to help you navigate and overcome these emotions. What do you think about the idea of working with a therapist or counselor to explore these feelings and develop coping strategies?",
             "It's tough to feel like you're not valued or worthy. One way to start changing these feelings is by challenging negative thoughts and replacing them with more positive and realistic ones. Are there any activities or hobbies that bring you a sense of purpose or joy? Is there something you used to enjoy or have always wanted to try that could bring a sense of fulfillment and purpose to your life?",
             "It sounds like these feelings of worthlessness are weighing heavily on you. Have you considered practicing self-compassion and kindness towards yourself? Sometimes, treating ourselves with the same empathy we offer others can make a significant difference. What are some ways you can show yourself kindness and compassion during difficult times?",
             "Your desire to fix these feelings is a positive step forward. Have you explored any mindfulness or relaxation techniques? Sometimes, taking a moment to ground ourselves in the present can alleviate feelings of worthlessness. Would you be open to trying some relaxation exercises or mindfulness techniques to help calm your mind and manage these feelings?"]

    print("Best Response:", evaluateMultipleResponses(context, responses))