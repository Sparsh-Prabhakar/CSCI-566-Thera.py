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