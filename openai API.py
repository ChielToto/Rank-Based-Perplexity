from openai import OpenAI
import tiktoken
import math
import pandas as pd
import numpy as np


def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def linear_score(reftoken, tokens):
    """"Linear scoring function where 1 is the best and 0 the worst score"""
    tokens = [t.strip() for t in tokens]
    try:
        rank = tokens.index(reftoken.strip()) + 1
    except:
        return 0
    n = len(tokens)
    return (n - rank + 1) / n


def non_linear_score(reftoken, tokens, alpha=1.0):
    """Non-linear scoring function using exponential decay"""
    tokens = [t.strip() for t in tokens]
    try:
        rank = tokens.index(reftoken.strip()) + 1
    except:
        return 0
    return math.exp(-alpha * (rank - 1))

def reciprocal_score(reftoken, tokens):
    """"Scoring function that returns 1/rank"""
    tokens = [t.strip() for t in tokens]
    try:
        rank = tokens.index(reftoken.strip()) + 1
    except:
        return 0
    return 1 / rank

def perplexity(reftoken, tokens, logprobs):
    """"Returns the logprob of a generated token, if the token is not in the ranking,
    a penalty of the last logprob - 3 is returned"""
    tokens = [t.strip() for t in tokens]
    try:
        rank = tokens.index(reftoken.strip())
        return logprobs[rank]
    except:
        return logprobs[-1] - 3


text = 'reftext1'
your_folder = 'INSERT FOLDER NAME'
client = OpenAI(api_key=read_text(rf'{your_folder}/api_key.text'))

# Encodes the reference text according to the OpenAI model with GPT-4 as the base encoder
# This is used to start the models at the same token despite tokenizer differences
reftext = read_text(rf'{your_folder}\{text}.txt')
enc_base = tiktoken.encoding_for_model("gpt-4")
tokencoded_base = enc_base.encode(reftext)


def get_start_test_at(model, start_test_at):
    """Makes sure that the start_test_at points to the same token despite different tokenizers"""
    enc = tiktoken.encoding_for_model(model)
    start_tokens = tokencoded_base[:start_test_at]
    test_string = enc_base.decode(start_tokens)
    return len(enc.encode(test_string))


def generate_token_ranking(ref_text_sentence, model, Chat = True):
    """"Returns an array of generated tokens, an array of arrays of ranked tokens and
    an array of arrays of the corresponding logprobs. Chat is True is for the newer models,
    Chat = False should be used with models[4,5,6] (see above)"""
    tokens_total = []
    gentokens = []
    logprobs_total = []
    if Chat:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "Complete the text:"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ref_text_sentence
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=True,
            top_logprobs=top_logprobs
        )
        for i in range(max_tokens):
            tokens = []
            logprobs = []
            for t in range(top_logprobs):
                try:
                    tokens.append(response.choices[0].logprobs.content[i].top_logprobs[t].token)
                except:
                    pass
                    # print(f"FAILED AT {model} at token: {i} and logprob: {t} token")
                try:
                    logprobs.append(response.choices[0].logprobs.content[i].top_logprobs[t].logprob)
                except:
                    pass
                    # print(f"FAILED AT {model} at token: {i} and logprob: {t} logprob")
            try:
                gentokens.append(response.choices[0].logprobs.content[i].token)
            except:
                pass
                # print("FAILED AT GENTOKENS APPEND ", i)
            logprobs_total.append(logprobs)
            tokens_total.append(tokens)
    else:
        response = client.completions.create(
            model=model,
            prompt=ref_text_sentence,
            max_tokens=max_tokens,
            temperature=0,
            logprobs=5
        )
        for i in range(max_tokens):
            gentokens.append(response.choices[0].logprobs.tokens[i])
            logprobs_total.append(list(response.choices[0].logprobs.top_logprobs[i].values()))
            tokens_total.append(list(response.choices[0].logprobs.top_logprobs[i].keys()))
    return gentokens, tokens_total, logprobs_total


def generate_scores(random_starts, model, length, alpha, Exhaust = False, SaveData = False, Chat = True):
    """"Returns an array of arrays of scores for each token and scoring function. The order is:
    [#token 1#[Linear, Reciprocal, PPL, Non-linear alpha 1, Non-linear alpha 2, Non-linear alpha ...], #token2#[]]
    Max_tokens which is set equal to the test_length number of tokens are requested each time such that,
    when the model scored correctly, it can reuse the requested logprobs to check the next token.
    If Exhaust is True, the model will keep running until it has requested generated_token_ranking test_length times,
    else it will stop when it has test_length token scores."""
    enc = tiktoken.encoding_for_model(model)
    tokencoded = enc.encode(reftext)
    reftokens1 = enc.decode_tokens_bytes(tokencoded)
    reftokens = [reftok.decode('utf-8') for reftok in reftokens1]
    scores = [[] for a in range(len(alpha) + 3)]
    save_gentoken = []
    save_ranking = []
    save_logprobs = []
    save_reftoken = []

    if Exhaust:
        correct_count = 0
        for i in range(test_length):
            start_test_at = get_start_test_at(model, random_starts)
            start_index = i + start_test_at + correct_count
            ref_text_sentence = enc.decode(tokencoded[start_index - length:start_index])
            gentokens, all_ranks, all_logprobs = generate_token_ranking(ref_text_sentence, model, Chat)
            for j in range(max_tokens):
                start_index = i + start_test_at + correct_count
                linearscore = linear_score(reftokens[start_index], all_ranks[j])
                reciprocalscore = reciprocal_score(reftokens[start_index], all_ranks[j])
                ppl = perplexity(reftokens[start_index], all_ranks[j], all_logprobs[j])
                save_reftoken.append(reftokens[start_index])
                save_gentoken.append(gentokens[j])
                save_ranking.append(all_ranks[j])
                save_logprobs.append(all_logprobs[j])
                scores[0].append(linearscore)
                scores[1].append(reciprocalscore)
                scores[2].append(ppl)
                for a in alpha:
                    nonlinearscore = non_linear_score(reftokens[start_index], all_ranks[j], a)
                    scores[alpha.index(a) + 3].append(nonlinearscore)
                if linearscore == 1:
                    correct_count += 1
                    print('Efficiency gain count:', correct_count)
                    if j == max_tokens - 1:
                        #This is to make sure that if the model gets the max number of predictions
                        #correct, it will not skip the next token's evaluation
                        print('MAX TOKENS CORRECT EXCEEDED')
                        correct_count -= 1
                else:
                    break
        perplexity_final = math.exp(-(sum(scores[2]) / test_length))
        scores[2] = [perplexity_final for score in scores[2]]

    else:
        correct_count = 0
        for i in range(test_length):
            start_test_at = get_start_test_at(model, random_starts)
            start_index = i + start_test_at + correct_count
            ref_text_sentence = enc.decode(tokencoded[start_index - length:start_index])
            gentokens, all_ranks, all_logprobs = generate_token_ranking(ref_text_sentence, model, Chat)
            for j in range(max_tokens):
                start_index = i + start_test_at + correct_count
                linearscore = linear_score(reftokens[start_index], all_ranks[j])
                reciprocalscore = reciprocal_score(reftokens[start_index], all_ranks[j])
                ppl = perplexity(reftokens[start_index], all_ranks[j], all_logprobs[j])
                save_reftoken.append(reftokens[start_index])
                save_gentoken.append(gentokens[j])
                save_ranking.append(all_ranks[j])
                save_logprobs.append(all_logprobs[j])
                scores[0].append(linearscore)
                scores[1].append(reciprocalscore)
                scores[2].append(ppl)
                for a in alpha:
                    nonlinearscore = non_linear_score(reftokens[start_index], all_ranks[j], a)
                    scores[alpha.index(a) + 3].append(nonlinearscore)
                if len(scores[0]) == test_length:
                    break
                if linearscore == 1:
                    correct_count += 1
                    print('Efficiency gain count:', correct_count)
                    if j == max_tokens - 1:
                        #This is to make sure that if the model gets the max number of predictions
                        #correct, it will not skip the next token's evaluation
                        print('MAX TOKENS CORRECT EXCEEDED')
                        correct_count -= 1
                else:
                    break
            if len(scores[0]) == test_length: break
        perplexity_final = math.exp(-(sum(scores[2])/test_length))
        scores[2] = [perplexity_final for score in scores[2]]

    if SaveData:
        data = pd.DataFrame({'Reference token': save_reftoken, 'Generated token': save_gentoken, 'Token ranking': save_ranking, 'Token logprobs': save_logprobs})
        for t, score_array in enumerate(scores):
            if t == 0:
                data[f'Linear Score'] = score_array
            elif t == 1:
                data[f'Recipr. Score'] = score_array
            elif t == 2:
                data[f'Perplexity'] = score_array
            else:
                data[f'NL Score a={alpha[t-3]}'] = score_array
        final = pd.concat([data], axis=1)
        try:
            final.to_excel(f'Saved Data {random_starts} {text} {model} {length}.xlsx', index=False)
        except:
            print(final)
    return scores


def run_for_models_lengths(lengths, model_selection, alpha):
    """"Runs and saves the specified lengths for the models.
    Important: when using the exhaustive variant, the total input might exceed the context length of a model
    because it is going further than the test_length. This should be adjusted accordingly"""
    count = 0
    for length in lengths:
        data_frame = pd.DataFrame()
        random_start = np.random.randint(length, len(tokencoded_base) - length - test_length, size=1)[0]
        save_length = length
        for model in model_selection:
            scores_array = []
            if model == "gpt-4" and length == 8192:
                # This is to make sure then when GPT-4 is tested at 8192 context length it will not exceed
                # its max tokens. This way it will be able to request the right amount with 5 a token margin
                length = length - (2 * max_tokens) - 5
            else:
                length = save_length
            print(model, length, random_start)
            linear_score_array = generate_scores(random_start, model, length, alpha, Exhaust=False, SaveData=True, Chat=True)
            sums = 0
            for n, scores in enumerate(linear_score_array):
                avg_score = sum(scores)/len(scores)
                scores_array.append(avg_score)
                if not n == 2:
                    sums += avg_score
            final_sum = (sums/(len(linear_score_array)-1))
            scores_array.append(final_sum)
            data_frame[f'{model}'] = scores_array

        final_data = pd.concat([data_frame], axis=1)
        try:
            final_data.to_excel(f'Saved Data {count} {text} {length} AVERAGES.xlsx', index=False)
        except:
            "FAILED TO SAVE DATA:"
            print(final_data)
        count += 1


# lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# lengths = [128, 128, 128,128 ,128]
lengths = [2, 4]
alpha = [0.1, 0.3]
# The last three models are completion models and require Chat=False in the generate_scores function
#           0         1           2               3              4             5                       6
models = ["gpt-4o", "gpt-4","gpt-4-turbo", "gpt-3.5-turbo", "babbage-002", "davinci-002", "gpt-3.5-turbo-instruct"]
top_logprobs = 20
test_length = 2
max_tokens = test_length
model_selection = models[:4]

if __name__ == "__main__":
    run_for_models_lengths(lengths, model_selection, alpha)








