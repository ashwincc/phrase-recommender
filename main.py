 
# importing necessary libraries 

import spacy #spacy for cleaning the input text
import gradio as gr # gradio for web interface
from sentence_transformers import SentenceTransformer, util


encoding_model = 'all-MiniLM-L6-v2'
spacy_model = 'en_core_web_sm'

def load_text_to_analyse(text):
    # with open(file_path, 'r') as f:
    #     text = f.read()
    nlp = spacy.load(spacy_model)
    sentences = [i.text for i in nlp(text).sents]

    return sentences


def load_std_text(file_path):
    with open(file_path, 'r') as fp:
        return fp.read().splitlines()

def emb_enconde(text):
    model = SentenceTransformer(encoding_model)
    return model.encode(text, convert_to_tensor=True)


def cal_cosine_scores(emb1, emb2):
    return util.cos_sim(emb1, emb2)


def get_high_cos_sim_pairs(emb1, emb2, cosine_scores):
        #Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(emb1)):
        for j in range(len(emb2)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    #Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    return pairs

def get_results(cos_pairs, text, std_text):
    result = []
    for pair in cos_pairs[0:10]:
        i, j = pair['index']
        result.append((text[i], std_text[j], pair['score']))
    
    re = {}
    for t, p, s in result:
        re.setdefault(t,[]).append([p, round(s.item(), 3)])
    return re


def main(text):
    std_phrase = 'Standardised terms.csv'
    text_to_analyse = load_text_to_analyse(text)
    std_text = load_std_text(std_phrase)
    text_emb = emb_enconde(text)
    std_emb = emb_enconde(std_text)
    pairs = get_high_cos_sim_pairs(text_emb, std_emb, cal_cosine_scores(text_emb, std_emb))
    output = get_results(pairs, text, std_text)
    return output


examples = [
    "Does Chicago have any stores and does Joe live here?",
]
if __name__ == '__main__':

    demo = gr.Interface(main,
             gr.Textbox(placeholder="Enter sentence here..."),
                gr.Textbox(),
             examples=examples)

    demo.launch()
