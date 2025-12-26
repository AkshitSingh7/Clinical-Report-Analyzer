import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

# Global variables for caching model/tokenizer
_TOKENIZER = None
_MODEL = None

def load_bert_model():
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        print("Loading BERT model...")
        # Check if local model exists, otherwise let transformers download standard bert
        model_path = './models' if os.path.exists('./models/tf_model.h5') else 'bert-base-uncased'
        
        _TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        _MODEL = TFAutoModelForQuestionAnswering.from_pretrained(model_path)
        print("BERT model loaded.")
    return _TOKENIZER, _MODEL

def prepare_bert_input(question, passage, tokenizer, max_seq_length=384):
    question_tokens = tokenizer.tokenize(question)
    passage_token = tokenizer.tokenize(passage)

    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token

    tokens = [CLS] + question_tokens + [SEP] + passage_token
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Padding
    input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
    input_mask = input_mask + [0] * (max_seq_length - len(input_mask))

    return tf.expand_dims(tf.convert_to_tensor(input_ids), 0), input_mask, tokens

def get_span_from_scores(start_scores, end_scores, input_mask):
    n = len(start_scores)
    max_sum = -np.inf
    max_start_i = -1
    max_end_j = -1

    for i in range(n):
        for j in range(i, n):
            if (input_mask[i] == 1) & (input_mask[j] == 1):
                if (start_scores[i] + end_scores[j]) > max_sum:
                    max_sum = start_scores[i] + end_scores[j]
                    max_start_i = i
                    max_end_j = j
    return max_start_i, max_end_j

def construct_answer(tokens):
    out_string = ' '.join(tokens)
    out_string = out_string.replace(' ##', '')
    out_string = out_string.strip()
    if '@' in tokens:
        out_string = out_string.replace(' ', '')
    return out_string

def get_model_answer(question, passage):
    tokenizer, model = load_bert_model()
    
 
    max_seq_length = 384
    
    input_ids, input_mask, tokens = prepare_bert_input(question, passage, tokenizer, max_seq_length)
    
    outputs = model(input_ids)
    start_scores = outputs.start_logits.numpy()[0]
    end_scores = outputs.end_logits.numpy()[0]

    span_start, span_end = get_span_from_scores(start_scores, end_scores, input_mask)
    answer_tokens = tokens[span_start:span_end + 1]
    
    return construct_answer(answer_tokens)
