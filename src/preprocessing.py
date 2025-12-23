import sys
import os
import re
import pickle
import bioc
import nltk
from nltk.tokenize import sent_tokenize

# Add the cloned repository to the system path so imports work
sys.path.append(os.path.abspath("ClinicalReport"))

# Try importing dependencies from the cloned repo
try:
    from text2bioc import text2document
    from ssplit import NegBioSSplitter
except ImportError:
    print("⚠️ Warning: ClinicalReport dependencies not found. Run setup_env.py first.")

# Categories defined in your notebook
CATEGORIES = [
    "Cardiomegaly", "Lung Lesion", "Airspace Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture"
]

# --- Helper Functions ---

def get_dict(path):
    label_to_mention = {}
    if not os.path.exists(path):
        return {}
        
    mention_files = os.listdir(path)
    for f in mention_files:
        with open(os.path.join(path, f), 'r') as mention_file:
            condition = os.path.basename(f)[:-4]
            condition = condition.replace("_", " ").title()
            if condition not in label_to_mention:
                label_to_mention[condition] = []

            for line in mention_file:
                label_to_mention[condition].append(line.split("\n")[0])
    return label_to_mention

# Load mentions/unmentions (Lazy loading pattern)
_MENTIONS = None

def get_mention_keywords(observation):
    global _MENTIONS
    if _MENTIONS is None:
        # Assuming the standard path inside the cloned repo
        phrases_path = "ClinicalReport/NegBio/negbio/chexpert/phrases/mention"
        if os.path.exists(phrases_path):
            _MENTIONS = get_dict(phrases_path)
        else:
            _MENTIONS = {}
            
    if observation in _MENTIONS:
        return _MENTIONS[observation]
    else:
        return []

def clean(sentence):
    punctuation_spacer = str.maketrans({key: f"{key} " for key in '.,'})
    lower_sentence = sentence.lower()
    corrected_sentence = re.sub('and/or', 'or', lower_sentence)
    corrected_sentence = re.sub('(?<=[a-zA-Z])/(?=[a-zA-Z])', ' or ', corrected_sentence)
    clean_sentence = corrected_sentence.replace('..', '.')
    clean_sentence = clean_sentence.translate(punctuation_spacer)
    clean_sentence = ' '.join(clean_sentence.split())
    return clean_sentence

def get_labels(sentence_l):
    """
    Negation prefilter , Keyword match
    """
    negative_word_l = ["no", "not", "doesn't", "does not", "have not", "can not", "can't", "n't"]
    observation_d = {}

    for cat in CATEGORIES:
        observation_d[cat] = False

    for s in sentence_l:
        s = s.lower()
        negative_flag = True

        for neg in negative_word_l:
            if neg in s:
                negative_flag = False
                break

        if negative_flag != False:
            for cat in CATEGORIES:
                for phrase in get_mention_keywords(cat):
                    phrase = phrase.lower()
                    if phrase in s:
                        observation_d[cat] = True
    return observation_d
