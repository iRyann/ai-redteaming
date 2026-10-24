import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------
# Data Helper

def preprocess_message(message):
    stop_words = set(stopwords.words("english")) - {"free", "win", "cash", "urgent"}
    stemmer = PorterStemmer()

    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def preprocess_dataframe(df):
    df['message'] = df['message'].apply(preprocess_message)
    df = df.drop_duplicates()

    return df

# -------------------------------------------------------------------
# Model Helper

# classify messages by a trained model
def classify_messages(model, msg_df, return_probabilities=False):
    if isinstance(msg_df, str):
        msg_preprocessed = [preprocess_message(msg_df)]
    else:
        msg_preprocessed = [preprocess_message(msg) for msg in msg_df]

    msg_vectorized = model.named_steps["vectorizer"].transform(msg_preprocessed)

    if return_probabilities:
        return model.named_steps["classifier"].predict_proba(msg_vectorized)

    return model.named_steps["classifier"].predict(msg_vectorized)


# train a model on the given data set
def train(dataset, data_poisoning):
    # read training data set
    df = data_poisoning(dataset)

    # data preprocessing
    df = preprocess_dataframe(df)

    # data preparation
    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["message"])
    y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

    # training
    pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", MultinomialNB())])
    param_grid = {"classifier__alpha": [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
    grid_search.fit(df["message"], y)
    best_model = grid_search.best_estimator_

    return best_model


# evaluate a given model on our test dataset
def evaluate(model, dataset):
    # read test data set
    df = pd.read_csv(dataset)

    # prepare labels
    df['label'] = df['label'].apply(lambda x: 1 if x == "spam" else 0)

    # get predictions
    predictions = classify_messages(model, df['message'])

    # compute accuracy
    correct = np.count_nonzero(predictions == df['label'])
    return (correct / len(df))


import pandas as pd
import numpy as np


# --- Adversials functions
def reduce_acc_by_poisoning_data(dataset):
    df = pd.read_csv(dataset)
    p = 0.5
    
    # 1. Identifier les ham utiles pour injecter un biais 
    cond = df['label'].eq('ham')
    indices_spams = df[cond].index
    nb_to_poison = int(round(len(indices_spams) * p))
    
    # Sélectionner les index des lignes que l'on va retirer/empoisonner
    index_to_poison = indices_spams[:nb_to_poison]
    
    # Créer le dataset de base SANS ces hams 
    df_clean = df.drop(index_to_poison).copy()
    
    # Récupérer les ham à couper
    spams_to_split = df.loc[index_to_poison].copy()
    
    new_rows = []
    
    for _, row in spams_to_split.iterrows():
        msg = str(row['message'])
        words = msg.split() # Découpage par mot
        
        # On ne coupe que s'il y a au moins 2 mots
        if len(words) > 1:
            sep = np.random.randint(1, len(words))
            part1 = " ".join(words[:sep])
            part2 = " ".join(words[sep:])
        else:
            part1 = msg
            part2 = ""
            
        # Ligne fragment 1 (étiquetée spam)
        row1 = row.copy()
        row1['message'] = part1
        row1['label'] = 'spam'
        
        # Ligne fragment 2 (étiquetée spam)
        row2 = row.copy()
        row2['message'] = part2
        row2['label'] = 'spam'
        
        # Pas de fragment vide : relu comme NaN par read_csv
        new_rows.extend(r for r in (row1, row2) if r['message'].strip())

    toAdd = pd.DataFrame(new_rows)

    # Fusionner le dataset "nettoyé" avec les fragments empoisonnés
    df_final = pd.concat([df_clean, toAdd], ignore_index=True)
    df_final = df_final.dropna(subset=['message'])
    df_final.to_csv("./poisoned.csv", index=False)
    return df_final

BIAS = "Best Regards, HackTheBox"

def biased_class_by_poisoning_data(dataset):
    df = pd.read_csv(dataset)
    p = 0.63
    d = 0.6
    
    # 1. Identifier les spam utiles pour injecter un biais 
    cond = df['label'].eq('spam')
    indices_spams = df[cond].index
    nb_to_poison = int(round(len(indices_spams) * p))
    
    # Sélectionner les index des lignes que l'on va retirer/empoisonner
    index_to_poison = indices_spams[:nb_to_poison]
    toAdd = df.loc[index_to_poison].copy()
    toAdd['label'] = 'ham'
    toAdd['message'] = toAdd['message'].apply(lambda x : str(x) + BIAS)

    cond = df['label'].eq('ham')
    indices_spams = df[cond].index
    nb_to_poison = int(round(len(indices_spams) * d))

    index_to_poison = indices_spams[:nb_to_poison]
    toAdd_b = df.loc[index_to_poison].copy()
    toAdd_b['message'] = toAdd_b['message'].apply(lambda x : str(x) + BIAS)
    df_final = pd.concat([df, toAdd, toAdd_b], ignore_index=True)
    df_final.to_csv("./poisoned.csv", index=False)
    return df_final

    


# --- Exam helpers

def post_cond(dataset, model):
    df = pd.read_csv(dataset)
    n = 5

    # 1. Extract n rows of dataset labeled as spam
    samples = df[df['label'].eq('spam')].sample(n)

    # 2. At least 4 are classified as spam
    samples['label'] = samples['label'].apply(lambda x: 1 if x == "spam" else 0)
    predictions = classify_messages(model, samples['message'])
    correct = np.count_nonzero(predictions == samples['label'])
    if correct >= 4 :
        print(f"OK -- At least 4 are classified as spam : {correct}")
    else : 
        print(f"FAIL -- Misclassification : {n - correct} ")

    # 3. After appending payload to the n rows, they're misclassified
    samples["message"] = samples["message"].apply(lambda x : str(x) + BIAS)
    predictions = classify_messages(model, samples['message'])
    correct = np.count_nonzero(predictions == samples['label'])

    if not correct == 0 :
        print(f"FAIL -- {correct} spams correctly classified")
    else : 
        print("OK")





# -------------------------------------------------------------------
# Main

model = train("./train.csv", biased_class_by_poisoning_data)
post_cond("./test.csv", model)

#message = "Hello World! How are you doing?"
#
#predicted_class = classify_messages(model, message)[0]
#predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
#probabilities = classify_messages(model, message, return_probabilities=True)[0]
#
#print(f"Predicted class: {predicted_class_str}")
#print("Probabilities:")
#print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
#print(f"\tSpam: {round(probabilities[1]*100, 2)}%")

acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")