from fastapi import FastAPI, Request, Body
from pydantic import BaseModel
import json
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import numpy as np
import random
import string
import warnings
import requests
warnings.filterwarnings('ignore')
import pandas as pd
import nltk
import time
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
#nltk.download('all')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os

app2 = FastAPI()

def creer_base_de_donnees():
    # Connexion à la base de données (création si elle n'existe pas)
    conn = sqlite3.connect("sante.db")
    cursor = conn.cursor()
    
    # Création de la table maladie
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maladie (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL UNIQUE,
            informations TEXT NOT NULL
        )
    ''')
    
    # Chemin du répertoire contenant les fichiers de maladies
    repertoire_maladies = "Maladies"
    
    if os.path.exists(repertoire_maladies) and os.path.isdir(repertoire_maladies):
        for fichier in os.listdir(repertoire_maladies):
            chemin_fichier = os.path.join(repertoire_maladies, fichier)
            if os.path.isfile(chemin_fichier) and fichier.endswith(".txt"):
                with open(chemin_fichier, "r", encoding="utf-8") as f:
                    contenu = f.read()
                nom_maladie = os.path.splitext(fichier)[0]
                
                # Vérifier si la maladie existe déjà
                cursor.execute("SELECT id FROM maladie WHERE nom = ?", (nom_maladie,))
                if cursor.fetchone() is None:
                    cursor.execute("INSERT INTO maladie (nom, informations) VALUES (?, ?)", (nom_maladie, contenu))
    
    # Validation des changements et fermeture de la connexion
    conn.commit()
    conn.close()
    print("Base de données et table créées avec succès, informations insérées sans doublons.")

creer_base_de_donnees()

# Exemple d'utilisation

# Connexion à la base
conn = sqlite3.connect('sante.db')
cursor = conn.cursor()

# Compter les lignes
cursor.execute("SELECT COUNT(*) FROM maladie")
count = cursor.fetchone()[0]
   
print(f"Base de données mise à jour")
print(f"Nombre de lignes : {count}")

# Fermer la connexion
conn.close()



def afficher_informations_maladie(nom_maladie):
    # Connexion à la base de données
    conn = sqlite3.connect("sante.db")
    cursor = conn.cursor()
    
    # Requête pour récupérer les informations de la maladie
    cursor.execute("SELECT informations FROM maladie WHERE nom = ?", (nom_maladie,))
    resultat = cursor.fetchone()
    
    # Affichage du résultat
    if resultat:
        #print(f"Informations sur {nom_maladie} :\n{resultat[0]}")
        return resultat[0]
    else:
        return "Aucune information trouvée pour la maladie : " + {nom_maladie}
    
    # Fermeture de la connexion
    conn.close()

def tokenisation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens1=[]
    tokens1+=tokenizer.tokenize(text.lower())
    return tokens1

def repondre(utilisateur,maladie):

    # # Récupère le chemin du répertoire courant
    # chemin_courant = os.path.dirname(__file__)

    # # Spécifie le chemin relatif vers le fichier lions.txt
    # chemin_fichier = os.path.join(chemin_courant, 'Maladies/'+maladie+'.txt')

    # f= open(chemin_fichier,'r',errors='ignore',encoding="utf-8")
    # lignes = f.read()
    # Exemple d'utilisation
    lignes = afficher_informations_maladie(maladie)
    lignes = lignes.lower() 
    tokens_phrase = nltk.sent_tokenize(lignes) 
    tokens_mots = nltk.word_tokenize(lignes) 
    mots = stopwords.words('french')
    chatbot_rep =''
    tokens_phrase.append(utilisateur)
    TfidfVec = TfidfVectorizer(tokenizer = tokenisation, stop_words=mots) #english
    tfidf = TfidfVec.fit_transform(tokens_phrase)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf =flat[-2] 
    if (req_tfidf==0):
        chatbot_rep = chatbot_rep+"Je ne comprend pas. Pouvez vous reformulez votre question ?"
        return chatbot_rep
    else:
        chatbot_rep = chatbot_rep+tokens_phrase[idx]
        return chatbot_rep
    
# Endpoint GET pour tester l'API
@app2.get("/")
def read_root():
    return {"message": "API is working"}

@app2.post("/Q&A_Chatbot")   
async def Q_A(request: Request):
    try:
        data = await request.json()
        print(data)
        json_QA = json.loads(data)
        # Récupération des valeurs
        Question = json_QA["question_reponse"]
        maladie = json_QA["maladie"] 
        reponse = repondre(Question,maladie)
        print({"message": reponse})
        return {"message": reponse}
    except Exception as e:
        return {"error": str(e)}  

