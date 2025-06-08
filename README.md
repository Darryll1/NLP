# Medibot – Chatbot Médical Intelligent en NLP

**Medibot** est un chatbot médical intelligent conçu pour analyser des descriptions de symptômes saisies en langage naturel, prédire la maladie probable à l’aide d’un modèle d’apprentissage non supervisé (KMeans), puis répondre aux questions des utilisateurs grâce à une base de connaissances médicale structurée.

## Objectif du projet

Dans un contexte où la surcharge des systèmes de santé ralentit l’accès à une orientation médicale rapide, **Medibot vise à fournir une réponse préliminaire fiable**, en simulant un dialogue médical à partir d’une simple saisie textuelle.

Le projet intègre :
- Prétraitement du texte avec NLTK
- Modélisation par clustering (KMeans)
- APIs RESTful pour modulariser les services
- Interface utilisateur via Streamlit


## Architecture technique

Le système repose sur **une architecture modulaire** articulée autour de 4 composants principaux :

|            Script       |                        Rôle                                           |
|------------------------ |  ---------------------------------------------------------------------|
| `prediction_diseases.py`| Interface utilisateur Streamlit, point d’entrée principal             |
| `API_BD.py`             | Création et gestion d’une base de données SQLite sur les maladies     |
| `API_diseases_prediction.py` | API RESTful pour la prédiction de maladies via KMeans            |
| `API_chatbot.py`        | API RESTful pour les réponses aux questions médicales                 |



##  Technologies utilisées

- **Langage** : Python
- **Frameworks** : FastAPI, Streamlit
- **NLP** : NLTK, RegexpTokenizer, TF-IDF, Bag of Words
- **Machine Learning** : KMeans (Scikit-learn)
- **Stockage** : SQLite
- **Support** : Pandas, NumPy


##  Fonctionnement du pipeline

1. **Extraction & nettoyage des textes** : suppression des stopwords, tokenisation (RegexpTokenizer)
2. **Vectorisation** :
   - Bag of Words (pour KMeans)
   - TF-IDF (pour similarité avec les textes médicaux)
3. **Modélisation** : Clustering KMeans (k=41, un cluster par pathologie)
4. **Réponse** : Similarité cosinus pour sélectionner la phrase la plus pertinente
5. **APIs REST** :
   - `/Kmeans` → retourne la maladie prédite
   - `/question` → répond aux questions sur la maladie


##  Cas d'utilisation

###  Cas 1 – Entrée non compréhensible
- **Input** : “Il fait beau aujourd’hui !”
- **Output** : “Pouvez-vous réessayer ? Je n’ai pas compris vos symptômes.”

### Cas 2 – Moins de 3 symptômes détectés
- **Input** : “J’ai mal à la tête”
- **Output** : “Reposez-vous, ça ira peut-être mieux demain.”

###  Cas 3 – Symptômes multiples détectés
- **Input** : Description longue avec plusieurs symptômes
- **Output** : Prédiction d'une maladie (via KMeans), puis possibilité de poser des questions


##  Exemple de question après diagnostic
- "Quels sont les traitements ?"
- "Cette maladie est-elle contagieuse ?"
- "Combien de temps dure-t-elle ?"

##  Modélisation & Évaluation

- **Modèle utilisé** : KMeans (non supervisé)
- **Données d'entraînement** : `Training.csv`, avec codage binaire des symptômes
- **Prétraitement** : Imputation, normalisation (StandardScaler)
- **Nombre de clusters** : 41
- **Évaluation** : `classification_report` (scikit-learn) sur étiquettes reconstruite depuis les clusters
  
## Arborescence du projet : 
Medibot/
├── API_BD.py
├── API_diseases_prediction.py
├── API_chatbot.py
├── prediction_diseases.py
├── training.csv
├── maladies/              # Dossiers de fichiers .txt de maladies
├── sante.db               # Base SQLite générée
└── requirements.txt

##   Lancer les APIs
uvicorn API_diseases_prediction:app --reload
uvicorn API_BD:app3 --reload --port 8002
uvicorn API_chatbot:app2 --reload --port 8001

###  Lancer l'interface utilisateur

streamlit run prediction_diseases.py
