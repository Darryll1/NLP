from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi import Body
import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report


# Initialisation de l'application FastAPI
app = FastAPI()

dict_maladies_traduction = {
    "Fungal infection": "Infection fongique",
    "Allergy": "Allergie",
    "GERD": "Reflux gastro-œsophagien",
    "Chronic cholestasis": "Cholestase chronique",
    "Drug Reaction": "Réaction médicamenteuse",
    "Peptic ulcer disease": "Maladie de l'ulcère peptique",
    "AIDS": "SIDA",
    "Diabetes": "Diabète",
    "Gastroenteritis": "Gastroentérite",
    "Bronchial Asthma": "Asthme bronchique",
    "Hypertension": "Hypertension artérielle",
    "Migraine": "Migraine",
    "Cervical spondylosis": "Spondylose cervicale",
    "Paralysis (brain hemorrhage)": "Paralysie (hémorragie cérébrale)",
    "Jaundice": "Ictère",
    "Malaria": "Paludisme",
    "Chicken pox": "Varicelle",
    "Dengue": "Dengue",
    "Typhoid": "Typhoïde",
    "Hepatitis A": "Hépatite A",
    "Hepatitis B": "Hépatite B",
    "Hepatitis C": "Hépatite C",
    "Hepatitis D": "Hépatite D",
    "Hepatitis E": "Hépatite E",
    "Alcoholic hepatitis": "Hépatite alcoolique",
    "Tuberculosis": "Tuberculose",
    "Common Cold": "Rhume",
    "Pneumonia": "Pneumonie",
    "Dimorphic hemmorhoids(piles)": "Hémorroïdes",
    "Heart attack": "Infarctus du myocarde",
    "Varicose veins": "Varices",
    "Hypothyroidism": "Hypothyroïdie",
    "Hyperthyroidism": "Hyperthyroïdie",
    "Hypoglycemia": "Hypoglycémie",
    "Osteoarthristis": "Ostéoarthrite",
    "Arthritis": "Arthrite",
    "(vertigo) Paroymsal  Positional Vertigo": "Vertige positionnel paroxystique",
    "Acne": "Acné",
    "Urinary tract infection": "Infection des voies urinaires",
    "Psoriasis": "Psoriasis",
    "Impetigo": "Impétigo"
}

dict_symptomes_traduction = {
    "itching": "Démangeaison",
    "skin_rash": "Eruption cutanée",
    "nodal_skin_eruptions": "Éruptions cutanées nodales",
    "continuous_sneezing": "Éternuements continus",
    "shivering": "Frisson",
    "chills": "Frissons",
    "joint_pain": "Douleur articulaire",
    "stomach_pain": "Douleur estomac",
    "acidity": "Acidité",
    "ulcers_on_tongue": "Ulcères sur la langue",
    "muscle_wasting": "Atrophie musculaire",
    "vomiting": "Vomissement",
    "burning_micturition": "Miction brûlante",
    "spotting_urination": "Urination ponctuelle",
    "fatigue": "Fatigue",
    "weight_gain": "Gain de poids",
    "anxiety": "Anxiété",
    "cold_hands_and_feets": "Mains et pieds froids",
    "mood_swings": "Changements d'humeur",
    "weight_loss": "Perte de poids",
    "restlessness": "Agitation",
    "lethargy": "Léthargie",
    "patches_in_throat": "Patches dans la gorge",
    "irregular_sugar_level": "Niveau de sucre irrégulier",
    "cough": "Toux",
    "high_fever": "Fièvre élevée",
    "sunken_eyes": "Yeux enfoncés",
    "breathlessness": "Essoufflement",
    "sweating": "Sueur",
    "dehydration": "Déshydratation",
    "indigestion": "Indigestion",
    "headache": "Céphalée",
    "yellowish_skin": "Peau jaunâtre",
    "dark_urine": "Urines foncées",
    "nausea": "Nausée",
    "loss_of_appetite": "Perte d'appétit",
    "pain_behind_the_eyes": "Douleur derrière les yeux",
    "back_pain": "Douleur de dos",
    "constipation": "Constipation",
    "abdominal_pain": "Douleur abdominale",
    "diarrhoea": "Diarrhée",
    "mild_fever": "Fièvre légère",
    "yellow_urine": "Urines jaunes",
    "yellowing_of_eyes": "Ictère",
    "acute_liver_failure": "Insuffisance hépatique aiguë",
    "fluid_overload": "Surcharge de liquides",
    "swelling_of_stomach": "Gonflement de l'estomac",
    "swelled_lymph_nodes": "Ganglions lymphatiques enflés",
    "malaise": "Malaise",
    "blurred_and_distorted_vision": "Vision floue et déformée",
    "phlegm": "Crachat",
    "throat_irritation": "Irritation de la gorge",
    "redness_of_eyes": "Rougeur des yeux",
    "sinus_pressure": "Pression sinusoïdale",
    "runny_nose": "Écoulement nasal",
    "congestion": "Congestion",
    "chest_pain": "Douleur thoracique",
    "weakness_in_limbs": "Faiblesse dans les membres",
    "fast_heart_rate": "Rythme cardiaque rapide",
    "pain_during_bowel_movements": "Douleur pendant les mouvements intestinaux",
    "pain_in_anal_region": "Douleur dans la région anale",
    "bloody_stool": "Selles sanglantes",
    "irritation_in_anus": "Irritation de l'anus",
    "neck_pain": "Douleur de cou",
    "dizziness": "Vertige",
    "cramps": "Crampes",
    "bruising": "Ecchymoses",
    "obesity": "Obésité",
    "swollen_legs": "Jambes enflées",
    "swollen_blood_vessels": "Vaisseaux sanguins enflés",
    "puffy_face_and_eyes": "Visage et yeux gonflés",
    "enlarged_thyroid": "Thyroïde élargie",
    "brittle_nails": "Ongles cassants",
    "swollen_extremeties": "Extrémités enflées",
    "excessive_hunger": "Faim excessive",
    "extra_marital_contacts": "Contacts extra-conjugaux",
    "drying_and_tingling_lips": "Lèvres sèches et picotantes",
    "slurred_speech": "Parole confuse",
    "knee_pain": "Douleur de genou",
    "hip_joint_pain": "Douleur de 'articulation de la hanche",
    "muscle_weakness": "Faiblesse musculaire",
    "stiff_neck": "Cou raide",
    "swelling_joints": "Enflure des articulations",
    "movement_stiffness": "Raideur des mouvements",
    "spinning_movements": "Mouvements de rotation",
    "loss_of_balance": "Perte d'équilibre",
    "unsteadiness": "Instabilité",
    "weakness_of_one_body_side": "Faiblesse d'un côté du corps",
    "loss_of_smell": "Perte de l'odorat",
    "bladder_discomfort": "Inconfort de la vessie",
    "foul_smell_of_urine": "Odeur nauséabonde de l'urine",
    "continuous_feel_of_urine": "Sensation continue d'urine",
    "passage_of_gases": "Passage de gaz",
    "internal_itching": "Démangeaison interne",
    "toxic_look_(typhos)": "Aspect toxique (typhos)",
    "depression": "Dépression",
    "irritability": "Irritabilité",
    "muscle_pain": "Douleur musculaire",
    "altered_sensorium": "Sensorium altéré",
    "red_spots_over_body": "Taches rouges sur le corps",
    "abnormal_menstruation": "Ménstruation anormale",
    "dischromic_patches": "Patches dischromiques",
    "watering_from_eyes": "Larmes",
    "increased_appetite": "Appétit accru",
    "polyuria": "Polyurie",
    "family_history": "Antécédents familiaux",
    "mucoid_sputum": "Crachat mucide",
    "rusty_sputum": "Crachat rouillé",
    "lack_of_concentration": "Manque de concentration",
    "visual_disturbances": "Troubles visuels",
    "receiving_blood_transfusion": "Réception de transfusion sanguine",
    "receiving_unsterile_injections": "Réception d'injections non stériles",
    "coma": "Coma",
    "stomach_bleeding": "Hémorragie de l'estomac",
    "distention_of_abdomen": "Distension de l'abdomen",
    "history_of_alcohol_consumption": "Antécédents de consommation d'alcool",
    "blood_in_sputum": "Sang dans le crachat",
    "prominent_veins_on_calf": "Veines saillantes sur la jambe",
    "palpitations": "Palpitations",
    "painful_walking" : "Marche douloureuse",
    "pus_filled_pimples" : "Pustules remplies de pus",
    "blackheads" : "Comédons noirs",
    "scurring" : "Cicatrices",
    "skin_peeling" : "Desquamation cutanée",
    "silver_like_dusting" : "Poudrage argenté",
    "small_dents_in_nails" : "Petites indentations sur les ongles",
    "inflammatory_nails" : "Ongles inflammés",
    "blister" : "Cloque",
    "red_sore_around_nose" : "Éruption rouge et douloureuse autour du nez",
    "yellow_crust_ooze" : "Croute jaune et suintante",
    "prognosis" : "Pronostic"}






try:
    # Récupération du chemin du fichier
    chemin_courant = os.path.dirname(__file__)
    chemin_fichier = os.path.join(chemin_courant, 'Training.csv')

    # Chargement des données
    df = pd.read_csv(chemin_fichier)
    df = df.drop(columns=['fluid_overload.1', 'belly_pain','Unnamed: 133'])
    df = df.rename(columns=dict_symptomes_traduction)

    # Suppression de la colonne "Pronostic" si elle existe
    X = df.drop(columns=['Pronostic'], errors='ignore')

    # ⚠ GESTION DES VALEURS MANQUANTES (NaN)
    imputer = SimpleImputer(strategy="mean")  # Remplace les NaN par la moyenne
    X_imputed = imputer.fit_transform(X)

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    #Clustering avec k=41
    k = 41
    modele = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = modele.fit_predict(X_scaled)

    print(f"Clustering terminé avec k={k} !")
    print(df["Cluster"].value_counts())

    # Trouver la maladie la plus fréquente dans chaque cluster
    mapping_clusters = df.groupby("Cluster")["Pronostic"].agg(lambda x: x.value_counts().idxmax())

    print(mapping_clusters)  # Affiche quel cluster est associé à quelle maladie
    
    #Transformer les clusters en labels de maladies prédits
    df["Cluster_Pred_Maladie"] = df["Cluster"].map(mapping_clusters)

    #Comparer avec les vraies classes
    y_true = df["Pronostic"]
    y_pred = df["Cluster_Pred_Maladie"]

    #Générer le rapport de classification
    print(classification_report(y_true, y_pred,zero_division=1))

except Exception as e:
    print("Erreur lors du clustering :", e)
    raise


# Endpoint GET pour tester l'API
@app.get("/")
def read_root():
    return {"message": "API is working"}


@app.post("/Kmeans")   
async def desease_prediction(request: Request):
    try:
        data = await request.json()
        #vecteur_Bow = pd.json_normalize(data)
        vecteur_Bow = pd.DataFrame([data])
        vecteur_Bow = vecteur_Bow.drop(columns=['Unnamed: 133'])
        vecteur_Bow.to_csv("Vecteur_to_predict.csv")
        # Normaliser les nouvelles données avec le même scaler
        vecteur_Bow = scaler.transform(vecteur_Bow)
        # Prédire le cluster
        cluster_pred = modele.predict(vecteur_Bow)
        print(f"La nouvelle observation appartient au cluster {cluster_pred[0]}")
        maladie_predite = mapping_clusters[cluster_pred]
        print(f"Le patient est classé dans le cluster {cluster_pred}, ce qui correspond à la maladie : {mapping_clusters.loc[cluster_pred[0]]}")
        print(f"prediction:{mapping_clusters.loc[cluster_pred[0]]}")
        return {mapping_clusters.loc[cluster_pred[0]]}
    except Exception as e:
        return {"error": str(e)}  
    


