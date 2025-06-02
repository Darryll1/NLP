import numpy as np
import json
import random
import string
import warnings
import requests
warnings.filterwarnings('ignore')
import os
import pandas as pd
import pandas as pd
import nltk
import time
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
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


# # Télécharger les ressources nécessaires si ce n'est pas déjà fait
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Récupère le chemin du répertoire courant
chemin_courant = os.path.dirname(__file__)

# Spécifie le chemin relatif vers le fichier Training.csv
chemin_fichier = os.path.join(chemin_courant, 'Training.csv')

# Lit le fichier CSV
df = pd.read_csv(chemin_fichier)

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

df = df.drop(columns=['fluid_overload.1', 'belly_pain'])
df = df.rename(columns= dict_symptomes_traduction)

def df_vector_symptome_0():
    X_Bow = df.drop("Pronostic", axis=1)
    X_Bow = pd.DataFrame(columns=X_Bow.columns)
    X_Bow.loc[0] = 0
    return X_Bow

dict_symptomes_alias = {
    "Démangeaison": ["Démangeaison","picotements", "irritation cutanée", "prurit"],
    "Eruption cutanée": ["Eruption cutanée","rougeur", "irritation cutanée", "eczéma"],
    "Éruptions cutanées nodales": ["Éruptions cutanées","Éruptions cutanées nodales","lésions cutanées", "tubérosités"],
    "Éternuements continus": ["Éternuements continus","crises d'éternuements", "éternuements répétitifs"],
    "Frisson": ["Frisson","tremblements"],
    "Frissons": ["Frissons", "tremblements", "sueurs froides"],
    "Douleur articulaire": ["Douleur articulaire","arthralgie", "douleur dans les articulations"],
    "Acidité": ["Acidité","acidité gastrique", "réflux acide"],
    "Ulcères sur la langue": ["Ulcères sur la langue","aphtes", "lésions buccales"],
    "Atrophie musculaire": ["Atrophie musculaire","perte de masse musculaire", "faiblesse musculaire"],
    "Vomissement": ["Vomissement", "nausées"],
    "Miction brûlante": ["Miction brûlante","douleur à la miction", "brûlure urinaire"],
    "Urination ponctuelle": [ "Urination ponctuelle","mictions fréquentes", "urgences urinaires"],
    "Fatigue": [ "Fatigue","épuisement", "lassitude"],
    "Gain de poids": ["Gain de poids","prise de poids", "surpoids"],
    "Anxiété": ["Anxiété","angoisse", "nervosité"],
    "Mains et pieds froids": ["Mains et pieds froids","Mains froids","pieds froids""extremités froides", "engourdissement"],
    "Changements d'humeur": ["Changements d'humeur","instabilité émotionnelle", "souplesse d'humeur"],
    "Perte de poids": ["Perte de poids","amaigrissement", "perte de poids involontaire"],
    "Agitation": ["Agitation","nervosité", "impatience"],
    "Léthargie": ["Léthargie","apathie", "lassitude"],
    "Patches dans la gorge": ["Patches dans la gorge","enrouement", "irritation de la gorge"],
    "Niveau de sucre irrégulier": ["Niveau de sucre irrégulier","dysglycémie", "instabilité glycémique"],
    "Toux": ["Toux","toux sèche", "toux productive"],
    "Fièvre élevée": ["Fièvre élevée","hyperthermie", "fièvre intense"],
    "Yeux enfoncés": [ "Yeux enfoncés","cerne sous les yeux", "yeux creux"],
    "Essoufflement": ["Essoufflement","dyspnée", "respiration difficile"],
    "Sueur": ["Sueur","transpiration", "sueur abondante"],
    "Déshydratation": ["Déshydratation","déshydratation", "soif intense"],
    "Indigestion": ["Indigestion","dyspepsie", "ballonnement"],
    "Céphalée": ["Céphalée","douleur de tête", "migraine","maux de tête"],
    "Peau jaunâtre": ["Peau jaunâtre","ictère", "coloration jaune de la peau"],
    "Urines foncées": ["Urines foncées","urines sombres", "coloration foncée des urines"],
    "Nausée": ["nausée", "mal au cœur"],
    "Perte d'appétit": ["Perte d'appétit","anorexie", "perte de l'appétit"],
    "Douleur derrière les yeux": ["Douleur derrière les yeux","douleur orbitaire", "douleur yeux"],
    "Douleur de dos": ["Douleur de dos","dorsalgie", "douleur dans le dos"],
    "Constipation": ["constipation", "difficulté à déféquer"],
    "Douleur abdominale": ["Douleur abdominale","douleur au ventre", "crampes abdominales", "maux de ventre"],
    "Diarrhée": ["diarrhée", " selles liquides"],
    "Fièvre légère": ["Fièvre","Fièvre légère","fièvre basse", "fièvre modérée"],
    "Urines jaunes": ["Urines jaunes","urines claires", "coloration jaune des urines"],
    "Ictère": ["Ictère","jaunisse", "coloration jaune des yeux et de la peau"],
    "Insuffisance hépatique aiguë": ["Insuffisance hépatique aiguë","Insuffisance hépatique","échec hépatique", "défaillance du foie"],
    "Gonflement de l'estomac": ["Gonflement de l'estomac","distension abdominale", "gonflement du ventre"],
    "Ganglions lymphatiques enflés": ["Ganglions lymphatiques enflés","Ganglions lymphatiques","adénopathie", "enflure des ganglions lymphatiques"],
    "Malaise": ["Malaise","mal-être", "sentiment de malaise"],
    "Vision floue et déformée": ["Vision floue et déformée","Vision floue","Vision déformée","trouble de la vision", "vision trouble"],
    "Crachat": ["expectoration", "crachat"],
    "Irritation de la gorge": ["enrouement", "irritation de la gorge","maux de gorge","mal de gorge"],
    "Rougeur des yeux": ["conjunctivite", "rougeur des yeux"],
    "Pression sinusoïdale": ["Pression sinusoïdale","douleur sinusoïdale", "pression dans les sinus"],
    "Écoulement nasal": ["rhinorrhée", "écoulement nasal"],
    "Congestion": ["Congestion","congestion nasale", "congestion sinusoïdale"],
    "Douleur thoracique": ["Douleur thoracique","Douleur thorax","douleur dans la poitrine", "angine de poitrine"],
    "Faiblesse dans les membres": ["Faiblesse dans les membres","faiblesse musculaire", "paralysie"],
    "Rythme cardiaque rapide": ["Rythme cardiaque rapide","Rythme rapide","coeur vite","tachycardie", "rythme cardiaque accéléré"],
    "Douleur pendant les mouvements intestinaux": ["Douleur pendant les mouvements intestinaux","Douleur intestinaux","douleur abdominale", "crampes abdominales"],
    "Douleur dans la région anale": ["Douleur dans la région anale","Douleur anale","Douleur anus","proctalgie", "douleur dans l'anus"],
    "Selles sanglantes": ["Selles sanglantes","hématochézie", "sang dans les selles"],
    "Irritation de l'anus": ["proctite", "irritation de l'anus"],
    "Douleur de cou": ["cervicalgie", "douleur dans le cou"],
    "Vertige": ["vertige", "sensation de tournoiement"],
    "Crampes": ["Crampes","crampes musculaires", "contractions musculaires"],
    "Ecchymoses": ["ecchymoses", "bleus"],
    "Obésité": ["surpoids", "obésité"],
    "Jambes enflées": ["Jambes enflées","œdème des jambes", "enflure des jambes"],
    "Vaisseaux sanguins enflés": ["Vaisseaux sanguins enflés","varices", "enflure des veines"],
    "Visage et yeux gonflés": ["Visage et yeux gonflés","Visage gonflés","yeux gonflés","œdème facial", "enflure du visage et des yeux"],
    "Thyroïde élargie": ["goitre", "thyroïde élargie"],
    "Ongles cassants": ["onychocryptose", "ongles fragiles","Ongles cassants"],
    "Extrémités enflées": ["œdème des extrémités","Extrémités enflées","enflure des mains et des pieds","enflure des mains","enflure des pieds"],
    "Faim excessive": ["hyperphagie", "faim intense","Faim excessive"],
    "Contacts extra-conjugaux": ["Contacts extra-conjugaux","relations sexuelles extra-conjugales", "infidélité","rapport sexuel non protéges","rapport sexuel sans preservatif","rapport sexuel sans capotes", "Sexe sans capote", "Sexe sans preservatif", "sans capote","sans preservatif","non protégé", "sans protection"],
    "Lèvres sèches et picotantes": ["Lèvres sèches","Lèvres picotantes","xérostomie", "sècheresse des lèvres"],
    "Parole confuse": ["Parole confuse","dysarthrie", "parole difficile à comprendre"],
    "Douleur de genou": ["gonalgie", "douleur dans le genou"],
    "Douleur de l'articulation de la hanche": ["Douleur de l'articulation de la hanche","Douleur hanche","Douleur de l'articulation","coxalgie", "douleur dans l'articulation de la hanche"],
    "Faiblesse musculaire": ["myasthénie", "faiblesse musculaire"],
    "Cou raide": ["Cou raide","torticollis", "cou rigide"],
    "Enflure des articulations": ["Enflure des articulations","Gonflement des articulations", "Raideur articulaire"],
    "Raideur des mouvements": ["Raideur des mouvements","Lourdeur des mouvements", "Difficulté à bouger"],
    "Mouvements de rotation": ["Vertige","Mouvements de rotation", "Sensation de tournoiement"],
    "Perte d'équilibre": ["Perte d'équilibre","Déséquilibre", "Instabilité"],
    "Instabilité": ["Instabilité","Manque de stabilité", "Équilibre précaire"],
    "Faiblesse d'un côté du corps": ["Faiblesse d'un côté du corps","Paralysie partielle", "Faiblesse musculaire unilatérale"],
    "Perte de l'odorat": ["Anosmie", "Perte de l'odorat"],
    "Inconfort de la vessie": ["Inconfort de la vessie","Douleur vésicale", "Irritation vésicale"],
    "Odeur nauséabonde de l'urine": ["Odeur nauséabonde de l'urine","Odeur de l'urine","Urine à l'odeur forte", "Urine nauséabonde"],
    "Sensation continue d'urine": ["Sensation continue d'urine","Urgence urinaire", "Sensation de vessie pleine"],
    "Passage de gaz": ["Passage de gaz","Flatulence", "Émission de gaz"],
    "Démangeaison interne": ["Démangeaison interne","Prurit interne", "Démangeaison viscérale"],
    "Aspect toxique (typhos)": ["Aspect toxique","Aspect maladif", "Teint maladif"],
    "Dépression": ["Dépression","Humeur dépressive", "Dépression nerveuse"],
    "Irritabilité": ["Irritabilité","Nervosité", "Irritabilité nerveuse"],
    "Douleur musculaire": ["Douleur musculaire","Myalgie", "Douleur musculaire diffuse"],
    "Sensorium altéré": ["Sensorium altéré","Troubles sensoriels", "Perte de conscience"],
    "Taches rouges sur le corps": ["Taches rouges sur le corps","Taches rouges","Taches sur le corps","Éruption cutanée", "Rougeurs cutanées"],
    "Douleur estomac": ["Douleur estomac"],
    "Ménstruation anormale": ["Ménstruation anormale","Troubles menstruels", "Ménstruation irrégulière"],
    "Patches dischromiques": ["Patches dischromiques","Taches de peau", "Pigmentation anormale"],
    "Larmes": ["Larmes","Pleurage", "Larmes abondantes"],
    "Appétit accru": ["Appétit accru","Augmentation de l'appétit", "Hyperphagie"],
    "Polyurie": ["Polyurie","Production excessive d'urine", "Diabète insipide"],
    "Antécédents familiaux": ["Antécédents familiaux","Histoire familiale", "Antécédents héréditaires"],
    "Crachat mucide": ["Crachat mucide","Expectoration mucide", "Crachat visqueux"],
    "Crachat rouillé": ["Crachat rouillé","Expectoration rouillée", "Crachat sanglant"],
    "Manque de concentration": ["Manque de concentration","Difficulté de concentration", "Troubles de l'attention"],
    "Troubles visuels": ["Troubles visuels","Perte de vision", "Troubles de la vue"],
    "Réception de transfusion sanguine": ["Réception de transfusion sanguine","Transfusion sanguine", "Greffe de sang"],
    "Réception d'injections non stériles": ["Réception d'injections non stériles","Injections non stériles", "Contamination par injection"],
    "Coma": ["Coma","Perte de conscience", "État comateux"],
    "Hémorragie de l'estomac": ["Hémorragie de l'estomac","Saignement gastrique", "Hémorragie digestive"],
    "Distension de l'abdomen": ["Distension de l'abdomen","Gonflement abdominal", "Distension abdominale"],
    "Antécédents de consommation d'alcool": ["Antécédents de consommation d'alcool","Histoire d'alcoolisme", "Consommation excessive d'alcool"],
    "Surcharge de liquides": ["Surcharge de liquides","Rétention d'eau", "Surcharge hydrique"],
    "Sang dans le crachat": ["Sang dans le crachat","Hémoptysie", "Expectoration sanglante"],
    "Veines saillantes sur la jambe": ["Veines saillantes sur la jambe","Varices", "Veines apparentes"],
    "Palpitations": ["Palpitations","Battements cardiaques irréguliers", "Palpitations cardiaques"],
    "Marche douloureuse": ["Marche douloureuse","Douleur à la marche", "Claudication"],
    "Pustules remplies de pus": ["Pustules","Pustules remplies de pus","Pustules suppurées", "Furuncles"],
    "Comédons noirs": ["Comédons noirs","Points noirs", "Comédons"],
    "Cicatrices": ["Cicatrices","Blessures cicatrisées", "Marques de cicatrisation"],
    "Desquamation cutanée": ["Desquamation cutanée","Perte de peau", "Desquamation"],
    "Poudrage argenté": ["Poudrage argenté","Poudrage métallique", "Aspect argenté"],
    "Petites indentations sur les ongles": ["Petites indentations sur les ongles","Petites dépressions sur les ongles", "Ongle enfoncé"],
    "Ongles inflammés": ["Ongles inflammés","Ongle rouge", "Inflammation de l'ongle"],
    "Cloque": ["Cloque","Vésicule", "Bulle"],
    "Éruption rouge et douloureuse autour du nez": ["Éruption rouge et douloureuse autour du nez","Éruption rouge","Éruption douloureuse autour du nez","Éruption nasale", "Rougeur nasale"],
    "Croute jaune et suintante": ["Croute jaune et suintante","Croute suintante""Croute jaune","Croute purulente", "Écoulement jaune"]}



def dict_Value_Preprocessing(dict_symptomes_alias):
 
    
    # Instancier le lemmatiseur et les mots vides
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('french'))
    
    # Fonction de traitement d'un élément de la liste
    def process_element(element):
        # Supprimer les mots vides
        words = [word for word in nltk.word_tokenize(element) if word.lower() not in stop_words]
        
        # Lemmatisation
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        
        return lemmatized_words
    
    # Traitement des valeurs du dictionnaire
    for key, values in dict_symptomes_alias.items():
        processed_values = [process_element(value) for value in values]
        dict_symptomes_alias[key] = processed_values
    
    return dict_symptomes_alias

def concatene_liste_de_liste(dict):
    return {
    cle: [' '.join([mot.lower() for mot in mots]) for mots in valeurs]
    for cle, valeurs in dict.items()
}


dict = dict_Value_Preprocessing(dict_symptomes_alias)
Alias_Prepocessed = concatene_liste_de_liste(dict)


def Input_Preprocessing(texte):
    # Séparer le texte en phrases avec comme délimiteur "."
    phrases = sent_tokenize(texte)

    # Initialiser la liste des mots traités
    mots_traites = []

    # Itérer sur les phrases
    for phrase in phrases:
        # Séparer la phrase en mots
        mots = word_tokenize(phrase)

        # Supprimer les mots vides (stopwords)
        stop_words = set(stopwords.words('french'))
        mots = [mot.lower() for mot in mots if mot.lower() not in stop_words]

        # Lemmatisation des mots
        lemmatizer = WordNetLemmatizer()
        mots = [lemmatizer.lemmatize(mot) for mot in mots]

        # Ajouter les mots traités à la liste
        mots_traites.extend(mots)       
         
    return ' '.join(mots_traites)

def Create_Vecteur_BoW (Input_Preprocessed):
    vector_Bow = df_vector_symptome_0()
    for colonne in df.columns:
        for key, values in Alias_Prepocessed.items():
            for value in values:
                if value in Input_Preprocessed:
                    vector_Bow.loc[0, key] = 1  
                    break      
    return vector_Bow 

def afficher_texte(texte):
    container = st.empty()
    texte_affiche = ""
    for lettre in texte:
        texte_affiche += str(lettre)
        container.write(texte_affiche)
        time.sleep(0.05)

def afficher_texte2(texte):
     st.write(texte)

def tokenisation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens1=[]
    tokens1+=tokenizer.tokenize(text.lower())
    return tokens1

def repondre(utilisateur,maladie):

    # Récupère le chemin du répertoire courant
    chemin_courant = os.path.dirname(__file__)

    # Spécifie le chemin relatif vers le fichier lions.txt
    chemin_fichier = os.path.join(chemin_courant, 'Maladies/'+maladie+'.txt')

    f= open(chemin_fichier,'r',errors='ignore',encoding="utf-8")
    lignes = f.read()
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
    
def saluer(phrase):
    salutations = ("salut","bonjour","hello","bonjour, comment ça va ?")
    rep_salutations = ("Salut","Bonjour","Hello","Bonjour, ça me fait plaisir de répondre à vos questions")
    for word in phrase.split():
        if word.lower() in salutations:
            return random.choice(rep_salutations)
        

def appel_api2(Json_QA):
    url = "http://127.0.0.1:8001/Q&A_Chatbot"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=Json_QA)
    # Ajout de prints pour afficher la réponse brute
    print("JSON envoyé :", Json_QA)
    print("Statut HTTP :", response.status_code)
    print("Réponse brute :", response.text)  # Voir la réponse du serveur
   # Essayer de convertir en JSON, sinon afficher une erreur
    try:
       prediction_API2 =  response.json()
       return prediction_API2.get('message')

    except requests.exceptions.JSONDecodeError:
       print("Erreur : La réponse n'est pas un JSON valide !")
       return {"error": "Réponse invalide"}   
    
def Appel_API_BD():
    url = "http://localhost:8002/creer-bdd"

    response = requests.post(url)

    if response.status_code == 200:
        print("Base de données créée avec succès !")
        print("Maladies ajoutées :", response.json()["maladies_ajoutees"])
    else:
        print("Erreur :", response.status_code)
        print(response.text)

def question_reponse_chatbot(maladie):  
    st.write(f"MediBot 🤖 : Que souhaitez-vous savoir sur {maladie} ? Si vous n'avez pas de question, tapez simplement 'Au revoir'.")
    with st.form("form_questions_reponses"):
        question_reponse = st.text_input("Entrez votre question :")
        submitted_question_reponse = st.form_submit_button("Valider la question")
        if submitted_question_reponse:
            if question_reponse.lower() in ["non", "au revoir"]:
                st.success("MediBot 🤖: 👋 **Au revoir et prenez soin de vous !**")
            else:
                Appel_API_BD()
                record = {
                    "question_reponse": question_reponse,
                    "maladie": maladie
                }
                json_record = json.dumps(record, indent=4, ensure_ascii=False)
                reponse = appel_api2(json_record)
                afficher_texte2(f"MediBot 🤖:")
                afficher_texte(reponse)
                st.session_state.question_reponse = question_reponse  # stocke la question dans session_state
                st.session_state.reponse = reponse  # stocke la réponse dans session_state
                time.sleep(3)
                st.rerun()


def appel_api(Json_Bow):
    url = "http://127.0.0.1:8000/Kmeans"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=Json_Bow)
    #Ajout de prints pour afficher la réponse brute
    print("JSON envoyé :", Json_Bow)
    print("Statut HTTP :", response.status_code)
    print("Réponse brute :", response.text)  # Voir la réponse du serveur
   # Essayer de convertir en JSON, sinon afficher une erreur
    try:
       prediction_API =  response.json()
       print(f"prediction_API1 : {prediction_API[0]}")
       return prediction_API[0]

    except requests.exceptions.JSONDecodeError:
       print("Erreur : La réponse n'est pas un JSON valide !")
       return {"error": "Réponse invalide"}    


#Création de l'interface utilisateur
def chatbot():
    if st.session_state.page_reload == 0 :
        afficher_texte("Bonjour ! Je suis MediBot 🤖, votre assistant médical virtuel. Je suis là pour écouter vos symptômes et vous aider à comprendre ce qui pourrait être la cause de votre mal-être. Dites-moi, qu'est-ce qui vous dérange ?")
        st.session_state.page_reload = 1
    else:
        afficher_texte2("Bonjour ! Je suis MediBot 🤖, votre assistant médical virtuel. Je suis là pour écouter vos symptômes et vous aider à comprendre ce qui pourrait être la cause de votre mal-être. Dites-moi, qu'est-ce qui vous dérange ?")
    #afficher_texte("MediBot: Si vous n'avez pas de question tapez alors << au revoir >>")
    with st.form("symptomes"):
        symptome = st.text_input("**Veuillez entrer vos symptômes**", st.session_state.symptome)
        submitted_symptome = st.form_submit_button("Valider les symptomes")
        if not submitted_symptome:
            st.stop()
        else :
            st.session_state.symptome = symptome
            #st.rerun()
        Input_Preprocessed = Input_Preprocessing(symptome)
        Vector_Bow = Create_Vecteur_BoW(Input_Preprocessed)
        colonnes_avec_1 = Vector_Bow.columns[Vector_Bow.isin([1]).any()]
        if(len(colonnes_avec_1) == 0):
           st.info("MediBot 🤖: Pouvez-vous réessayer ? Je n'ai pas compris vos symptômes.") 
           time.sleep(3)
           st.rerun()
           prediction_user = None
        elif(len(colonnes_avec_1) < 3):
            st.success("MediBot 🤖: Reposez-vous, ça ira peut-être mieux demain.")
            st.balloons()
            st.session_state.symptome_valide = False
            prediction_user = None
        else : 
            st.info(f"symptomes enregistré : {st.session_state.symptome}")
            with st.spinner("Analyse de vos symptomes..."):
                time.sleep(7)
            text_str = ' '.join(colonnes_avec_1)
            # Générer la wordcloud
            wordcloud = WordCloud(width=800, height=400).generate(text_str)

            # Créer une figure et un axe
            fig, ax = plt.subplots()

            # Afficher la wordcloud sur l'axe
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            time.sleep(5)
            afficher_texte("MediBot 🤖: Ce qui m'interpelle 🤔")
            # Afficher la figure avec Streamlit
            st.pyplot(fig)
            time.sleep(5)
            Json_Bow = Vector_Bow.to_json(orient='records',force_ascii=False)
            data_list = json.loads(Json_Bow)
            json_data = data_list[0]
            prediction_user = appel_api(json_data)
            print(f"prediction_API2 {prediction_user}")
            afficher_texte(f"MediBot 🤖: Aïe ! Il y a de fortes chances que vous souffrez de: {dict_maladies_traduction.get(prediction_user)}")
            st.session_state.symptome_valide = True

        
            
        if prediction_user is not None and len(prediction_user) > 0:
            return dict_maladies_traduction.get(prediction_user)
        else:
            return None
    

#question_reponse_chatbot("SIDA")
def main():
   st.title("Medibot - Votre chatbot médical")
   col1, col2, col3 = st.columns([1,2,1])
   with col2:
    st.image(Image.open("medibot.jpg"),width= 300)  # Image du chatbot
   # Initialisation des variables de session
   if "symptome" not in st.session_state:
       st.session_state.symptome = ""
   if "question_reponse" not in st.session_state:
       st.session_state.question_reponse = ""
   if "symptome_valide" not in st.session_state:
       st.session_state.symptome_valide = False
   if "maladie" not in st.session_state:
       st.session_state.maladie = None
   if "utilisateur_veut_question" not in st.session_state:
       st.session_state.utilisateur_veut_question = None  # Ne pas mettre False au départ !
   if "page_reload" not in st.session_state:
       st.session_state.page_reload = 0  
   #  Premier formulaire : Entrée des symptômes
   if not st.session_state.symptome_valide:
       maladie = chatbot()  # Affiche le premier formulaire
       if maladie is not None:
           st.session_state.maladie = maladie  # Stocker la maladie détectée
           st.session_state.symptome_valide = True  # Confirmer la validation du premier formulaire
           st.info("Vous allez etre redirigé !")
           # Créer une barre de progression
           bar = st.progress(0)
           # Boucle pour simuler une tâche longue
           for i in range(1000):
            # Mettre à jour la barre de progression
             progress = int((i + 1) / 10)
             bar.progress(progress)
             # Simuler une pause
             time.sleep(0.01)
           
           time.sleep(2)
           st.rerun()  #Recharge la page pour passer à l'étape suivante
   #Afficher le checkbox après validation des symptômes
   elif st.session_state.utilisateur_veut_question is None:
       st.write(f"MediBot 🤖: ❓ **Voulez-vous poser des questions sur {st.session_state.maladie} ?**")
       if st.button("Oui"):
           st.session_state.utilisateur_veut_question = True
           st.rerun()  # 🔄 Recharge pour afficher le deuxième formulaire
       elif st.button("Non"):
           st.session_state.utilisateur_veut_question = False
           st.rerun()  # 🔄 Recharge pour afficher "Au revoir"
   # Gérer la sélection utilisateur
   elif st.session_state.utilisateur_veut_question:
       question_reponse_chatbot(st.session_state.maladie)
   else:
       st.success("MediBot 🤖: 👋 **Au revoir et prenez soin de vous !**")
if __name__ == "__main__":
   main()


