from fastapi import FastAPI
import sqlite3
import os

app3 = FastAPI()

@app3.post("/creer-bdd")
def creer_base_de_donnees():
    conn = sqlite3.connect("sante.db")
    cursor = conn.cursor()

    # Création de la table si elle n'existe pas
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maladie (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL UNIQUE,
            informations TEXT NOT NULL
        )
    ''')

    # Dossier contenant les fichiers .txt
    repertoire_maladies = "Maladies"

    maladies_ajoutees = []

    if os.path.exists(repertoire_maladies) and os.path.isdir(repertoire_maladies):
        for fichier in os.listdir(repertoire_maladies):
            chemin_fichier = os.path.join(repertoire_maladies, fichier)
            if os.path.isfile(chemin_fichier) and fichier.endswith(".txt"):
                with open(chemin_fichier, "r", encoding="utf-8") as f:
                    contenu = f.read()
                nom_maladie = os.path.splitext(fichier)[0]
                cursor.execute("SELECT id FROM maladie WHERE nom = ?", (nom_maladie,))
                if cursor.fetchone() is None:
                    cursor.execute(
                        "INSERT INTO maladie (nom, informations) VALUES (?, ?)",
                        (nom_maladie, contenu)
                    )
                    maladies_ajoutees.append(nom_maladie)

    conn.commit()
    conn.close()

    return {
        "message": "Base de données créée et mise à jour.",
        "maladies_ajoutees": maladies_ajoutees
    }
