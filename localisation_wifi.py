#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projet académique : localisation intérieure à partir des signaux Wi-Fi (RSSI).

Script Python issu du TP réalisé individuellement avec l'accompagnement du professeur.
Placer les fichiers RSSI_0.csv à RSSI_3.csv dans le même dossier que ce script.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Chargement des fichiers CSV
fichiers_rssi = ['RSSI_0.csv', 'RSSI_1.csv', 'RSSI_2.csv', 'RSSI_3.csv']
cartes_rssi = [pd.read_csv(fichier, header=None) for fichier in fichiers_rssi]

# Fonction d’interpolation spatiale
def interpoler_cartes_rssi(carte_rssi):
    lignes_connues, colonnes_connues = np.where(~carte_rssi.isna())
    valeurs_connues = carte_rssi.values[lignes_connues, colonnes_connues]

    lignes_manquantes, colonnes_manquantes = np.where(carte_rssi.isna())
    valeurs_interpolees = griddata(
        points=(lignes_connues, colonnes_connues), values=valeurs_connues,
        xi=(lignes_manquantes, colonnes_manquantes), method='linear'
    )

    carte_interpolee = carte_rssi.copy()
    carte_interpolee.values[lignes_manquantes, colonnes_manquantes] = valeurs_interpolees

    # Complément des valeurs manquantes restantes
    if np.isnan(carte_interpolee).sum().sum() > 0:
        valeurs_interpolees_proches = griddata(
            points=(lignes_connues, colonnes_connues), values=valeurs_connues,
            xi=(lignes_manquantes, colonnes_manquantes), method='nearest'
        )
        carte_interpolee.values[lignes_manquantes, colonnes_manquantes] = valeurs_interpolees_proches

    return carte_interpolee

# Interpolation des cartes RSSI
cartes_rssi_interpolees = [interpoler_cartes_rssi(carte_rssi) for carte_rssi in cartes_rssi]

# Visualisation des cartes RSSI interpolées
figure, tableaux_plt = plt.subplots(2, 2, figsize=(12, 10))
for indice_ligne, graphique in enumerate(tableaux_plt.flat):
    image = graphique.imshow(cartes_rssi_interpolees[indice_ligne], cmap='jet', origin='lower')
    graphique.set_title(f'Carte RSSI interpolée {indice_ligne}')
    figure.colorbar(image, ax=graphique)
plt.tight_layout()
plt.show()

# Préparation des données
donnees_entree, coordonnees_reelles = [], []
nb_lignes, nb_colonnes = cartes_rssi_interpolees[0].shape

for ligne in range(nb_lignes):
    for colonne in range(nb_colonnes):
        signaux_wifi = [cartes_rssi_interpolees[indice_ligne].iloc[ligne, colonne] for indice_ligne in range(4)]
        donnees_entree.append(signaux_wifi)
        coordonnees_reelles.append([ligne, colonne])

donnees_entree, coordonnees_reelles = np.array(donnees_entree), np.array(coordonnees_reelles)
coordonnees_reelles = coordonnees_reelles.astype('float32')  # Conversion des coordonnées en nombres décimaux

donnees_entree = (donnees_entree - donnees_entree.mean(axis=0)) / donnees_entree.std(axis=0)
# Séparation des données d’entraînement et de test
entrees_entrainement, entrees_test, coordonnees_entrainement, coordonnees_test = train_test_split(donnees_entree, coordonnees_reelles, test_size=0.3, random_state=42)

# Définition du réseau de neurones
modele_1 = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2)
])

modele_1.compile(optimizer=Adam(0.001), loss='mse', metrics=[tf.keras.metrics.R2Score])

# Entraînement du modèle
historique = modele_1.fit(entrees_entrainement, coordonnees_entrainement, epochs=5000, batch_size=2641, validation_data=(entrees_test, coordonnees_test), verbose=1)

print(historique.history.keys())
figure=plt.figure(figsize=(15, 6))
# Évolution du score R²
figure.add_subplot(1,2,1)
plt.plot(historique.history['r2_score'])
plt.plot(historique.history['val_r2_score'])
plt.title('Score R² du modèle')
plt.ylabel('Score R²')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Test'], loc='upper left')
# Évolution de la fonction de perte
figure.add_subplot(1,2, 2)
plt.plot(historique.history['loss'])
plt.plot(historique.history['val_loss'])
plt.title('Erreur du modèle')
plt.ylabel('Erreur')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Test'], loc='upper left')
plt.show()

# Prédictions et calcul des erreurs
coordonnees_predites = modele_1.predict(donnees_entree)
distances_erreur = np.linalg.norm(coordonnees_predites - coordonnees_reelles, axis=1)

# Distribution des erreurs de distance
plt.hist(distances_erreur, bins=30)
plt.title('Histogramme des erreurs de distance')
plt.xlabel('Distance à la position réelle (m)')
plt.ylabel('Nombre de positions')
plt.show()

# Analyse des seuils de précision
percentiles_etudies = [80, 90, 95, 99]
seuils_erreur = np.percentile(distances_erreur, percentiles_etudies)
for pourcentage, seuil_erreur in zip(percentiles_etudies, seuils_erreur):
    print(f"Pour {pourcentage}% des positions, l’écart est inférieur ou égal à {seuil_erreur:.2f} m")

# Prédictions sur l’ensemble des positions
predictions_toutes_positions = modele_1.predict(donnees_entree)
erreurs_toutes_positions = np.linalg.norm(predictions_toutes_positions - coordonnees_reelles, axis=1)

# Réorganisation des erreurs selon la grille d’origine
carte_des_erreurs = erreurs_toutes_positions.reshape(nb_lignes, nb_colonnes)

# Visualisation spatiale des erreurs de localisation
plt.figure(figsize=(10, 8))
image = plt.imshow(carte_des_erreurs, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(image, label='Distance à la position réelle (m)')
plt.title('Carte des écarts de localisation')
plt.xlabel('Coordonnée horizontale')
plt.ylabel('Coordonnée verticale')

# Affichage des erreurs à intervalles réguliers
for indice_ligne in range(0, nb_lignes, 5):
    for indice_colonne in range(0, nb_colonnes, 5):
        plt.text(indice_colonne, indice_ligne, f'{carte_des_erreurs[indice_ligne, indice_colonne]:.1f}', 
                 ha='center', va='center', 
                 color='w' if carte_des_erreurs[indice_ligne, indice_colonne] > np.median(carte_des_erreurs) else 'k')

plt.show()

# Définition du réseau de neurones
modele_2 = Sequential([
    Dense(64,activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2)
])

def erreur_moyenne_ponderee(coordonnees_attendues, coordonnees_predites):
    difference_coordonnees = coordonnees_attendues - coordonnees_predites
    erreur_carree_moyenne = tf.reduce_mean(tf.square(difference_coordonnees), axis=-1)
    coefficient_penalite = tf.where(erreur_carree_moyenne > 5.0, 5.0, 1.0)  # Pénalité multipliée par cinq pour les erreurs supérieures à 5 m
    return tf.reduce_mean(coefficient_penalite * erreur_carree_moyenne)

modele_2.compile(optimizer=Adam(0.005), loss=erreur_moyenne_ponderee, metrics=[tf.keras.metrics.R2Score])

# Entraînement du modèle
historique = modele_2.fit(entrees_entrainement, coordonnees_entrainement, epochs=400, validation_data=(entrees_test, coordonnees_test), verbose=1)

print(historique.history.keys())
figure=plt.figure(figsize=(15, 6))
# Évolution du score R²
figure.add_subplot(1,2,1)
plt.plot(historique.history['r2_score'])
plt.plot(historique.history['val_r2_score'])
plt.title('Score R² du modèle')
plt.ylabel('Score R²')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Test'], loc='upper left')
# Évolution de la fonction de perte
figure.add_subplot(1,2, 2)
plt.plot(historique.history['loss'])
plt.plot(historique.history['val_loss'])
plt.title('Erreur du modèle')
plt.ylabel('Erreur')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Test'], loc='upper left')
plt.show()

# Prédictions et calcul des erreurs
coordonnees_predites = modele_2.predict(donnees_entree)
distances_erreur = np.linalg.norm(coordonnees_predites - coordonnees_reelles, axis=1)

# Distribution des erreurs de distance
plt.hist(distances_erreur, bins=30)
plt.title('Histogramme des erreurs de distance')
plt.xlabel('Distance à la position réelle (m)')
plt.ylabel('Nombre de positions')
plt.show()

# Analyse des seuils de précision
percentiles_etudies = [80, 90, 95, 99]
seuils_erreur = np.percentile(distances_erreur, percentiles_etudies)
for pourcentage, seuil_erreur in zip(percentiles_etudies, seuils_erreur):
    print(f"Pour {pourcentage}% des positions, l’écart est inférieur ou égal à {seuil_erreur:.2f} m")

# Prédictions sur l’ensemble des positions
predictions_toutes_positions = modele_2.predict(donnees_entree)
erreurs_toutes_positions = np.linalg.norm(predictions_toutes_positions - coordonnees_reelles, axis=1)

# Réorganisation des erreurs selon la grille d’origine
carte_des_erreurs = erreurs_toutes_positions.reshape(nb_lignes, nb_colonnes)

# Visualisation spatiale des erreurs de localisation
plt.figure(figsize=(10, 8))
image = plt.imshow(carte_des_erreurs, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(image, label='Distance à la position réelle (m)')
plt.title('Carte des écarts de localisation')
plt.xlabel('Coordonnée horizontale')
plt.ylabel('Coordonnée verticale')

# Affichage des erreurs à intervalles réguliers
for indice_ligne in range(0, nb_lignes, 5):
    for indice_colonne in range(0, nb_colonnes, 5):
        plt.text(indice_colonne, indice_ligne, f'{carte_des_erreurs[indice_ligne, indice_colonne]:.1f}', 
                 ha='center', va='center', 
                 color='w' if carte_des_erreurs[indice_ligne, indice_colonne] > np.median(carte_des_erreurs) else 'k')

plt.show()

def augmenter_donnees(donnees, colonne, niveau_bruit=0.05, nombre_augmentations=5):
    entrees_augmentees, coordonnees_augmentees = [], []
    for _ in range(nombre_augmentations):
        # Génération d’un bruit gaussien
        bruit_aleatoire = np.random.normal(0, niveau_bruit, donnees.shape)
        entrees_augmentees.append(donnees + bruit_aleatoire)
        coordonnees_augmentees.append(colonne)
    return np.vstack(entrees_augmentees), np.vstack(coordonnees_augmentees)

# Augmentation des données avant la séparation entraînement-test
entrees_augmentees, coordonnees_augmentees = augmenter_donnees(donnees_entree, coordonnees_reelles)
entrees_entrainement, entrees_test, coordonnees_entrainement, coordonnees_test = train_test_split(entrees_augmentees, coordonnees_augmentees, test_size=0.3)

# Définition du réseau de neurones
modele_3 = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2)
])

modele_3.compile(optimizer=Adam(0.005), loss='mse',metrics=[tf.keras.metrics.R2Score])

# Entraînement du modèle
historique = modele_3.fit(entrees_entrainement, coordonnees_entrainement, epochs=500, validation_data=(entrees_test, coordonnees_test), verbose=1)

# Prédictions et calcul des erreurs
coordonnees_predites = modele_3.predict(donnees_entree)
distances_erreur = np.linalg.norm(coordonnees_predites - coordonnees_reelles, axis=1)

# Distribution des erreurs de distance
plt.hist(distances_erreur, bins=30)
plt.title('Histogramme des erreurs de distance')
plt.xlabel('Distance à la position réelle (m)')
plt.ylabel('Nombre de positions')
plt.show()

# Analyse des seuils de précision
percentiles_etudies = [80, 90, 95, 99]
seuils_erreur = np.percentile(distances_erreur, percentiles_etudies)
for pourcentage, seuil_erreur in zip(percentiles_etudies, seuils_erreur):
    print(f"Pour {pourcentage}% des positions, l’écart est inférieur ou égal à {seuil_erreur:.2f} m")

# Prédictions sur l’ensemble des positions
predictions_toutes_positions = modele_3.predict(donnees_entree)
erreurs_toutes_positions = np.linalg.norm(predictions_toutes_positions - coordonnees_reelles, axis=1)

# Réorganisation des erreurs selon la grille d’origine
carte_des_erreurs = erreurs_toutes_positions.reshape(nb_lignes, nb_colonnes)

# Visualisation spatiale des erreurs de localisation
plt.figure(figsize=(10, 8))
image = plt.imshow(carte_des_erreurs, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(image, label='Distance à la position réelle (m)')
plt.title('Carte des écarts de localisation')
plt.xlabel('Coordonnée horizontale')
plt.ylabel('Coordonnée verticale')

# Affichage des erreurs à intervalles réguliers
for indice_ligne in range(0, nb_lignes, 5):
    for indice_colonne in range(0, nb_colonnes, 5):
        plt.text(indice_colonne, indice_ligne, f'{carte_des_erreurs[indice_ligne, indice_colonne]:.1f}', 
                 ha='center', va='center', 
                 color='w' if carte_des_erreurs[indice_ligne, indice_colonne] > np.median(carte_des_erreurs) else 'k')

plt.show()

def augmenter_donnees(donnees, colonne, niveau_bruit=0.05, nombre_augmentations=5):
    entrees_augmentees, coordonnees_augmentees = [], []
    for _ in range(nombre_augmentations):
        # Génération d’un bruit gaussien
        bruit_aleatoire = np.random.normal(0, niveau_bruit, donnees.shape)
        entrees_augmentees.append(donnees + bruit_aleatoire)
        coordonnees_augmentees.append(colonne)
    return np.vstack(entrees_augmentees), np.vstack(coordonnees_augmentees)

# Augmentation des données avant la séparation entraînement-test
entrees_augmentees, coordonnees_augmentees = augmenter_donnees(donnees_entree, coordonnees_reelles)
entrees_entrainement, entrees_test, coordonnees_entrainement, coordonnees_test = train_test_split(entrees_augmentees, coordonnees_augmentees, test_size=0.3)

# Définition du réseau de neurones
modele_4 = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2)
])

def erreur_moyenne_ponderee(coordonnees_attendues, coordonnees_predites):
    difference_coordonnees = coordonnees_attendues - coordonnees_predites
    erreur_carree_moyenne = tf.reduce_mean(tf.square(difference_coordonnees), axis=-1)
    coefficient_penalite = tf.where(erreur_carree_moyenne > 5.0, 5.0, 1.0)  # Pénalité multipliée par cinq pour les erreurs supérieures à 5 m
    return tf.reduce_mean(coefficient_penalite * erreur_carree_moyenne)
    
modele_4.compile(optimizer=Adam(0.005), loss=erreur_moyenne_ponderee ,metrics=[tf.keras.metrics.R2Score])

# Entraînement du modèle
historique = modele_4.fit(entrees_entrainement, coordonnees_entrainement, epochs=500, validation_data=(entrees_test, coordonnees_test), verbose=1)

# Prédictions et calcul des erreurs
coordonnees_predites = modele_4.predict(donnees_entree)
distances_erreur = np.linalg.norm(coordonnees_predites - coordonnees_reelles, axis=1)

# Distribution des erreurs de distance
plt.hist(distances_erreur, bins=30)
plt.title('Histogramme des erreurs de distance')
plt.xlabel('Distance à la position réelle (m)')
plt.ylabel('Nombre de positions')
plt.show()

# Analyse des seuils de précision
percentiles_etudies = [80, 90, 95, 99]
seuils_erreur = np.percentile(distances_erreur, percentiles_etudies)
for pourcentage, seuil_erreur in zip(percentiles_etudies, seuils_erreur):
    print(f"Pour {pourcentage}% des positions, l’écart est inférieur ou égal à {seuil_erreur:.2f} m")

# Prédictions sur l’ensemble des positions
predictions_toutes_positions = modele_4.predict(donnees_entree)
erreurs_toutes_positions = np.linalg.norm(predictions_toutes_positions - coordonnees_reelles, axis=1)

# Réorganisation des erreurs selon la grille d’origine
carte_des_erreurs = erreurs_toutes_positions.reshape(nb_lignes, nb_colonnes)

# Visualisation spatiale des erreurs de localisation
plt.figure(figsize=(10, 8))
image = plt.imshow(carte_des_erreurs, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(image, label='Distance à la position réelle (m)')
plt.title('Carte des écarts de localisation')
plt.xlabel('Coordonnée horizontale')
plt.ylabel('Coordonnée verticale')

# Affichage des erreurs à intervalles réguliers
for indice_ligne in range(0, nb_lignes, 5):
    for indice_colonne in range(0, nb_colonnes, 5):
        plt.text(indice_colonne, indice_ligne, f'{carte_des_erreurs[indice_ligne, indice_colonne]:.1f}', 
                 ha='center', va='center', 
                 color='w' if carte_des_erreurs[indice_ligne, indice_colonne] > np.median(carte_des_erreurs) else 'k')

plt.show()