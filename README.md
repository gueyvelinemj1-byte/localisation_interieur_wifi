# Localisation intérieure à partir des signaux Wi-Fi (RSSI)

## Présentation du projet

Ce projet porte sur la localisation en intérieur à partir de signaux Wi-Fi, en utilisant des réseaux de neurones pour estimer une position spatiale à partir des niveaux de puissance des signaux reçus (RSSI).

L'objectif est d'étudier comment les informations issues de plusieurs points d'accès Wi-Fi peuvent être exploitées pour estimer une position et d'évaluer différentes approches de modélisation.

## Démarche suivie

Le travail s'articule autour de plusieurs étapes :

1. **Prétraitement des données :** chargement des cartes de signaux RSSI et traitement des valeurs manquantes par interpolation spatiale.
2. **Construction des variables :** association des niveaux de signal Wi-Fi aux différentes positions de la grille.
3. **Normalisation :** standardisation des variables avant l'apprentissage.
4. **Modélisation :** conception et entraînement de deux réseaux de neurones pour estimer les coordonnées spatiales.
5. **Comparaison des approches :** étude de deux fonctions de perte, la MSE classique et une fonction personnalisée qui renforce la pénalisation des erreurs importantes.
6. **Évaluation :** analyse des performances à partir du coefficient de détermination (R²), de l'évolution de la fonction de perte et des erreurs de localisation.

## Résultats et analyse

Le projet a permis de mettre en œuvre et d'étudier deux approches de réseaux de neurones pour la localisation intérieure à partir des signaux Wi-Fi :

- Une première approche fondée sur la fonction de perte classique MSE.
- Une seconde approche intégrant une fonction de perte personnalisée, destinée à renforcer la pénalisation des erreurs importantes.

L'évaluation repose notamment sur le coefficient de détermination (R²), l'évolution de la fonction de perte et l'analyse des erreurs de localisation.

Ce travail permet d'étudier l'influence du choix de la fonction de perte sur un problème de régression spatiale et de comparer différentes stratégies d'apprentissage.

## Environnement de développement

- **Langage :** Python
- **Manipulation des données :** NumPy, Pandas
- **Interpolation et calcul scientifique :** SciPy
- **Prétraitement et évaluation :** Scikit-learn
- **Réseaux de neurones :** TensorFlow / Keras
- **Visualisation :** Matplotlib

## Données

Les données utilisées sont issues de fichiers CSV fournis dans le cadre du cours. Elles représentent les niveaux de puissance des signaux Wi-Fi sur différentes zones de l'environnement étudié.

Les fichiers de données ne sont pas inclus dans ce dépôt.

## Cadre du projet

Travail pratique réalisé dans le cadre du Master Mathématiques appliquées et Statistique, parcours Science des données, à l'Université Claude Bernard Lyon 1.