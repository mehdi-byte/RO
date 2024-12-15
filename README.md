# Objectifs
Dans le domaine en constante évolution de la science des données et de l’apprentissage automatique, la construction d’arbres de décision optimaux représente un
défi crucial pour la classification et l’analyse prédictive. Notre projet, encadré par
Zacharie Ales pour l’année universitaire 2022-2023, se concentre sur l’application
de la modélisation F à divers jeux de données pour générer des arbres de décision
optimaux. Cette approche est explorée à travers l’utilisation de séparations univariées et multivariées ainsi que des méthodes de regroupement des données, à la fois
naïves et exactes, présentées au cours.
Le projet exploite trois jeux de données initiaux — Iris, Wine, et Seeds —
comme plateforme de test pour évaluer l’efficacité de notre modélisation. En outre,
il se propose d’étendre cette exploration à deux autres ensembles de données de
notre choix, afin de valider la robustesse et l’adaptabilité de nos méthodes. Nous
avons choisi de travailler avec ces deux jeux de données :
— Mushroom : Ce jeu de données comprend des descriptions d’échantillons
hypothétiques correspondant à 23 espèces de champignons à lamelles de la
famille Agaricus et Lepiota. Chaque espèce est identifiée comme définitivement comestible, définitivement toxique, ou d’édibilité inconnue et non
recommandée. Cette dernière classe a été combinée avec la classe toxique.
— Dry_bean_seeds : Ce jeu de données correspond a la classification de sept
types d’haricots a partir de plusieurs caracteristiques y compris 12 dimensions
et 4 formes.
Le coeur de notre investigation réside dans l’application des fonctions main()
et main-merge(), conçues pour traiter respectivement les données sans et avec
regroupement préalable, en fonction de différents paramètres de profondeur d’arbre
(D ∈ {2, 3, 4}).
À travers ce rapport, nous ambitionnons de présenter une analyse complète des
résultats obtenus en mettant l’accent sur les temps de calcul et la performance
des classifieurs en fonction des jeux de données, de la profondeur des arbres, du
type de séparations et de la stratégie de regroupement utilisée. Nous abordons
également une question d’ouverture, choisie parmi plusieurs proposées, dans le
but d’améliorer soit les temps de calcul, soit la précision des prédictions, via des méthodes innovantes de regroupement, l’utilisation d’inégalités valides ou d’autres
approches permettant d’affiner la formulation F.
Cette exploration se veut non seulement un exercice académique pour approfondir notre compréhension des arbres de décision optimaux mais aussi une
contribution modeste à la vaste quête d’efficacité dans le domaine de l’apprentissage
automatiques.
