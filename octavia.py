import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import sys

# Lire les données
df = pd.read_csv("data.csv")

# Mapping des valeurs pour Nationality et Go
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

# Définir les features et la cible
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

# Créer le modèle de l'arbre de décision
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Plot de l'arbre de décision
plt.figure(figsize=(12, 8))  # Ajuster la taille de l'image
tree.plot_tree(dtree, feature_names=features, filled=True)

# Sauvegarder l'image dans un fichier PNG
plt.savefig('decision_tree.png')

# Si tu veux simplement afficher l'image dans un environnement graphique
# plt.show()
