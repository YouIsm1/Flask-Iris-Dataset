# from flask import Flask, render_template, request
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# import numpy as np

# # Création de l'application Flask
# app = Flask(__name__)

# # Chargement des données iris
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Division des données en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Modèles d'arbre de décision et de réseaux neuronaux
# clf_decision_tree = DecisionTreeClassifier()
# clf_neural_network = MLPClassifier()

# # Route pour la page d'accueil
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         # Obtenir l'algorithme sélectionné par l'utilisateur
#         selected_algorithm = request.form.get('algorithm')
#         if selected_algorithm == 'decision_tree':
#             # Entraîner le modèle d'arbre de décision
#             clf_decision_tree.fit(X_train, y_train)
#         elif selected_algorithm == 'neural_network':
#             # Entraîner le modèle de réseaux neuronaux
#             clf_neural_network.fit(X_train, y_train)
#         return render_template('predict.html')
#     return render_template('index.html')

# # Route pour afficher les résultats des prédictions
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if request.method == 'POST':
# #         # Obtenir les données saisies par l'utilisateur depuis le formulaire
# #         data = [float(x) for x in request.form.values()]
# #         data = np.array(data)
# #         # Faire une prédiction avec le modèle sélectionné
# #         selected_algorithm = request.form.get('algorithm')
# #         if selected_algorithm == 'decision_tree':
# #             prediction = clf_decision_tree.predict([data])
# #         elif selected_algorithm == 'neural_network':
# #             prediction = clf_neural_network.predict([data])
# #         # Convertir la sortie en nom de la classe
# #         predicted_class = iris.target_names[prediction[0]]
# #         return render_template('result.html', predicted_class=predicted_class)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Obtenir les données saisies par l'utilisateur depuis le formulaire
#         data = [float(x) for x in request.form.values()]
#         data = np.array(data)
#         # Initialiser la variable prediction
#         prediction = None
#         # Faire une prédiction avec le modèle sélectionné s'il est entraîné
#         selected_algorithm = request.form.get('algorithm')
#         if selected_algorithm == 'decision_tree' and clf_decision_tree:
#             prediction = clf_decision_tree.predict([data])
#         elif selected_algorithm == 'neural_network' and clf_neural_network:
#             prediction = clf_neural_network.predict([data])
#         # Si aucune prédiction n'a été faite, renvoyer un message d'erreur
#         if prediction is None:
#             return "Error: The selected algorithm is not trained yet. Please train the model first."
#         # Convertir la sortie en nom de la classe
#         predicted_class = iris.target_names[prediction[0]]
#         return render_template('result.html', predicted_class=predicted_class)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Création de l'application Flask
app = Flask(__name__)

# Chargement des données iris
iris = load_iris()
X = iris.data
y = iris.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèles d'arbre de décision et de réseaux neuronaux
clf_decision_tree = DecisionTreeClassifier()
clf_neural_network = MLPClassifier()

# Variable pour stocker l'exactitude du modèle
accuracy_decision_tree = None
accuracy_neural_network = None

# Route pour la page d'accueil
@app.route('/', methods=['GET', 'POST'])
def home():
    global accuracy_decision_tree, accuracy_neural_network
    if request.method == 'POST':
        # Obtenir l'algorithme sélectionné par l'utilisateur
        selected_algorithm = request.form.get('algorithm')
        if selected_algorithm == 'decision_tree':
            # Entraîner le modèle d'arbre de décision
            clf_decision_tree.fit(X_train, y_train)
            # Calculer l'exactitude du modèle d'arbre de décision
            accuracy_decision_tree = accuracy_score(y_test, clf_decision_tree.predict(X_test))
        elif selected_algorithm == 'neural_network':
            # Entraîner le modèle de réseaux neuronaux
            clf_neural_network.fit(X_train, y_train)
            # Calculer l'exactitude du modèle de réseaux neuronaux
            accuracy_neural_network = accuracy_score(y_test, clf_neural_network.predict(X_test))
        return render_template('predict.html', accuracy_decision_tree=accuracy_decision_tree, accuracy_neural_network=accuracy_neural_network, selected_algorithm= selected_algorithm)
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtenir les données saisies par l'utilisateur depuis le formulaire
        feature1 = request.form.get("feature1")
        feature2 = request.form.get("feature2")
        feature3 = request.form.get("feature3")
        feature4 = request.form.get("feature4")
        data = [feature1, feature2, feature3, feature4]
        data = [float(i) for i in data] 
        data = np.array(data)
        print(data)
        # Initialiser la variable prediction
        prediction = None
        # Faire une prédiction avec le modèle sélectionné s'il est entraîné
        selected_algorithm = request.form.get('selected_algorithm')
        print(selected_algorithm)
        if selected_algorithm == 'decision_tree' and clf_decision_tree:
            prediction = clf_decision_tree.predict([data])
        elif selected_algorithm == 'neural_network' and clf_neural_network:
            prediction = clf_neural_network.predict([data])
        # Si aucune prédiction n'a été faite, renvoyer un message d'erreur
        if prediction is None:
            return "Error: The selected algorithm is not trained yet. Please train the model first."
        # Convertir la sortie en nom de la classe
        predicted_class = iris.target_names[prediction[0]]
        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
