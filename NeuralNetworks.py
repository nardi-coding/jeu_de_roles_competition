from NeuralNetwork import Network as net
class NeuralNetworks:
    def __init__(self, nr_entrees, nr_d_actions):
        self.reseau_probas = net(optimizer = "Adam")
        self.reseau_values = net(optimizer = "Adam")

        self.reseau_probas.add_layer(nr_entrees, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 32, "sigmoid")
        self.reseau_probas.add_layer(32, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 128, "sigmoid")
        self.reseau_probas.add_layer(128, nr_d_actions, "sigmoid")

        self.reseau_values.add_layer(nr_entrees, 64, "sigmoid")
        self.reseau_values.add_layer(64, 32, "sigmoid")
        self.reseau_values.add_layer(32, 1, "tanh")

    def predict(self, entree):
        vecteur_probas = self.reseau_probas.predict(entree)
        valeur_de_gain = self.reseau_values.predict(entree)
        return vecteur_probas, valeur_de_gain

    def train(self, donnees):
        X, Y_P, Y_V = donnees
        for _ in range(len(X_P)):
            self.reseau_probas.train(X[i], Y_P[i], 0.001)
            self.reseau_values.train(X[i], Y_V[i], 0.001)
