import numpy as np
from NeuralNetwork import Network as net



PROBABILITIES = 0; VALUE = 1


class Noeud:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        self.Ns = 0
        self.jeJoue = False
        self.edges = {}

    def est_feuille(self):
        return len(self.edges) == 0

    def ajouter_enfant(self, edge_action, P):
        self.edges[edge_action] = {
                                    "N_s_a": 0,
                                    "W_s_a": 0,
                                    "Q_s_a": 0,
                                    "P_s_a": P
                                }


class MCTS:
    def __init__(self, state, network, jeu, cupt = 1, temperature = 1):
        self.root = Noeud(state, None, None)
        self._expand_noeud(self.root)
        self.arbre = {}
        self.network = network
        self.cupt = cupt
        self.jeu = jeu
        self.temperature = temperature

    def get_action_probability(self):
        """
        Renvoyer le vecteur contenant les probabilités pour chaque action pour l'état "root"

        """

        vecteur_probabilite = [0 for _ in range(len(possible_actions))]
        noeud = self.root
        for (action, c_a) in noeud.edges.items():
            index = list(possible_actions.keys())[list(possible_actions.values()).index(action)]
            vecteur_probabilite[index - 1] = c_a["N_s_a"] / noeud.Ns
        return vecteur_probabilite

    def _expand_noeud(self, noeud):
        actions = _get_possible_actions_for_actual_state(state)
        prediction = self.network.predict(state)
        Ps, v = prediction[PROBABILITIES], prediction[VALUE]
        for i in range(len(actions)):
            if actions[i]:
                noeud.ajouter_enfant(possible_actions[i + 1], Ps[0][i])
        return v

    def search_and_backup(self, st):

        """ Etape de recherche. On descend l'arbre de MCTS à l'aide du réseau de neurones. """

        state = st
        if self.root.est_feuille():
            v = self._expand_noeud(self.root)

        noeud = self.root
        max_u = -float("inf")
        best_action = -1

        """ On arrête la recherche dès qu'on trouve une feuille """

        while not noeud.est_feuille():
            for (action, characteristics) in noeud.edges.items():
                P_a = characteristics["P_s_a"]; Q_a = characteristics["Q_s_a"]; N_a = characteristics["N_s_a"]; N = noeud.Ns

                U = self.cupt * P_a * ( N** 0.5 ) / N_a

                if Q + U > max_u:
                    max_u = Q + U
                    best_action = action

            older_state = state
            state = self.jeu(best_action)
            if (older_state, best_action) in self.arbre:
                noeud = self.arbre[(older_state, best_action)]
            else:
                noeud = Noeud(state, noeud, best_action)
                self.arbre[(older_state, best_action)] = noeud

        """ On vérifie d'abord si le jeu est fini """

        game_finished = _is_game_finished()

        """
        Si le jeu n'est pas fini, on développe le noeud feuille et on remonte l'arbre, sinon on obtient la valeur à la fin du jeu et on remonte l'arbre

        """
        if noeud.est_feuille() and not game_finished:
            v = self._expand_noeud(noeud)

        if game_finished:
            v = _points()

        while noeud != None:
            previous_action = noeud.action
            noeud = noeud.parent
            c_a = noeud.edges[previous_action]
            noeud.Ns -= c_a["N_s_a"] ** (1/self.temperature)
            c_a["N_s_a"] += 1
            c_a["W_s_a"] += v
            c_a["Q_s_a"] +=  c_a["W_s_a"]/c_a["N_s_a"]
            noeud.Ns += c_a["N_s_a"] ** (1/self.temperature)