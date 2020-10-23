"""
@author: N.X.

"""


from NeuralNetworks import NeuralNetworks as net
from MCTS import MCTS, Noeud


network_to_train = net(51, 768)

def initializer_actions():
    global possible_actions
    possible_actions = {}
    s = 0
    for i in range(1, 17):
        for j in range(1, 17):
            for k in range(3):
                s += 1
                possible_actions[s] = (i, j, k)



def _get_possible_actions_for_actual_state(etatDuJeu):
    """
    Accès à la base de données pour obtenir l'état du jeu et les actions possible à jouer.
    On renvoie une liste de taille 300 (la taille de toutes les actions possibles avec des 0 aux indices qui correspondent à une action qu'on ne peut pas prendre et des 1 là où on peut prendre une action

    """
    actions = [0] * 768

    for personnageA in etatDuJeu.equipes[0]:
        for personnageE in etatDuJeu.equipes[1]:
            if personnageA != None and personnageE != None:
                for capacite in range(3):
                    if personnageA.capacites[capacite].attente == 0:
                        index = list(possible_actions.keys())[list(possible_actions.values()).index((personnageA.pid, personnageE.pid, capacite))]
                        actions[index - 1] = 1

    return actions

def _get_game_state(etatJeu):
    """ Renvoyer une liste contenant : le joueur qui doit jouer + les personnages + les caracteristiques de chaque personnage """
    state = [0] * 51
    personnageQuiJoue = etatJeu.doitJouer.pid
    state[0] = personnageQuiJoue
    s = 1
    L = [1, 3, 5, 7, 9]

    A = etatJeu.equipes[0]; E = etatJeu.equipes[1]
    for personnage in A:
        if personnage != None:
            state[  s  ] = personnage.pid
            state[s + 1] = personnage.vie
            state[s + 2] = personnage.force
            state[s + 3] = personnage.vitesse
            state[s + 4] = personnage.esquive
        if A.index(personnage) in L:
            s += 5

    for personnage in E:
        if personnage != None:
            state[  s  ] = personnage.pid
            state[s + 1] = personnage.vie
            state[s + 2] = personnage.force
            state[s + 3] = personnage.vitesse
            state[s + 4] = personnage.esquive
        if E.index(personnage) in L:
            s += 5


    """ normalizer les données """

    somme = sum(state)
    state = [i / somme for i in state]
    return state


def _is_game_finished():
    return statut == EJ_VICTOIRE or statut == EJ_DEFAITE

def _points():
    """
    Renvoyer 1 si j'ai gagné, -1 sinon

    """
    if statut == EJ_VICTOIRE:
        return 1
    elif statut == EJ_DEFAITE:
        return -1



def _jouer_match(old_network, new_network):
    """ renvoyer 1 si new_network gagne, 0 sinon """
    lance_jeu(new_network)
    while not _is_game_finished():
        pass
    if statut == EJ_VICTOIRE:
        return 1
    elif statut == EJ_DEFAITE:
        return 0


def _evaluate_network(old_network, new_network, number_of_plays):

    """
    Jouer plusieurs parties de jeu entre les deux réseaux et choisir celui qui gagne plus que 55% des matchs
    """
    s = 0
    for _ in range(number_of_plays):
        s += _jouer_match(old_network, new_network)
    if s / number_of_plays >= 0.55:
        return new_network
    else:
        return old_network


def _lancer_le_jeu(etat, action):

    """ lancer le jeu à partir d'un état donné et appliquer l'action donnée en argument """

    pass
def _jouer_a_partir_d_un_etat(etat, action):
    lancer_le_jeu(etat, possible_actions[action])
    return etatJeu

def _affecter_la_valeur_de_gain(gain, episode_jeu):
    for episode in episode_jeu:
        episode[2] = gain


def _obtenir_des_donnes_pour_entrainement(neural_network):

    """ Ici on applique Monte Carlo Tree Search pour produire des données du jeu et on renvoie la liste avec des données """

    lance_jeu(neural_network)
    state = _get_game_state(etatJeu)
    mcts = MCTS(state, neural_network)
    donnees = []
    while True:

        for i in range(500):
            mcts.search_and_backup(state)
        vecteur_probabilite = mcts.get_action_probability()
        donnees.append([state, vecteur_probabilite, None])
        action = vecteur_probabilite.index(max(vecteur_probabilite)) + 1
        etatJeu = _jouer_a_partir_d_un_etat(state, action)
        mcts.root = mcts.arbre[(state, action)]
        state = _get_game_state(etatJeu)

        if _is_game_finished():
            gain = _points()
            _affecter_la_valeur_de_gain(gain, donnees)
            return donnees



def _entrainer_le_reseau(donnees, reseau):
    """ On entraine le réseau """
    epochs = 500
    copie_reseau = reseau
    for _ in range(500):
        reseau.train(donnees)

    meilleur_reseau = _evaluate_network(copie_reseau, reseau)

    return meilleur_reseau


def obtenir_un_bon_reseau(reseau):
    """ On renvoie une meilleure version pour le réseau donnée en entrée """

    donnees = _obtenir_des_donnes_pour_entrainement(reseau)
    meilleur_reseau = _entrainer_le_reseau(donnees, reseau)
    return meilleur_reseau
