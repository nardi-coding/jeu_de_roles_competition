from stlacore import *
from random import random, randrange
from copy import deepcopy
import pickle
import stlacore
from shutil import copyfile



## Quelques programmes auxiliaires nécessaires pour la sauvegarde du jeu

base_de_donnees = "C:\\Users\\eleve\\anaconda_3\\Lib\\site-packages\\stla.sqlite"
emplacement_copie_DB = "C:\\Users\\eleve\\Downloads\\IAMarcher\\ChallengeIA\\Copies\\stla.sqlite"
emplacement_permament = "C:\\Users\\eleve\\Downloads\\stla.sqlite"

def open_variable(name):
    filename = "C:\\Users\\eleve\\Downloads\\IAMarcher\\ChallengeIA\\Copies\\" + name + ".pickle"
    with open(filename, "rb") as f:
        variable = pickle.load(f)
        return variable


def save_variable(name, variable):
    filename = "C:\\Users\\eleve\\Downloads\\IAMarcher\\ChallengeIA\\Copies\\" + name + ".pickle"
    with open(filename, "wb") as f:
        pickle.dump(variable, f)
        f.close()

def val_to_list(dict):
    return list(dict.values())

def key_to_list(dict):
    return list(dict.keys())


def copy(fichier, destination):
    copyfile(fichier, destination)

##  JEU ##

#copy(emplacement_permament, base_de_donnees)

dernierHid = 0
def affiche_console(historique):
    global dernierHid
    for i in range(dernierHid, len(historique)):
        print(*historique[i][:-1])
    dernierHid = len(historique)

from itertools import permutations, product

def verifie_draft(choixJoueur, listesChoix):
    for possibilites in product(*listesChoix):
        if tuple(choixJoueur) in permutations(possibilites):
            return True
    return False

## STRATEGIE IA

import random
from NeuralNetworks import NeuralNetworks as net
from MCTS import Noeud
from shutil import copyfile
import pickle
import numpy as np

PROBABILITIES = 0
VALUE = 1

network_to_train = net(51, 75)

cupt = 1
temperature = 1
root = None
noeud = root

dicPidEmplA = {}
dicPidEmplE = {}

def draft(listesChoix):
    L = []
    for x in listesChoix:
        L.append(x[random.randint(0, 1)])
    return L




def initializer_actions():

    """ obtenir la liste de toutes les actions """
    global possible_actions
    possible_actions = {}
    s = 0
    for i in range(1, 17):
        for j in range(1, 17):
            for k in range(3):
                possible_actions[s] = (i, j, k)
                s += 1

    return possible_actions

initializer_actions()


def _pid_empl_relation(etatJeu):

    """ creer un dictionnaire qui associe le pid à l'emplacement """

    global dicPidEmpl
    for p in etatJeu.equipes[0]:
        if p != None:
            dicPidEmplA[p.pid] = etatJeu.equipes[0].index(p)

    for p in etatJeu.equipes[1]:
        if p != None:
            dicPidEmplE[p.pid] = etatJeu.equipes[1].index(p)



def _get_possible_actions_for_actual_state(etatDuJeu):
    """
    On renvoie une liste de taille 768 (la taille de toutes les actions possibles avec des 0 aux indices qui correspondent à une action qu'on ne peut pas prendre et des 1 là où on peut prendre une action

    """
    actions_filter = [0] * 75
    actions = [None] * 75
    s = 0

    for personnageA in etatDuJeu.equipes[0]:
        for personnageE in etatDuJeu.equipes[1]:
            if personnageA != None and personnageE != None:
                for capacite in range(3):
                    if personnageA.capacites[capacite].attente == 0:
                        action = (personnageA.pid, personnageE.pid, capacite)
                        actions_filter[s] = 1
                        actions[s] = action
                        s += 1

    return actions, actions_filter




########################################################



""" Programmes pour obtenir l'etat de jeu """


def _normalizer(state):
    somme = sum(state)
    state = [i / somme for i in state]
    return state


def _get_game_state_not_normalized(etatJeu):

    """ Renvoyer une liste contenant : le joueur qui doit jouer + les personnages + les caracteristiques de chaque personnage """

    state = [0] * 51
    if etatJeu.doitJouer == None:
        personnageQuiJoue = 0
    else:
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
    return state

def _get_game_state(etatJeu):
    state = _get_game_state_not_normalized(etatJeu)
    return _normalizer(state)



#####################################################################"

""" programmes qui développent l'arbre de recherche """


def _initialise_root(etatDuJeu):
    global root, noeud
    root = Noeud(None, None, etatDuJeu.doitJouer.equipe)
    _expand_noeud(root, etatDuJeu, network_to_train)
    root.Ns = 1
    noeud = root


def _expand_noeud(noeud, etatDuJeu, network):
    global possible_actions
    actions, actions_filter = _get_possible_actions_for_actual_state(etatDuJeu)
    state = _get_game_state(etatDuJeu)
    prediction = network.predict(state)
    Ps, v = prediction[PROBABILITIES], prediction[VALUE]
    for i in range(len(actions)):
        if actions[i] != None:
            noeud.ajouter_enfant(actions[i], Ps[0][i], i)
    return v[0][0]

def _faire_un_back_up(n, v):
    previous = n.action
    n = n.parent
    while n != None:
        c_a = n.edges[previous]
        n.Ns -= c_a["N_s_a"] ** (1/temperature)
        c_a["N_s_a"] += 1
        c_a["W_s_a"] += v
        c_a["Q_s_a"] =  c_a["W_s_a"]/c_a["N_s_a"]
        n.Ns += c_a["N_s_a"] ** (1/temperature)
        n.N += 1
        previous = n.action
        n = n.parent


############################


##########################################################

def lance_jeu_une_fois(etatJeu, action):
    j = etatJeu.doitJouer.equipe

    pidA, pidE, i = action[0], action[1], action[2]

    cibleAdverse, cibleAlliee = dicPidEmplE[pidE], dicPidEmplA[pidA]


    etatJeu.change_cible_adverse(coords(cibleAdverse), j)
    etatJeu.change_cible_alliee(coords(cibleAlliee), j)


    if etatJeu.doitJouer.capacites[i].attente != 0:
        i = 0

    etatJeu.applique_capacite(etatJeu.doitJouer.capacites[i], etatJeu.doitJouer)

    etatJeu.fin_de_tour()


    return etatJeu


def lance_jeu(etatInitial, done):
    if etatInitial != None:
        etatJeu = etatInitial
        _pid_empl_relation(etatJeu)
    else:
        etatJeu = EtatJeu()

        dernierPersonnage = list(curseur.execute("SELECT MAX(pid) FROM capacites WHERE EXISTS (SELECT 1 FROM occurences WHERE occurences.cid = capacites.cid)"))[0][0]

        listesChoix = [[1+randrange(dernierPersonnage) for j in range(2)] for i in range(5)]

        for j in range(2):
            try:
                choixJoueur = draft(listesChoix)
            except:
                print("Erreur à l'exécution du draft pour le joueur ", j)
                choixJoueur = [liste[0] for liste in listesChoix]

            # Draft incorrect, on remplace par le premier choix de chaque liste de choix
            if not verifie_draft(choixJoueur, listesChoix):
                print("Draft incorrect pour le joueur ", j)
                choixJoueur = [liste[0] for liste in listesChoix]
            etatJeu.equipes[j] = initialise_equipe([None, choixJoueur[1], None, choixJoueur[3], None, choixJoueur[0], None, choixJoueur[2], None, choixJoueur[4]], j)


        _pid_empl_relation(etatJeu)
        save_variable("etatJeu", etatJeu)

    memo = None
    s = 0
    while True:
        resultat = etatJeu.debut_de_tour()
        if not done:
            _initialise_root(etatJeu)
            done = True

        if resultat != PRET_AU_COMBAT:
            v = _points(etatJeu)
            etat = _get_game_state_not_normalized(etatJeu)
            noeud.enfants[(tuple(etat), memo)] = (Noeud(noeud, memo, None), etatJeu)
            _faire_un_back_up(noeud, v)
            return

        a, b, c, memo = tour_de_jeu(etatJeu, memo, etatJeu.doitJouer.equipe, s)

        s = 1
        action = a, b, c

        if memo == FIN_DU_JEU:
            return


        etatJeu = lance_jeu_une_fois(etatJeu, action)



def tour_de_jeu(etatDuJeu, memo, j, s):

    """ On applique MCTS en même temps que le jeu (c'est plus pratique)"""

    global root, network_to_train, noeud, previous_act
    global possible_actions


    etat = _get_game_state_not_normalized(etatDuJeu)

    if s != 0 :
        if (tuple(etat), previous_act) in noeud.enfants:
            noeud = noeud.enfants[(tuple(etat), previous_act)][0]
        else:
            noeud.enfants[(tuple(etat), previous_act)] = (Noeud(noeud, previous_act, j), etatDuJeu)
            noeud = noeud.enfants[(tuple(etat), previous_act)][0]

            v = _expand_noeud(noeud, etatDuJeu, network_to_train)
            if j == 0:
                _faire_un_back_up(noeud, v)
            else:
                _faire_un_back_up(noeud, -v)
            noeud = root

            return None, None, None, FIN_DU_JEU


    max_u = -float("inf")
    best_action = -1

    if s == 0:
        epsilon = 0.2
        nu = np.random.dirichlet([0.8] * len(noeud.edges))
    else:
        epsilon = 0
        nu = [0] * len(noeud.edges)

    for (action, c) in list(noeud.edges.items()):
        P_a = c["P_s_a"]; Q_s_a = c["Q_s_a"]; N_s_a = c["N_s_a"]; N = noeud.N

        idx = list(noeud.edges.keys()).index(action)
        U = cupt * ((1-epsilon) * P_a + epsilon * nu[idx]) * (N** 0.5 ) / (1+N_s_a)

        if Q_s_a + U > max_u:

            max_u = Q_s_a + U
            best_action = action

    previous_act = best_action

    return best_action[0], best_action[1], best_action[2], best_action



###########################################


training = []

def _is_game_finished(et):
    for pa in et.equipes[0]:
        for pe in et.equipes[1]:
            if pa != None and pe != None:
                return False
    return True




def _get_etatJeu_from_enfants(node, action):
    for ((s, act), enfant) in list(node.enfants.items()):
        if action == act and enfant[0].action == action:
            return s, enfant[1]

    return None, None



def _get_training_examples():
    global training, root, noeud, test
    lance_jeu(None, False)
    while True:
        for i in range(100):
            lance_jeu(open_variable("etatJeu"), True)
            print(i)




        etat = open_variable("etatJeu")
        resultat = etat.debut_de_tour()


        if resultat != PRET_AU_COMBAT or len(root.enfants)== 0:
            _set_gain(training, _points(etat))
            return training

        probabilities = _get_action_probabilities(root)
        training.append([_get_game_state(etat), probabilities, None])

        action = _choose_action_for_node(root)
        state, etatJeu = _get_etatJeu_from_enfants(root, action)




        test = root
        root = root.enfants[(tuple(state), action)][0]
        root.parent = None
        root.action = None
        noeud = root

        save_variable("etatJeu", etatJeu)




def _points(etatDuJeu):

    """ Renvoyer 1 si j'ai gagné, -1 sinon """

    for i in etatDuJeu.equipes[0]:
        if i != None:
            return 1
    return -1




def _get_action_probabilities(node):
    P = [0] * 75
    N = node.N
    if not node.est_feuille():
        for (action, c) in list(node.edges.items()):
            P[c["index"]]=c["N_s_a"] / N if N > 0 else 0
        return P
    else:
        return None


def _get_maximal_gain(L, node):
    index = L.index(max(L))
    action = possible_actions[index]
    c = node.edges[action]
    return c["Q_s_a"]



def _choose_action_for_node(node):
    P = _get_action_probabilities(node)
    if P == None:
        return P
    index = P.index(max(P))
    return node.index_to_action[index]

def _set_gain(training, v):
    for example in training:
        example[2] = v






## GENERER DES DONNEES

def generate_data(n):
    donnees = []
    for i in range(n):
        training = []
        data = _get_training_examples()
        donnees += data
        print("\n\n\n\n NEXT STEP \n\n\n\n\n")
    save_variable("donnees", donnees)




##ENTRAINER L'IA



def train(network):
    save_variable("old_network", network)
    donnees = open_variable("donnees")
    for _ in range(600):
        network.train(donnees)
    save_variable("trained_network", network)



## FAIRE COMBATRE LES IA

def tour_de_jeu_ia(etatJeu, network):
    global possible_actions
    actions, actions_filter = _get_possible_actions_for_actual_state(etatJeu)
    state = _get_game_state(etatJeu)
    prediction = network.predict(state)
    Ps, v = prediction[PROBABILITIES], prediction[VALUE]
    Ps = Ps * np.array([actions_filter])
    Ps = list(Ps[0])

    idx = Ps.index(max(Ps))

    return actions[idx]




def lance_jeu_ia(joueur1, joueur2):
    joueurs = [joueur1, joueur2]

    etatJeu = EtatJeu()

    dernierPersonnage = list(curseur.execute("SELECT MAX(pid) FROM capacites WHERE EXISTS (SELECT 1 FROM occurences WHERE occurences.cid = capacites.cid)"))[0][0]

    listesChoix = [[1+randrange(dernierPersonnage) for j in range(2)] for i in range(5)]

    for j in range(2):
        try:
            choixJoueur = draft(listesChoix)
        except:
            print("Erreur à l'exécution du draft pour le joueur ", j)
            choixJoueur = [liste[0] for liste in listesChoix]

        # Draft incorrect, on remplace par le premier choix de chaque liste de choix
        if not verifie_draft(choixJoueur, listesChoix):
            print("Draft incorrect pour le joueur ", j)
            choixJoueur = [liste[0] for liste in listesChoix]
        etatJeu.equipes[j] = initialise_equipe([None, choixJoueur[1], None, choixJoueur[3], None, choixJoueur[0], None, choixJoueur[2], None, choixJoueur[4]], j)

        _pid_empl_relation(etatJeu)

    while True:
        resultat = etatJeu.debut_de_tour()
        if resultat != PRET_AU_COMBAT:
            return etatJeu
        j = etatJeu.doitJouer.equipe
        #try:
        if True:

            pidA, pidE, i = tour_de_jeu_ia(deepcopy(etatJeu), joueurs[j])
            cibleAdverse, cibleAlliee = dicPidEmplE[pidE], dicPidEmplA[pidA]
        #except:
            #print("Erreur à l'exécution du tour de jeu du joueur", j)
            #exit()
        etatJeu.change_cible_adverse(coords(cibleAdverse), j)
        etatJeu.change_cible_alliee(coords(cibleAlliee), j)


        if etatJeu.doitJouer.capacites[i].attente != 0:
            i = 0

        etatJeu.applique_capacite(etatJeu.doitJouer.capacites[i], etatJeu.doitJouer)

        etatJeu.fin_de_tour()

""" Choisir la meilleure IA """

def choose_best_ia(old_network, trained_network):
    etatJeu = lance_jeu_ia(trained_network, old_network)
    p = _points(etatJeu)
    return int(p != - 1)


def choose():
    w = 0
    total = 0
    old = open_variable("old_network")
    new = open_variable("trained_network")
    for _ in range(20):
        b = choose_best_ia(old, new)
        w += b
        total += 1
        print(_)
    if w/total >= 0.5:
        return new
    else:
        return old
