# Importation des modules nécessaires
import numpy as np
import random
import gym
import time
from matplotlib import pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

env = gym.make("BipedalWalker-v3")

# Permet de récupérer le nombre d'actions possible pour la voiture
nb_actions = env.action_space.shape[0]

# Permet de récupérer les dimensions de l'espace d'observation (x,y)
dim_observations = env.observation_space.shape[0]

print('Nombre d\'actions possibles : {}'.format(nb_actions))
print('Dimension des observations : {}'.format(dim_observations))

"""
def Random_games():

    for episode in range(2):

        print("Episode : {}".format(episode + 1))

        env.reset()
        timer_init = time.time()
        # this is each frame, up to 500...but we wont make it that far with random.
        while True:
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be any of one how in list on 4, for example [0 1 0 0]
            action = np.random.uniform(-1.0, 1.0, size = nb_actions)

            timer = time.time() - timer_init

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            next_state, reward, done, info = env.step(action)

            # lets print everything in one line:
            #print(reward, action)
            if done or timer > 5:
                break

    env.close()
"""

with tf.device("cpu:0"):
   print("tf.keras code in this scope will run on CPU")

# Hyperparamètres
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.999
learning_rate = 0.0025
discount_rate = 0.95
memory = deque(maxlen = 1000000)
batch_size = 48

# Réseaux de neuronnes
model = Sequential()
model.add(Dense(256, input_dim = dim_observations))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(nb_actions, activation = 'tanh'))
model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate))

# Permet d'afficher la structure du réseau de neurones
model.summary()

"""

# Pré-entrainement de la mémoire
# Nombre d'épisodes
for i in range(10):

    # Initialisation de l'environnement
    observation = env.reset().reshape([1, dim_observations])

    for i in range(1000):

        env.render()

        # Choisir une action
        if np.random.sample() < epsilon:
            action = env.action_space.sample()
        else:
            action = model.predict(observation)[0]

        # Effectuer l'action
        nouvelle_observation, reward, done, _ = env.step(action)

        # Mémoriser l'ancienne et la nouvelle observation puis la remplacer
        nouvelle_observation = nouvelle_observation.reshape([1, dim_observations])
        memory.append((observation, action, reward, done, nouvelle_observation))
        observation = nouvelle_observation

        if done:
            break
"""

# On fixe le nombre de parties désirées
episodes = 5000
scores = []

for e in range(episodes):

  # Initialisation de l'environnement (à chaque partie)
  observation = env.reset().reshape([1, dim_observations])

  # Début d'une partie / nombre d'actions autorisées
  for i in range(5000):

    env.render()

    # Choisir une action : aléatoire ou prédite par le réseau de neurones
    # La part d'ations prédites augmente avec l'avancement de l'entraînement
    # Le but étant d'éviter que l'agent (la voiture) ne se focalise sur une éventuelle solution sub-optimale
    if np.random.sample() < epsilon:
      action = env.action_space.sample()
    else:
      action = model.predict(observation)[0]

    # Effectuer l'action et récupérer les nouvelles informations dooné par l'environement
    nouvelle_observation, reward, done, _ = env.step(action)

    # Modélisation de la récompense en fonction de la position de la voiture
    # Cela permet d'aider le modèle à bien s'entraîner

    nouvelle_observation = nouvelle_observation.reshape([1, dim_observations])

    # On mémorise l'ancienne observation, l'action choisie, la récompense obtenue,  et la nouvelle observation
    memory.append((observation, action, reward, done, nouvelle_observation))
    observation = nouvelle_observation

    # On additionne les reward afin de se constituer un score

    if done:
      # Fin de la partie (si on atteint le drapeau ou si l'on dépace le nombre d'actions autorisées par partie)
      break

  scores.append(reward)

  print('Episode : {}/{}, Score : {}, Exploration : {:.0%}'.format(e, episodes, reward, epsilon))

  # Entre chaque partie, on entraîne le réseau de neuronnes sur les expériences passées (stockées dans la memory)
  # Sélectionner des expériences aléatoirement dans la mémoire
  # On en sélectionne 48 (batch_size choisi)
  minibatch = random.sample(memory, batch_size)

  # Pour chaque expérience piocher dans le minibatch, on extrait les différents informations
  for observation, action, reward, done, nouvelle_observation in minibatch:

    # Si le drapeau n'est n'est pas atteint, on utilise l'équation de Bellman
    # Cette section du code est développée dans le rapport

    target = reward

    if not done:
        target = ((1.0-0.1) * reward) + 0.1 * discount_rate * np.amax(model.predict(nouvelle_observation)[0])

    target_old = model.predict(observation)
    target_old[0] = target

    model.fit(observation, target_old, epochs = 1, verbose = 0)

  # On baisse le niveau d'exploration (part d'actions aléatoires) à chaque épisode
  if epsilon > epsilon_min:
    epsilon *= epsilon_decay

  # Toutes les 50 parties, on enregistre les poids du réseau de neurones (checkpoints)
  # Nous traçons également l'évolution des scores en fonction du nombre de parties
  if (e+1) % 50 == 0:
    model.save('DQN_Agent.h5')
    print('Le modèle est sauvegardé')

env.close()
