import numpy as np
import kmeans
import common
import naive_em
import em
from common import GaussianMixture


# X = np.loadtxt("toy_data.txt")



# X = np.array([[0.14035078, 0.0],
#              [0.6235637 , 0.38438171],
#              [0.29753461, 0.05671298],
#              [0.27265629,0.47766512],
#              [0.81216873, 0.47997717],
#              [0.3927848 ,0.83607876],
#              [0.33739616, 0.64817187],
#              [0.36824154, 0.95715516],
#              [0.14035078,0.87008726],
#              [0.47360805, 0.80091075],
#              [0.52047748, 0.67887953],
#              [0.72063265, 0.58201979],
#              [0.53737323, 0.75861562],
#              [0.10590761, 0.47360042],
#              [0.18633234 ,0.73691818]])
#
# Mu = np.array([[0.6235637 , 0.38438171],
#              [0.3927848,  0.83607876],
#              [0.81216873 ,0.47997717],
#              [0.14035078 ,0.87008726],
#              [0.36824154 ,0.95715516],
#              [0.10590761, 0.47360042]])
#
# Var = np.array([0.10038354, 0.07227467, 0.13240693,0.12411825, 0.10497521,0.12220856])
# P = np.array([0.1680912 , 0.15835331 ,0.21384187, 0.14223565, 0.14295074 ,0.17452722])
#
# mixture = GaussianMixture(Mu,Var,P)
# weights_e, new_ll = naive_em.estep(X, mixture)
# print(weights_e, new_ll)

# mixture,post = common.init(X, 3, 0)

# new_mixture, new_post, new_ll = naive_em.run(X, mixture,post)

# post, new_ll = naive_em.estep(X, mixture)
# mixture = naive_em.mstep(X, post)
# new_ll2 = naive_em.log_likelihood(X,mixture,post)
# print(new_ll)

# K = [1,2,3,4]
# seeds = [0,1,2,3,4]

# cost = np.zeros((4,5))
# min_cost = np.zeros(4)
# for k in K:
#     seed_min = 0
#     cost_min = 100000000
#     for seed in seeds:
#         mixture,post = common.init(X, k, seed)
#         #common.plot(X, mixture, post, f'Befor Plot for {k} and {seed}')
#         new_mixture, new_post, cost[k-1,seed] = kmeans.run(X, mixture, post)
#         # if cost[k-1,seed] < cost_min:
#         #     cost_min = cost[k-1,seed]
#         #     seed_min = seed
#         #     #common.plot(X, new_mixture, new_post, f'K-means:After Plot for {k} and {seed}')
#
#         min_cost[k-1] = np.min(cost[k-1,:])
#print(min_cost)
#
# ll = np.zeros((4,5))
# max_ll = np.zeros(4)
# for k in K:
#     seed_max = 0
#     ll_max = -10000000
#     for seed in seeds:
#         mixture,post = common.init(X, k, seed)
#         new_mixture, new_post, ll[k-1,seed] = naive_em.run(X, mixture,post)
#         if ll[k-1,seed] > ll_max:
#             ll_max = ll[k-1,seed]
#             seed_max = seed
#             common.plot(X, new_mixture, new_post, f'GMM: After Plot for {k} and {seed}')
#
#     max_ll[k-1] = np.max(ll[k-1,:])
# print(max_ll)

# ll = np.zeros((4,5))
# bic_ll = np.zeros((4,5))
#
# for k in K:
#     for seed in seeds:
#         mixture,post = common.init(X, k, seed)
#         new_mixture, new_post, ll[k-1,seed] = naive_em.run(X, mixture,post)
#         bic_ll[k-1,seed] = common.bic(X,new_mixture,ll[k-1,seed])
#
# print(bic_ll)

#
# post, ll = em.estep(X, mixture)
# mixture = em.mstep(X,post,mixture)
#
# print(post,ll,mixture)

X = np.loadtxt("netflix_incomplete.txt")
# K = [1,12]
# seeds = [0,1,2,3,4]
# ll = np.zeros((2,5))
# max_ll = np.zeros(2)
# for k in range(1,3):
#     seed_max = 0
#     ll_max = -10000000
#     for seed in seeds:
#         mixture,post = common.init(X, K[k-1], seed)
#         new_mixture, new_post, ll[k-1,seed] = em.run(X, mixture,post)
#         if ll[k-1,seed] > ll_max:
#             ll_max = ll[k-1,seed]
#             seed_max = seed
#             # common.plot(X, new_mixture, new_post, f'GMM: After Plot for {k} and {seed}')
#
#     max_ll[k-1] = np.max(ll[k-1,:])
# print(max_ll,seed_max)

mixture,post = common.init(X, 12, 1)
new_mixture, new_post, ll_final = em.run(X, mixture,post)
X_pred = em.fill_matrix(X, new_mixture)
X_gold = np.loadtxt('netflix_complete.txt')
print(common.rmse(X_gold, X_pred))







