# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:53:35 2020

@author: VinLes

(see itch.io/TET9)

"""

from FourDirectionQLearner import FourDirectionQLearner as FDQL
import gc
gc.enable()


params = {}

# discount
params['gamma'] = 0.9
# lr
params['alpha'] = 0.0001
# exploration
params['epsilon'] = 0.4
# stop
params['num_of_rounds'] = 1000
# excat stop if
params['ES'] = False
# warm start
params['num_of_warm_iter'] = 10
# visualize random?
params["rand_vis"] = True


def main():
    fdql = FDQL('main.txt', params=params)
    fdql.train()


if __name__ == '__main__':
    main()
