# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:19:58 2020

@author: VinLes

(see itch.io/TET9)

"""

import numpy as np
from seq_it import seq_it
from tqdm import tqdm
import random
import joblib
import time
import pandas as pd


class FourDirectionQLearner(object):
    """
    This object can load 2d map from text file and learn in the following action space: ['left','right','up','down']

    Args:
        lab_file_name (str): File name path, ex, "main.txt"
        params (dict): dictionary of parameters
    example of params arg:
        params = {'gamma': 0.9, 'alpha': 0.001, 'epsilon': 0.4, 'num_of_rounds': 1000,'ES': False, 'num_of_warm_iter': 10, 'rand_vis': True}

    Attributes:
        lab_file_name (str): File name path
        params (dict): dictionary of parameters passed
        action_size (int): 4, should be fixed
        lab (numpy.ndarray): map loaded, using lab_file_name
        free_walls_dict (dict): dictionary of all cells that are free of walls and can be accessed, starting from the current cell 'i'
            such as, free_walls_dict[row_column]  -> [can_go_left,can_go_right, can_go_up, can_go_down]
        state_size (int): this field will be estimated, after free_walls_dict
        Q_INDEX (dict): mappings from the game identifier (number in row) to start_row,start_col - end_row,end_col
        Q_back (dict): inverse for Q_index
        TVR (list): total victory rate
        EVR (lsit): exact victory rate
        Q (numpy.ndarray): q_table state_size x self.action_size



    """

    def __init__(self, lab_file_name='main.txt', params=None):
        self.lab_file_name = lab_file_name
        self.params = params
        self.action_size = 4  # ['left','right','up','down']
        self.lab = self.read_lab()
        self.free_walls_dict = {}
        self.state_size = None
        self.action_size = None
        self.Q_INDEX = {}
        self.Q_BACK = {}

        self.get_free_walls_dict()

        self.TVR = [0]  # total victtory rate
        self.EVR = [0]  # exact victory rate
        self.Q = np.zeros((self.state_size, self.action_size))

    def read_lab(self):
        file = self.lab_file_name
        lab = []
        with open(file, 'r') as labfile:
            lab = labfile.readlines()
        lab = [[int(y) for y in list(x.strip())] for x in lab]
        return np.array(lab)

    def get_free_walls_dict(self):
        lab = self.lab
        free = []
        for row in range(len(self.lab)):
            for col in range(len(self.lab[row])):
                if self.lab[row][col] == 0:
                    free.append([row, col])

        # up down left right UD LR
        for rc in free:
            walls = []  # up
            if (lab[rc[0]-1][rc[1]]) == 0:
                walls.append(1)
            else:
                walls.append(0)

        # down
            if (lab[rc[0]+1][rc[1]]) == 0:
                walls.append(1)
            else:
                walls.append(0)

        # left
            if (lab[rc[0]][rc[1]-1]) == 0:
                walls.append(1)
            else:
                walls.append(0)

        # right
            if (lab[rc[0]][rc[1]+1]) == 0:
                walls.append(1)
            else:
                walls.append(0)

            self.free_walls_dict[str(rc[0]) + "_" + str(rc[1])] = walls

        # table // up down left right x y xs ys estimate // do_up do_down do_left do_right // reward
        self.state_size, self.action_size = len(
            free)*len(free), 4  # pos cube * pos fruit
        print(f"estimated state size: {self.state_size}, action size: {4}\n")

        CNT = 0
        # [row,col]
        for rc_player in free:
            for rc_fruit in free:
                self.Q_INDEX[CNT] = rc_player + \
                    rc_fruit  # list of for numericals
                self.Q_BACK[str(rc_player+rc_fruit)] = CNT
                CNT += 1

        joblib.dump(self.Q_BACK, "Q_state_dictionary.dic")

    def reward(self, x, y, xs, ys, N, R):
        """
        Calculates reward for all cycles of learning. ~ Estimated distance + reward for long path * is path found
        Args:
            x (int): start x position
            y (int): start y position
            xs (int): end x position
            ys (int): end y position
            N (int): Minkowski distance order
            R (int): current round 
        Returns:
            rewars_raw (lsit): returns all action space rewards raw
        """
        # heuristic to goal initial XY ok
        def fx(x, y, n): return (
            (np.absolute(x[0]-y[0])**n + np.absolute(x[1]-y[1])**n)**(1/n))+0.0001

        h = fx([x, y], [xs, ys], N)

        # !!!YX reversed!! row + col
        list_to_do = self.free_walls_dict[str(y) + "_" + str(x)]

        rewards_raw = [-100, -100, -100, -100]
        VS = 0  # victories counter

        # udlr
        if sum(list_to_do) > 0:  # if not single cell "island"
            for i, d in zip(range(4), [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]):
                if list_to_do[i] == 1:
                    rewards_raw[i] = (h / fx(d, [xs, ys], N))

                    if (R > self.params['num_of_warm_iter'] and np.random.uniform(0, 1, 1) < R/50):
                        won, path, LP, LSP = self.check_path(
                            y=d[1], x=d[0], xs=xs, ys=ys, NAME=None)
                        if(won):
                            VS += 1
                            rewards_raw[i] += LSP
                        elif LP == LSP:
                            rewards_raw[i] -= 0.1
                        else:
                            rewards_raw[i] -= LP/LSP

            if VS > 0:
                self.EVR[0] = (self.EVR[0]-VS)+1

        M = np.absolute(max(rewards_raw))
        rewards_raw = [(rr/M) for rr in rewards_raw]

        return rewards_raw

    def check_path(self, NAME, y, x, xs, ys):
        """
        Check is path exists
        Args:
            Name (any): save picture under specific name, regardless of outcome
            x (int): start x position
            y (int): start y position
            xs (int): end x position
            ys (int): end y position
        Returns:
            rewars_raw (lsit): returns all action space rewards raw
        """
        path = []
        table = []
        QT = []
        LP, LSP = 0, 0
        while LP <= LSP+10:
            LP = len(path)
            LSP = len(set(path))
            try:
                path.append((y, x))
                IND = self.Q_BACK[str([y, x, ys, xs])]
                vals = self.Q[IND]
                action = np.argmax(vals)
                QT.append([IND, action])
                if NAME != None:
                    line = {}
                    line['pos_agent'] = (y, x)
                    line['pos_target'] = (ys, xs)
                    line['up_reward'] = vals[0]
                    line['down_reward'] = vals[1]
                    line['left_reward'] = vals[2]
                    line['right_reward'] = vals[3]
                    table.append(line)
            except Exception as e:
                if NAME != None:
                    lab = self.read_lab()
                    seq_it(lab=lab, path=path, NAME=NAME)
                    print(f"\nFail with LP/LSP,LSP: {np.round(LP/LSP,2)},{LSP} at key {e}, which is not included (ex. wall)\n")
                return (False, path, LP, LSP)

            if action == 0:
                y -= 1

            elif action == 1:
                y += 1

            elif action == 2:
                x -= 1

            elif action == 3:
                x += 1

            if x == xs and y == ys:
                if NAME != None:
                    # a discardable copy
                    lab = self.read_lab()
                    seq_it(lab=lab, path=path, NAME=NAME)
                    print('Succes', len(path))
                    pd.DataFrame(table).to_csv(f'sample_rewards_at_{str(time.time())}.csv')

                if (self.params['rand_vis']):
                    if (len(path) > 80 and np.random.uniform(0, 1, 1) < 0.0001) or np.random.uniform(0, 1, 1) < 0.00001:
                        # a discardable copy
                        lab = self.read_lab()
                        seq_it(lab=lab, path=path, NAME=str(time.time()))
                        print("\ncurrent victory rate: ",
                              self.EVR[0]/self.Q.shape[0])
                        print(f"Succes with LP/LSP,LSP: {np.round(LP/LSP,2)},{LSP} with values: {vals}\n")

                self.TVR[0] += 1
                self.EVR[0] += 1
                for i in range(len(QT)):
                    qt = QT[i]
                    self.Q[qt[0], qt[1]] += 1
                return (True, path, LP, LSP)
        if NAME != None:
            # a discardable copy
            lab = self.read_lab()
            seq_it(lab=lab, path=path, NAME=NAME)
            print('\nFail max path ', LSP, '\n')
        return (False, path, LP, LSP)

    def choose_random_possible(self, x, y):
        '''
        This function returns a possible move from current cell

        Args:
            x (int): current cell x
            y (int): current cell y

        Returns:
            _ (int): index of possible move
        '''
        possible = self.free_walls_dict[str(y) + "_" + str(x)]
        moves_i = []
        for i in range(4):
            if possible[i] == 1:
                moves_i.append(i)
        return random.choice(moves_i)

    def train(self):
        '''
        This is the inner function. It starts q-learning on the given map, using the passed parameters.
        '''

        print(self.params)

        print('\n', "-"*64, '\n')

        with open('res.csv', 'a') as f:
            f.write(
                "cycle ,percent_0f_total_victories,percent_0f_distinct_game_victories,epsilon,alpha,timestamp,N")
            f.write('\n')

        alpha, epsilon, gamma = self.params['alpha'], self.params['epsilon'], self.params['gamma']
        WARM = self.params['num_of_warm_iter']
        IS = False
        # bad cells Double Walls
        DW = 1
        for ROUND in range(0, self.params['num_of_rounds']):

            if np.max(self.Q) > 500:
                self.Q = self.Q/2

            joblib.dump(self.Q, "ALL/"+str(ROUND)+"Q_tabel.qt")
            N = random.choice(
                [1, 1, 1, 1, 2, 2, 2, 2, 2, 0.25, 0.25, 0.71, 3, 4])
            print('-', DW)

            with open('res.csv', 'a') as f:
                f.write(str(ROUND)+","+str(self.TVR[0]/(self.Q.shape[0]-DW))+","+str(self.EVR[0]/(
                    self.Q.shape[0]-DW))+","+str(epsilon)+","+str(alpha)+','+str(time.time())+","+str(N))
                f.write('\n')
            if self.EVR[0]/(self.Q.shape[0]-DW) >= 0.9999:
                print('Congrats ~all of the games should be won')
                break
            self.TVR[0] = 0
            self.EVR[0] = 0
            if IS == False or self.params['ES'] == False:
                IS, path, _, _ = self.check_path(
                    NAME=ROUND, y=7, x=16, xs=self.lab.shape[1]-2, ys=self.lab.shape[0]-2)
                if IS:
                    joblib.dump(self.Q, str(ROUND)+"Q_tabelv.qt")

                if ROUND > WARM:
                    epsilon -= 0.01
                    alpha -= 0.000001
                    epsilon = max(0.001, epsilon)
                    alpha = max(0.000001, alpha)

                DW = 1

                for i in tqdm(range(self.Q.shape[0]), miniters=5000):
                    try:
                        coords = self.Q_INDEX[i]

                        # first iteration mark walls
                        if ROUND == 0:
                            list_to_do = self.free_walls_dict[str(
                                coords[0]) + "_" + str(coords[1])]
                            self.Q[i] = self.Q[i] + \
                                np.array([e*1 for e in list_to_do])
                            continue

                        # check if you are at place and stop
                        if coords[0] == coords[2] and coords[1] == coords[3]:
                            rewards = [1, 1, 1, 1]
                            continue
                        else:
                            rewards = self.reward(
                                coords[1], coords[0], coords[3], coords[2], R=ROUND, N=N)

                        if np.random.uniform(0, 1, 1) < epsilon:

                            # Explore: select a random action
                            action = self.choose_random_possible(
                                coords[1], coords[0])
                        else:

                            # Exploit: select the action with max value (future reward)
                            action = np.argmax(rewards)

                        cur_reward = rewards[action]

                        NS = 0  # new state
                        if action == 0:
                            # up
                            NS = self.Q_BACK[str(
                                [coords[0]-1, coords[1], coords[2], coords[3]])]
                        elif action == 1:
                            # down
                            NS = self.Q_BACK[str(
                                [coords[0]+1, coords[1], coords[2], coords[3]])]
                        elif action == 2:
                            # left
                            NS = self.Q_BACK[str(
                                [coords[0], coords[1]-1, coords[2], coords[3]])]
                        else:
                            # right
                            NS = self.Q_BACK[str(
                                [coords[0], coords[1]+1, coords[2], coords[3]])]

                        self.Q[i, action] = self.Q[i, action] + alpha * \
                            (cur_reward+gamma *
                             np.max(self.Q[NS, :]) - self.Q[i, action])

                    except Exception as e:
                        # double walls or separated island
                        DW += 1
