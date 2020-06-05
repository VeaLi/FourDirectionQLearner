# -*- coding: utf-8 -*-
"""
Created on ~

@author: VinLes

(see itch.io/TET9)
    for diploma project at 2020
"""


import matplotlib.pyplot as plt

import time
import numpy as np
import os
from PIL import Image
from toimage import *
import gc
gc.enable()


def seq_it(mode='RGB', lab=None, path=None, NAME=None):
    '''
    rearranged function from Labyrinth Animation project. See Labyrinth Animation project
    '''

    print('Calculating ...')

    if mode == 'RGB':
        NW = None
        #print('Animating in RGB mode, it may take time. For larger data please consider to use SIMPLE mode:).\n\n ***')
        NW = list(range(2, len(path)+2))

        t = [0.01 for _ in range(10)]
        tz = [t for _ in range(10)]
        WALL = np.array(tz)

        # empty cell
        EC = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0, 0],
                       [0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0],
                       [0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0],
                       [0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0],
                       [0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0],
                       [0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0],
                       [0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0],
                       [0, 0, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        for nw, xy in zip(NW, path):
            # print('!!!!!!!!!!!!!',path)
            lab[xy[0]][xy[1]] = nw

        # dictionary of colors
        DC, DC1, DC2 = {}, {}, {}

        def seq(lab, tocolor):
            IMG = []
            wall = 1
            for row in lab:
                tmp = []
                for item in row:
                    if item in tocolor:
                        tmp.append(DC[item])
                    elif (item == wall):
                        tmp.append(WALL)
                    else:
                        tmp.append(EC)
                IMG.append(tmp)

            IMG1 = []
            wall = 1
            for row in lab:
                tmp = []
                for item in row:
                    if item in tocolor:
                        tmp.append(DC1[item])
                    elif (item == wall):
                        tmp.append(WALL)
                    else:
                        tmp.append(EC)
                IMG1.append(tmp)

            IMG2 = []
            wall = 1
            for row in lab:
                tmp = []
                for item in row:
                    if item in tocolor:
                        tmp.append(DC2[item])
                    elif (item == wall):
                        tmp.append(WALL)
                    else:
                        tmp.append(EC)
                IMG2.append(tmp)

            return IMG, IMG1, IMG2

        IMAGES, IMAGES1, IMAGES2 = [], [], []
        tocolor = []

        for color in NW:
            if color != 1:
                tocolor.append(color)
                DC[color] = (np.where(EC == 0.98, np.random.random(), EC))
                DC1[color] = (np.where(EC == 0.98, np.random.random(), EC))
                DC2[color] = (np.where(EC == 0.98, np.random.random(), EC))

                IMG, IMG1, IMG2 = seq(lab, tocolor)
                IMAGES.append(IMG)
                IMAGES1.append(IMG1)
                IMAGES2.append(IMG2)

        ####part 2 #####

        I2, I3, I4 = [], [], []

        for IMG in IMAGES:
            STRIPES = []
            for row in IMG:
                STRIPE = []
                for item in row:
                    # print(item.shape)
                    try:
                        STRIPE[0] = np.hstack((STRIPE[0], item))
                    except:
                        STRIPE = [item]
                STRIPES.append(STRIPE[0])

            IMG2 = []
            for S in STRIPES:
                try:
                    IMG2[0] = np.vstack((IMG2[0], S))
                except Exception as e:
                    IMG2 = [S]
            I2.append(IMG2)

        for IMG in IMAGES1:

            STRIPES = []
            for row in IMG:
                STRIPE = []
                for item in row:
                    try:
                        STRIPE[0] = np.hstack((STRIPE[0], item))
                    except:
                        STRIPE = [item]
                STRIPES.append(STRIPE[0])

            IMG3 = []
            for S in STRIPES:
                try:
                    IMG3[0] = np.vstack((IMG3[0], S))
                except Exception as e:
                    IMG3 = [S]
            I3.append(IMG3)

        for IMG in IMAGES2:

            STRIPES = []
            for row in IMG:
                STRIPE = []
                for item in row:
                    try:
                        STRIPE[0] = np.hstack((STRIPE[0], item))
                    except:
                        STRIPE = [item]
                STRIPES.append(STRIPE[0])

            IMG4 = []
            for S in STRIPES:
                try:
                    IMG4[0] = np.vstack((IMG4[0], S))
                except Exception as e:
                    IMG4 = [S]
            I4.append(IMG4)

        #### part 3 ####
        try:
            os.makedirs('ANIM')
        except:
            pass

        IMG2, IMG3, IMG4 = I2[-1], I3[-1], I4[-1]
        IMG2[0] = np.stack((IMG2[0], IMG3[0], IMG4[0]), axis=2)
        toimage(IMG2[0], cmin=0.0, cmax=1.0).save('ANIM/{}.png'.format(NAME))
