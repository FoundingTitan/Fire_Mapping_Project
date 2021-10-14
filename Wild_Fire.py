# Wild Fire Simulation

import cv2
import sys
import math
import random
import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation

from PIL import Image
import os
import PIL
import glob

fig = plt.figure()

# initialize environment

# INITIALIZE DATA -------------------JONATHAN'S CHANGES
data_dir = 'C:/Users/jonat/dockerfiles/Parallel_CellularAutomaton_Wildfire/'

class Args:
  def __init__(self):
    self.img_dir = data_dir+"train_images"
    self.mask_dir = data_dir+"train_masks"
    self.fin_mask_dir = data_dir+"final_masks"

args = Args()
print(vars(args))

pic114 = cv2.imread(args.fin_mask_dir+"/114.png", 0)
#pic69 = cv2.imread(args.mask_dir+"/69.jpg", 0)
pic69 = Image.open(args.mask_dir+"/69.jpg")

# --------------------------------------

#Determine which image to use-------------JONATHAN'S CHANGES
img_to_use = pic69

image_resizing = True

# number of rows and columns of grid
n_row = 2973
n_col = 3981
generation = 100

wind = False
vegetation = False
density = False
altitude = False

# Image Resizing -----------------------JONATHAN'S CHANGES
if image_resizing == True:
    fixed_height = 500
    height_percent = (fixed_height / float(img_to_use.size[1]))
    width_size = int((float(img_to_use.size[0]) * float(height_percent)))
    img_to_use = img_to_use.resize((width_size, fixed_height), PIL.Image.NEAREST)
    n_row = int(fixed_height)
    n_col = int(width_size)

    _, img_to_use = cv2.threshold(np.array(img_to_use),127,255,cv2.THRESH_BINARY)

def colormap(i,array):
    np_array = np.array(array)
    plt.imshow(np_array, interpolation="none", cmap=cm.plasma)
    plt.title(i)
    plt.show()


def init_vegetation():
    veg_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    if vegetation == False:  # turn off vegetation
        for i in range(n_row):
            for j in range(n_col):
                veg_matrix[i][j] = 1
    else:
        for i in range(n_row):
            for j in range(n_col):
                if j <= n_col//3 : veg_matrix[i][j] = 1 #// <- integer division
                elif j <= n_col*2//3: veg_matrix[i][j] = 2
                else: veg_matrix[i][j] = 3
    return veg_matrix


def init_density():
    den_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    if density == False:  # turn off density
        for i in range(n_row):
            for j in range(n_col):
                den_matrix[i][j] = 1
    else:
        for i in range(n_row):
            for j in range(n_col):
                if j <= n_col//3: den_matrix[i][j] = 1
                elif j <= n_col*2//3: den_matrix[i][j] = 2
                else: den_matrix[i][j] = 3
    return den_matrix


def init_altitude():
    alt_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    if altitude == False:  # turn off altitude
        for i in range(n_row):
            for j in range(n_col):
                alt_matrix[i][j] = 1
    else:
        for i in range(n_row):
            for j in range(n_col):
                alt_matrix[i][j] = j
    return alt_matrix


def init_forest():
    forest = [[1 for col in range(n_col)] for row in range(n_row)]

    for i in range(n_row):
        if i == 0 or i == n_row - 1:  # [parallel] sub grid initial exchange row is one
            continue
        for j in range(n_col):
            if j == 0 or j == n_col - 1:  # [parallel] sub grid initial margin column is one
                continue
            forest[i][j] = 2

    #Jonathan's changes -> setting initial fire location to match with mask
    #ignite burning compare with picture
    for pixel_x in range(len(img_to_use)):
        for pixel_y in range(len(img_to_use[pixel_x])):
            if img_to_use[pixel_x][pixel_y] == 255:
                forest[pixel_x][pixel_y] = 3

    #ignite_col = int(n_col//2)
    #ignite_row = int(n_row//2)
    #if rank == size // 2:  # [parallel] sub grid only ignite center sub grid
        #for row in range(ignite_row-1, ignite_row+1):
            #for col in range(ignite_col-1,ignite_col+1):
                #forest[row][col] = 3
    # forest[ignite_row-2:ignite_row+2][ignite_col-2:ignite_col+2] = 3
    return forest


def print_forest(forest):
    for i in range(n_row):
        for j in range(n_col):
            sys.stdout.write(str(forest[i][j]))
        sys.stdout.write("\n")


def tg(x):
    return math.degrees(math.atan(x))


def get_slope(altitude_matrix):
    slope_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for row in range(n_row):
        for col in range(n_col):
            sub_slope_matrix = [[0,0,0],[0,0,0],[0,0,0]]
            if row == 0 or row == n_row-1 or col == 0 or col == n_col-1:  # margin is flat
                slope_matrix[row][col] = sub_slope_matrix
                continue
            current_altitude = altitude_matrix[row][col]
            sub_slope_matrix[0][0] = tg((current_altitude - altitude_matrix[row-1][col-1])/1.414)
            sub_slope_matrix[0][1] = tg(current_altitude - altitude_matrix[row-1][col])
            sub_slope_matrix[0][2] = tg((current_altitude - altitude_matrix[row-1][col+1])/1.414)
            sub_slope_matrix[1][0] = tg(current_altitude - altitude_matrix[row][col-1])
            sub_slope_matrix[1][1] = 0
            sub_slope_matrix[1][2] = tg(current_altitude - altitude_matrix[row][col+1])
            sub_slope_matrix[2][0] = tg((current_altitude - altitude_matrix[row+1][col-1])/1.414)
            sub_slope_matrix[2][1] = tg(current_altitude - altitude_matrix[row+1][col])
            sub_slope_matrix[2][2] = tg((current_altitude - altitude_matrix[row+1][col+1])/1.414)
            slope_matrix[row][col] = sub_slope_matrix
    return slope_matrix


def calc_pw(theta):
    c_1 = 0.045
    c_2 = 0.131
    V = 10
    t = math.radians(theta)
    ft = math.exp(V*c_2*(math.cos(t)-1))
    return math.exp(c_1*V)*ft


def get_wind():

    if wind == False:  # turn off wind
        wind_matrix = [[1 for col in [0,1,2]] for row in [0,1,2]]

    else:
        wind_matrix = [[0 for col in [0,1,2]] for row in [0,1,2]]
        thetas = [[45,0,45],
                [90,0,90],
                [135,180,135]]
        for row in [0,1,2]:
            for col in [0,1,2]:
                wind_matrix[row][col] = calc_pw(thetas[row][col])
        wind_matrix[1][1] = 0

    return wind_matrix


def burn_or_not_burn(abs_row,abs_col,neighbour_matrix):
    p_veg = {1:-0.3,2:0,3:0.4}[vegetation_matrix[abs_row][abs_col]]
    p_den = {1:-0.4,2:0,3:0.3}[density_matrix[abs_row][abs_col]]
    p_h = 0.58
    a = 0.078

    for row in [0,1,2]:
        for col in [0,1,2]:
            if neighbour_matrix[row][col] == 3: # we only care there is a neighbour that is burning
                # print(row,col)
                slope = slope_matrix[abs_row][abs_col][row][col]
                p_slope = math.exp(a * slope)
                p_wind = wind_matrix[row][col]
                p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_wind * p_slope
                if p_burn > random.random():
                    return 3  #start burning

    return 2 # not burning


def update_forest(old_forest):
    result_forest = [[1 for i in range(n_col)] for j in range(n_row)]
    for row in range(1, n_row-1):
        for col in range(1, n_col-1):

            if old_forest[row][col] == 1 or old_forest[row][col] == 4:
                result_forest[row][col] = old_forest[row][col]  # no fuel or burnt down
            if old_forest[row][col] == 3:
                if random.random() < 0.4:
                    result_forest[row][col] = 3  # TODO need to change back here
                else:
                    result_forest[row][col] = 4
            if old_forest[row][col] == 2:
                neighbours = [[row_vec[col_vec] for col_vec in range(col-1, col+2)]
                              for row_vec in old_forest[row-1:row+2]]
                # print(neighbours)
                result_forest[row][col] = burn_or_not_burn(row, col, neighbours)
    return result_forest


# start simulation
vegetation_matrix = init_vegetation()
density_matrix = init_density()
altitude_matrix = init_altitude()
wind_matrix = get_wind()
new_forest = init_forest()
slope_matrix = get_slope(altitude_matrix)

ims = []
for i in tqdm(range(generation)):
    new_forest = copy.deepcopy(update_forest(new_forest))
    forest_array = np.array(new_forest)
    im = plt.imshow(forest_array, animated=True, interpolation="none", cmap=cm.plasma )
    # plt.title(i)
    ims.append([im])
    # colormap(i,new_forest)

ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True,repeat_delay=500)
ani.save('animate_life.gif')
plt.show()