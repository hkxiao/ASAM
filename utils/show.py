import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color.any() == None:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_points(coords, labels, ax, color ,marker_size=375):
    pos_points = [coord for i, coord in enumerate(coords) if labels[i]==1]    
    ax.scatter(tuple([p[0] for p in pos_points]), tuple([p[1] for p in pos_points]), color=color, marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    # neg_points = [coord for i, coord in enumerate(coords) if labels[i]==0]  
    # ax.scatter(tuple([p[0] for p in neg_points]), tuple([p[0] for p in neg_points]), color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, color):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2)) 