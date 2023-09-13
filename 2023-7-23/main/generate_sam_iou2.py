import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam_hq_main.segment_anything import sam_model_registry_baseline, SamPredictor
import os
import json

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()

if __name__ == "__main__":
    sam_checkpoint = "sam_hq_main/pretrained_checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    dataset_dir = '/data/tanglv/data/sod_data/DUTS-TR'
    infoes = open('/data/tanglv/data/sod_data/DUTS-TR/box.json').readlines()
    
    for i in range(1):
        info = json.loads(infoes[i])
        img_path =  info['target']
        gt_path = info['source']
        box = info['box']
        
        image = cv2.imread(os.path.join(dataset_dir,img_path)) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024,1024))
        predictor.set_image(image)
        
        input_box = np.array([box])

        batch_box = False if input_box is None else len(input_box)>1 
        result_path = 'demo/baseline_sam_result/'
        os.makedirs(result_path, exist_ok=True)

        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box = input_box,
            multimask_output=False,
        )
        show_res(masks,scores,None, None, input_box, result_path + 'example'+str(i), image)





