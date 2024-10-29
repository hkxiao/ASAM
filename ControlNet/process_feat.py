import torch
from torch.nn import functional as F
from PIL import Image
def process_feat(imgs_desc, imgs_desc_dino, sd_target_dim, dino_target_dim, dino_pca, using_sd, using_dino):
    CO_PCA_DINO = True
    num_patches = 60

    # pca for sd
    imgs_desc = pca_4_sd(imgs_desc, dim=sd_target_dim) # [B,C,H,W]

    # pca for dino    
    if dino_pca:
        imgs_desc_dino = imgs_desc_dino.reshape(num_patches*num_patches, -1)
        mean = torch.mean(imgs_desc_dino, dim=0, keepdim=True)
        centered_features = imgs_desc_dino - mean
        U, S, V = torch.pca_lowrank(centered_features, q=dino_target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :dino_target_dim]) # (t_x+t_y)x(d)
        processed_co_features = reduced_features.unsqueeze(0) #[1 60*60 C]
        imgs_desc_dino = processed_co_features.permute(0,2,1).reshape(-1, processed_co_features.shape[-1], num_patches, num_patches)  
    
    # concat
    imgs_desc = imgs_desc / imgs_desc.norm(dim=1, keepdim=True)
    imgs_desc_dino = imgs_desc_dino / imgs_desc_dino.norm(dim=1, keepdim=True)
    
    B, C, H, W = imgs_desc.shape 
    feat = torch.empty(B,0,H,W).cuda()
    if using_sd: feat = torch.concat([feat, imgs_desc], 1)
    if using_dino: feat = torch.concat([feat, imgs_desc_dino], 1)  
    
    return feat # [1 C H W]

def clip_feat(feature, img_path):
    #feature1 shape (1,1,3600,768*2)
    feature = feature.squeeze() # shape (3600,768*2)
    chennel_dim = feature.shape[-1]
    
    num_patches = 60
    
    h, w = Image.open(img_path).size
    scale_h = h/num_patches
    scale_w = w/num_patches
    if scale_h > scale_w:
        scale = scale_h
        scaled_w = int(w/scale)
        feature = feature.reshape(num_patches,num_patches,chennel_dim)
        feature_uncropped=feature[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
    else:
        scale = scale_w
        scaled_h = int(h/scale)
        feature = feature.reshape(num_patches,num_patches,chennel_dim)
        feature_uncropped=feature[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]
    
    return feature_uncropped #[H W C]
     
def pca_4_sd(features, dim=[128,128,128]):
    processed_features = {}
    B = features['s5'].shape[0]
    
    s5_size = features['s5'].shape[-1]
    s4_size = features['s4'].shape[-1]
    s3_size = features['s3'].shape[-1]
    
    # Get the feature tensors
    s5 = features['s5'].reshape(features['s5'].shape[0], features['s5'].shape[1], -1) #B*C*H*W -> B*C*HW 
    s4 = features['s4'].reshape(features['s4'].shape[0], features['s4'].shape[1], -1)
    s3 = features['s3'].reshape(features['s3'].shape[0], features['s3'].shape[1], -1)

    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [s5, s4, s3]):
        target_dim = target_dims[name]
             
        tensors = tensors.permute(0,2,1).contiguous().reshape(-1,tensors.shape[1]) #B*C*HW -> BHW*C
        mean = torch.mean(tensors, dim=0, keepdim=True)
        centered_features = tensors - mean
        
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)    
        processed_features[name] = reduced_features.reshape(B,-1,target_dim).permute(0,2,1) # BHW*C -> B*HW*C -> B*C*HW

    processed_features['s5']=processed_features['s5'].reshape(processed_features['s5'].shape[0], -1, s5_size, s5_size)
    processed_features['s4']=processed_features['s4'].reshape(processed_features['s4'].shape[0], -1, s4_size, s4_size)
    processed_features['s3']=processed_features['s3'].reshape(processed_features['s3'].shape[0], -1, s3_size, s3_size)
    
    # Upsample s5 spatially by a factor of 2
    processed_features['s5'] = F.interpolate(processed_features['s5'], size=(processed_features['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features['s5'] = torch.cat([processed_features['s4'], processed_features['s5']], dim=1)

    # Set s3 as the new s4
    processed_features['s4'] = processed_features['s3']

    # Remove s3 from the features dictionary
    processed_features.pop('s3')

    # current order are layer 8, 5, 2
    features_gether_s4_s5 = torch.cat([processed_features['s4'], F.interpolate(processed_features['s5'], size=(processed_features['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features_gether_s4_s5
