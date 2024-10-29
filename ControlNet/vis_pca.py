def vis_pca(result,save_path,src_img_path,trg_img_path):
    
    # PCA visualization
    for (feature1,feature2,mask1,mask2) in result:
        # feature1 shape (1,1,3600,768*2)
        # feature2 shape (1,1,3600,768*2)
        num_patches=int(math.sqrt(feature1.shape[2]))
        # pca the concatenated feature to 3 dimensions
        feature1 = feature1.squeeze() # shape (3600,768*2)
        feature2 = feature2.squeeze() # shape (3600,768*2)
        chennel_dim = feature1.shape[-1]
        # resize back
        h1, w1 = Image.open(src_img_path).size
        scale_h1 = h1/num_patches
        scale_w1 = w1/num_patches
        
        if scale_h1 > scale_w1:
            scale = scale_h1
            scaled_w = int(w1/scale)
            feature1 = feature1.reshape(num_patches,num_patches,chennel_dim)
            feature1_uncropped=feature1[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
        else:
            scale = scale_w1
            scaled_h = int(h1/scale)
            feature1 = feature1.reshape(num_patches,num_patches,chennel_dim)
            feature1_uncropped=feature1[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]
        
        h2, w2 = Image.open(trg_img_path).size
        scale_h2 = h2/num_patches
        scale_w2 = w2/num_patches
        if scale_h2 > scale_w2:
            scale = scale_h2
            scaled_w = int(w2/scale)
            feature2 = feature2.reshape(num_patches,num_patches,chennel_dim)
            feature2_uncropped=feature2[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
        else:
            scale = scale_w2
            scaled_h = int(h2/scale)
            feature2 = feature2.reshape(num_patches,num_patches,chennel_dim)
            feature2_uncropped=feature2[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]

        f1_shape=feature1_uncropped.shape[:2]
        f2_shape=feature2_uncropped.shape[:2]
        feature1 = feature1_uncropped.reshape(f1_shape[0]*f1_shape[1],chennel_dim)
        feature2 = feature2_uncropped.reshape(f2_shape[0]*f2_shape[1],chennel_dim)
        n_components=3
        pca = sklearnPCA(n_components=n_components)
        feature1_n_feature2 = torch.cat((feature1,feature2),dim=0) # shape (7200,768*2)
        feature1_n_feature2 = pca.fit_transform(feature1_n_feature2.cpu().numpy()) # shape (7200,3)
        feature1 = feature1_n_feature2[:feature1.shape[0],:] # shape (3600,3)
        feature2 = feature1_n_feature2[feature1.shape[0]:,:] # shape (3600,3)
        
        
        fig, axes = plt.subplots(4, 2, figsize=(10, 14))
        for show_channel in range(n_components):
            # min max normalize the feature map
            feature1[:, show_channel] = (feature1[:, show_channel] - feature1[:, show_channel].min()) / (feature1[:, show_channel].max() - feature1[:, show_channel].min())
            feature2[:, show_channel] = (feature2[:, show_channel] - feature2[:, show_channel].min()) / (feature2[:, show_channel].max() - feature2[:, show_channel].min())
            feature1_first_channel = feature1[:, show_channel].reshape(f1_shape[0], f1_shape[1])
            feature2_first_channel = feature2[:, show_channel].reshape(f2_shape[0], f2_shape[1])

            axes[show_channel, 0].imshow(feature1_first_channel)
            axes[show_channel, 0].axis('off')
            axes[show_channel, 1].imshow(feature2_first_channel)
            axes[show_channel, 1].axis('off')
            axes[show_channel, 0].set_title('Feature 1 - Channel {}'.format(show_channel + 1), fontsize=14)
            axes[show_channel, 1].set_title('Feature 2 - Channel {}'.format(show_channel + 1), fontsize=14)


        feature1_resized = feature1[:, :3].reshape(f1_shape[0], f1_shape[1], 3)
        feature2_resized = feature2[:, :3].reshape(f2_shape[0], f2_shape[1], 3)

        axes[3, 0].imshow(feature1_resized)
        axes[3, 0].axis('off')
        axes[3, 1].imshow(feature2_resized)
        axes[3, 1].axis('off')
        axes[3, 0].set_title('Feature 1 - All Channels', fontsize=14)
        axes[3, 1].set_title('Feature 2 - All Channels', fontsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig(save_path+'/pca.png', dpi=300)