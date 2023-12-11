import torch
checkpoint1 = torch.load('asam_epoch_9.pth')
checkpoint2 = torch.load('epoch_9.pth')

for k,v in checkpoint1.items():
    if 'iou_token' in k or 'mask_tokens' in k:
        print(torch.sum(checkpoint1[k]),  torch.sum(checkpoint2[k[13:]]))
        checkpoint1[k] = checkpoint2[k[13:]]
torch.save(checkpoint1,'asam_epoch_9.pth')