import pandas as pd
import torch
from DP_util import get_samples, select_main_object, sam_model_registry
from file_path import sam_path
import matplotlib.pyplot as plt
import torchvision


'''fig_idx = [7000 + i for i in range(100)]
datapoints, _ = get_samples(image_idx=fig_idx, device='cpu')
plt.figure(figsize=(6.4, 1.8))

col_num = 15
tp = torchvision.transforms.ToPILImage()
for idx in range(len(datapoints)):
    plt.subplot(len(datapoints) // col_num + 1, col_num, idx + 1)
    plt.axis('off')
    plt.imshow(datapoints[idx])
plt.subplots_adjust(wspace=0.07, hspace=0)
plt.show()'''

'''sam = sam_model_registry["default"](checkpoint=sam_path)
sam.to('cuda')
for bs in (1,):
    for r in range(10):
        torch.cuda.empty_cache()
        print(bs, r)
        datapoints, labels = get_samples([], 'cuda', 'ImageNet', bs)
        datapoints, optimal_masks = select_main_object(sam, datapoints)  # process by SAM
        torch.save(datapoints, 'evaluation/processed data/main_object_{0}bs_{1}.pt'.format(bs, r))
        torch.save(optimal_masks, 'evaluation/processed data/mask_{0}bs_{1}.pt'.format(bs, r))'''
