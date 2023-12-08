import torch

feature_list = [[torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.2])], [torch.tensor([0.3, 0.4]), torch.tensor([0.3, 0.4]), torch.tensor([0.3, 0.4])], []]


new_feature_list = [f_l if len(f_l) == 0 else torch.stack(f_l, dim=0) for f_l in feature_list]

for f_l in new_feature_list:
    if type(f_l) == list:
        print(1)
    else:
        print(type(f_l))

print()