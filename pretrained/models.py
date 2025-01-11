import torch
import copy
model = torch.load('./pretrained/epoch_40.pth')
print(model.keys())

model2 = torch.load('/root/mmpose/pretrained/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth')

# print(model2)
model['state_dict']
new = copy.deepcopy(model)
for key, value in model['state_dict'].items():
    index = key.find(".")
    new_key = key[:index] + ".model" + key[index:]

    print(new_key)
    new['state_dict'][new_key] = new['state_dict'].pop(key)

torch.save(new, './pretrained/rtmo_s.pth')
