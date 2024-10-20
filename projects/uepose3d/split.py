import json



trainset_path = 'data/coco/annotations/person_keypoints_train2017.json'
small_trainset_path = 'data/coco/annotations/small_person_keypoints_train2017.json'

valset_path = 'data/coco/annotations/person_keypoints_val2017.json'
small_valset_path = 'data/coco/annotations/small_person_keypoints_val2017.json'



def split_to_small_set(load_path,dump_path):
    ratio = 0.01
    valset = json.load(open(load_path,'r'))
    new_size = int(len(valset['images']) * ratio)
    valset['images'] = valset['images'][:new_size]
    json.dump(valset,open(dump_path,'w'))


split_to_small_set(trainset_path,small_trainset_path)
split_to_small_set(valset_path,small_valset_path)