import json
from os import listdir, path
from pprint import pprint

final_dataset = {}

positive_files_path = "dataset/review_polarity/txt_sentoken/pos/"
positive_files=[f for f in listdir(positive_files_path)]

negative_files_path = "dataset/review_polarity/txt_sentoken/neg/"
negative_files=[f for f in listdir(negative_files_path)]

def add_to_dict(filename, positive_flag = True):
    if positive_flag:
        dir_path = positive_files_path
        label = 1
    else:
        dir_path = negative_files_path
        label = 0

    global final_dataset

    with open(path.join(dir_path,filename),"r") as fin:
        final_dataset[filename] = {}
        final_dataset[filename]["review"] = fin.read()
        final_dataset[filename]["polarity"] = label

for p_file in positive_files:
    add_to_dict(p_file)

for n_file in negative_files:
    add_to_dict(n_file,positive_flag =False)

with open("dataset/final_dataset.json",'w') as fout:
    json.dump(final_dataset,fout,indent=4)