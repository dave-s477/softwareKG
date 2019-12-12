from os import listdir, mkdir
from os.path import exists, join

file_names = [x.split('.')[0] for x in listdir('../data/FINISH/') if x.endswith('.txt')]

if not exists('../data/FINISH_transformed'):
    mkdir("../data/FINISH_transformed")
for f in file_names:
    with open(join("../data/FINISH/", f)+'.txt', 'r') as in_txt, \
            open(join("../data/FINISH/", f)+'.ann', 'r') as in_ann, \
            open(join("../data/FINISH/", f)+'.src', 'r') as in_src, \
            open(join("../data/FINISH_transformed/", f)+'.txt', 'w') as out_txt, \
            open(join("../data/FINISH_transformed/", f)+'.ann', 'w') as out_ann:
        print(f)