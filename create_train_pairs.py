import glob
import os
import numpy as np
import random
train_left = 1
train_right = 4
train_people = train_right - train_left + 1
valid_left= 5
valid_right= 10
valid_people = valid_right - valid_left + 1
path = "../dataSet/CASIAB_SGEI"

human_id = ["%03d" % i for i in range(train_left, train_right + 1)]  # CASIA B
type_dir = ["nm%02d" % i for i in range(1, 7)] #+ ["bg01","bg02","cl01","cl02"]  # CASIA B
angle_dir = ["%03d" % x for x in range(0, 181, 18)] #CASIA B
filelist = [os.path.join(path, f) for f in os.listdir(path) for id in human_id for type in type_dir for angle in angle_dir if f == ("%s-%s-%s_GEI.png" % (id, type, angle))]
filelist = sorted(filelist)
label_list = np.array([int(os.path.basename(label)[:3]) for label in filelist])
digit_indices = [np.where(label_list == i)[0] for i in range(train_left,train_right+1)]
img_per = len(filelist) / train_people
pairs = []
labels = []
n = min([len(digit_indices[d]) for d in range(train_people)])

for id in range(train_people):
    for i in range(n):
        for j in range(i + 1, n):
            positive_1, positive_2 = digit_indices[id][i], digit_indices[id][j]
            pairs += [[filelist[positive_1], filelist[positive_2]]]
            ne = random.randint(0, train_people - 1)
            print(id,ne)
            while id == ne:
                ne = random.randint(0, train_people - 1)

            po = random.randint(0, img_per-1)

            z1, z2 = digit_indices[id][i], digit_indices[ne][po]

            pairs += [[filelist[z1], filelist[z2]]]

            labels += [1, 0]
print(len(labels))

with open("./train_sgei.txt","w") as txt:
    for i in range(len(pairs)):
        txt.writelines("{},{},{}\n".format(pairs[i][0],pairs[i][1],labels[i]))


del pairs,labels,human_id,type_dir,angle_dir,filelist


human_id = ["%03d" % i for i in range(valid_left, valid_right + 1)]  # CASIA B
type_dir = ["nm%02d" % i for i in range(1, 7)]# + ["bg01","bg02","cl01","cl02"]  # CASIA B
angle_dir = ["%03d" % x for x in range(0, 181, 18)] #CASIA B
filelist = [os.path.join(path, f) for f in os.listdir(path) for id in human_id for type in type_dir for angle in angle_dir if f == ("%s-%s-%s_GEI.png" % (id, type, angle))]
filelist = sorted(filelist)
label_list = np.array([int(os.path.basename(label)[:3]) for label in filelist])

digit_indices = [np.where(label_list == i)[0] for i in range(valid_left,valid_right+1)]
# print(digit_indices)
img_per = len(filelist) / valid_people
pairs = []
labels = []
n = min([len(digit_indices[d]) for d in range(valid_people)])
for id in range(valid_people):
    for i in range(n):
        for j in range(i + 1, n):
            positive_1, positive_2 = digit_indices[id][i], digit_indices[id][j]
            pairs += [[filelist[positive_1], filelist[positive_2]]]

            ne = random.randint(0, valid_people - 1)
            while id == ne:
                ne = random.randint(0, valid_people - 1)
            po = random.randint(0, img_per-1)
            z1, z2 = digit_indices[id][i], digit_indices[ne][po]
            pairs += [[filelist[z1], filelist[z2]]]
            labels += [1, 0]
print(len(labels))
with open("./valid_sgei.txt","w") as txt:
    for i in range(len(pairs)):
        txt.writelines("{},{},{}\n".format(pairs[i][0],pairs[i][1],labels[i]))






