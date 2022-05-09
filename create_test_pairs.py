import glob
import os
import numpy as np
import random
from random import sample

testDataLeft = 75
testDataRight = 116

data_path = "../dataSet/CASIAB_SGEI"
save_txt_dir = "./load_data/averageData/nm"

def loadProbeData(angle):
    human_id = ["%03d" % i for i in range(testDataLeft, testDataRight+1)]  # CASIA B
    type_dir = ["nm%02d" % i for i in range(5, 7)]#["cl01", "cl02"]#["bg01", "bg02"]#["nm%02d" % i for i in range(1, 5)]  # CASIA B   , "cl01", "cl02"
    filelist = [os.path.join(data_path, "%s-%s-%s_GEI.png" % (id, type,angle)) for id in human_id for type in type_dir]
    filelist = sorted(filelist)

    x = []
    y = []
    for img_path in filelist:
        x.append(img_path)
        name = int(os.path.basename(img_path)[:3]) - 1
        y.append(name)
    return x,y

def loadGallaryData(angle):
    human_id = ["%03d" % i for i in range(testDataLeft, testDataRight+1)]  # CASIA B
    type_dir = ["nm%02d" % i for i in range(1, 5)] # CASIA B
    angle_dir = ["%03d" % x for x in range(0, 181, 18)]  # CASIA B
    #angle_dir.remove(angle)

    filelist = [os.path.join(data_path, "%s-%s-%s_GEI.png" % (id, type, angle)) for id in human_id for type in type_dir for angle in angle_dir]
    filelist = sorted(filelist)
    x = []
    y = []
    for img_path in filelist:
        x.append(img_path)
        name = int(os.path.basename(img_path)[:3]) - 1
        y.append(name)
    return x,y

if __name__ == '__main__':
    angle_dir = ["%03d" % x for x in range(0, 181, 18)]  # CASIA B
    for angle in angle_dir:
        probe_x,probe_y = loadProbeData(angle)
        gallary_x,gallary_y = loadGallaryData(angle)
        print(len(probe_x),len(gallary_x))

        if not os.path.isdir(save_txt_dir):
            os.makedirs(save_txt_dir)
        positivePair = []
        negativePair = []
        with open("{}/{}.txt".format(save_txt_dir,angle),"w") as txt:
            for i in range(len(probe_x)):
                for j in range(len(gallary_x)):
                    if probe_y[i] == gallary_y[j]:
                        positivePair.append([probe_x[i], gallary_x[j], 1])
                    else : negativePair.append([probe_x[i], gallary_x[j], 0])
            

            negativePair = sample(negativePair, len(positivePair))
            for data in (positivePair + negativePair) :
                txt.writelines("{},{},{}\n".format(data[0], data[1], data[2]))



