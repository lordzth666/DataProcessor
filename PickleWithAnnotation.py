import sys
import os
import numpy as np
import cv2
import pickle

if len(sys.argv) < 5:
    print("Usage: python PickleWtihAnnotation.py [image_path] [annotation_path] [class-label kv pair path] [output_path]")
    exit(1)

image_path = sys.argv[1]
annotation_path = sys.argv[2]
kv_path = sys.argv[3]
output_path = sys.argv[4]

# Pair annotation
annotation_labels = {}

# Load kv pair
kvs = {}
with open(kv_path) as fp:
    line = fp.readline()
    while line != "":
        line = line.split("\t")
        kvs[line[0]] = int(line[1])
        line = fp.readline()
    fp.close()

line_cnt = 0
with open(annotation_path) as fp:
    line = fp.readline()
    while line != "":
        line_cnt += 1
        line = line.split("\t")
        annotation_labels[line[0]] = kvs[line[1]]
        line = fp.readline()
    fp.close()


patch = {}
patch["data"] = None
patch["labels"] = np.zeros(line_cnt, dtype=np.int16)

patch_shape = None
cnt = 0

dirs = os.listdir(image_path)
np.random.shuffle(dirs)

for files in dirs:
    filename = image_path + "/" + files
    img = np.asarray(cv2.imread(filename))
    label = annotation_labels[files]
    if patch["data"] is None:
        data_shape = np.shape(img)
        patch_shape = [line_cnt] + list(data_shape)
        patch["data"] = np.zeros(patch_shape, dtype=np.uint8)
    patch["data"][cnt] = img
    patch["labels"][cnt] = label
    cnt += 1
    sys.stdout.flush()
    sys.stdout.write("Processing %d/%d\r" % (cnt, line_cnt))

pickle.dump(patch, open(output_path, "wb"))