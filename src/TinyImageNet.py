from src.Dataset import DataSet
from os import listdir, makedirs
from os.path import exists
import pickle
import cv2
import numpy as np
import operator
import sys

class ImageNet(DataSet):
    def __init__(self, config):
        super().__init__(source_path=config.source_path,
                         save_base_path=config.save_base_path,
                         filename=config.filename,
                         num_batches=config.num_batches)

    def process(self):
        # Calculating how many samples do we have
        tot_files = 0
        for folders in listdir(self.source_path):
            tot_files += len(listdir(self.source_path + "/" + folders + "/images/"))
        files_per_batch = int(tot_files / self.num_batches)
        patch = {}
        patch["data"] = None
        patch["labels"] = np.zeros(files_per_batch, dtype=int)
        class_id = 0
        classes = {}
        cur_batch = 1
        count = 0
        tot_count = 0

        if not exists(self.save_base_path):
            makedirs(self.save_base_path)

        patch_shape = None

        for folders in listdir(self.source_path):
            dst_dir = self.source_path + "/" + folders + "/images/"
            for files in listdir(dst_dir):
                tot_count += 1
                img = np.asarray(cv2.imread(dst_dir + files), dtype=np.uint8)
                if patch["data"] is None:
                    dshape = list(np.shape(img))
                    patch_shape = [files_per_batch] + dshape
                    print(patch_shape)
                    patch["data"] = np.zeros(patch_shape, dtype=np.uint8)
                sys.stdout.write("Processing %d/%d\r" %(tot_count, tot_files))
                patch["data"][count] = img
                patch["labels"][count] = class_id
                count += 1
                if count == files_per_batch:
                    num_elements = np.shape(patch["data"])[0]
                    indices = np.arange(num_elements)
                    np.random.shuffle(indices)
                    patch["data"][:] = patch["data"][indices]
                    patch["labels"][:] = patch["labels"][indices]
                    count = 0
                    output_filename = self.save_base_path + "/" + self.filename + '%d' %cur_batch
                    pickle.dump(patch, open(output_filename, "wb"))
                    patch["data"] = np.zeros(patch_shape, dtype=np.uint8)
                    patch["labels"] = np.zeros(files_per_batch, dtype=int)
                    cur_batch += 1
                sys.stdout.flush()
            classes[folders] = class_id
            class_id += 1
        # Finish the rest
        if count != 0:
            patch["data"] = patch["data"][0:count]
            patch["labels"] = patch["labels"][0:count]
            num_elements = np.shape(patch["data"])[0]
            indices = np.random.permutation(num_elements)
            patch["data"] = patch["data"][indices]
            patch["labels"] = patch["labels"][indices]
            count = 0
            output_filename = self.save_base_path + "/" + self.filename + '%d' % cur_batch
            pickle.dump(patch, open(output_filename, "wb"))

        # Print classes
        sorted_classes = dict(sorted(classes.items(), key=operator.itemgetter(1)))
        fp = open(self.save_base_path + "/" + "labels.txt", "w")
        for key in sorted_classes.keys():
            fp.write('%s\t%d\n' %(key, sorted_classes[key]))
        fp.close()

