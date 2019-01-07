from src.Dataset import DataSet
from os import listdir
import pickle
import cv2
import numpy as np
import operator

class ImageNet(DataSet):
    def __init__(self, config):
        super().__init__(source_path=config.source_path,
                         save_base_path=config.save_base_path,
                         filename=config.filename,
                         num_batches=config.num_batches)

    def process(self):
        # Calculating how many samples do we have
        classes = listdir(self.source_path)
        tot_files = 0
        for folders in listdir(self.source_path):
            tot_files += len(listdir(self.source_path + "/" + folders + "/images/"))
        files_per_batch = int(tot_files / self.num_batches)

        patch = {}
        patch["data"] = None
        patch["labels"] = None

        class_id = 0
        classes = {}
        cur_batch = 1
        count = 0
        for folders in listdir(self.source_path):
            dst_dir = self.source_path + "/" + folders + "/images/"
            for files in listdir(dst_dir):
                img = np.asarray(cv2.imread(folders + files))
                if patch["labels"] is None:
                    patch["data"] = img
                    patch["labels"] = np.asarray([class_id])
                else:
                    patch["data"] = np.concatenate([patch["data"], [img]], axis=0)
                    patch["labels"] = np.concatenate([patch["labels"], [class_id]])
                count += 1
                if count == files_per_batch:
                    count = 0
                    output_filename = self.save_base_path + "/" + self.filename + '%d' %cur_batch
                    pickle.dump(patch, open(output_filename, "wb"))
                    del patch["data"]
                    del patch["labels"]
                    del patch
                    patch = {}
                    patch["data"] = None
                    patch["labels"] = None
                    cur_batch += 1
            classes[folders] = class_id
            class_id += 1

        sorted_classes = dict(sorted(classes.items(), key=operator.itemgetter(1)))
        fp = open(self.save_base_path + "/" + "labels.txt")
        for key in sorted_classes.keys():
            fp.write('%s:%d\n' %(key, sorted_classes[key]))
        fp.close()

