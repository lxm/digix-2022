import os
import sys
import shutil
import pathlib
import json
from PIL import Image
from threading import Thread
from threading import Semaphore

crop_param = (0, 300, 1080, 2160)
sem = Semaphore(16)
def crop_img(src, dst, fname):
    if os.path.exists(dst + fname) and os.path.getsize(dst + fname) > 0:
        print("file " + fname +  "exist and non empty, return")
        sem.release()
        return
    print("file " + fname +  "\tstart process")
    img = Image.open(src)
    img = img.resize((1080, 2400))
    cropped = img.crop(crop_param)
    pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    target = cropped.resize((1080, 1080)) #480,240 (256, 256)
    target.save(dst + fname)
    sem.release()

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    target_dir = sys.argv[2]

    dataset_all = os.walk(dataset_dir)
    
    for path, dir_list, file_list in dataset_all:
        for f_name in file_list:
            sem.acquire()
            full_path = (dataset_dir + "/" +f_name)
            target_path = target_dir + "/"
            Thread(target=crop_img, args=(full_path, target_path, f_name)).start()
