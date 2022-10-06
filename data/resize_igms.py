from glob import glob
import os
from multiprocessing import Pool

from PIL import Image

max_sizviz_utils.pye = 1024

def main_worker(fn):
    split_fn = fn.split('/')
    output_fn = f'/home/amir/Datasets/arxiv_resized/{split_fn[5]}/{split_fn[6]}'
    if os.path.exists(output_fn):
        try:
            Image.open(output_fn).convert('RGB')
            return
        except Exception:
            pass

    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    img = Image.open(fn).convert('RGB')
    img.thumbnail((max_size, max_size))
    img.save(output_fn)

if __name__ == "__main__":
    import time
    fns = glob('/home/amir/Datasets/arxiv/*/*.png')
    with Pool(55) as p:
        p.map(main_worker, fns)
