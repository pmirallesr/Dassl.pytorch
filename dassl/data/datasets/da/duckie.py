import random
import os.path as osp
import linecache
from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


ep_length = {f'episode_{i:2d}':501 for i in range(21)}
ep_length['episode_09'] = 448
ep_length['episode_20'] = 38
total_data = 0
for i in ep_length.values():
    total_data += i

def read_duckie_image_list(im_dir, domain, n, n_repeat=None):
    items = []
    total_imgs = 0
    for file_name in sorted(listdir_nohidden(im_dir)):
        if file_name.startswith('episode_'):
            n_ep = int(file_name[-2:])
            ep_dir = osp.join(im_dir, file_name)
            annotations_filename = osp.join(ep_dir,'annotation.txt')
            for n_im, imname in enumerate(sorted(listdir_nohidden(ep_dir))):
                if n_im + total_imgs >= n: # If all required images have been processed
                    if n_repeat is not None:
                        items *= n_repeat
                    return items
                imname_noext, ext = osp.splitext(imname)
                if ext == '.txt':
                    continue
                impath = osp.join(ep_dir, imname)
                # Retrieve label
                im_number = int(imname_noext.split('_')[-1]) #im_number better than n_im.
                # Line N corresponds to image N-1, clean newline
                ann_line = linecache.getline(annotations_filename, im_number + 1)[:-1]
                # getline returns '' when it fails to find the file
                if ann_line == '':
                    raise ValueError(f"The file {annotations_filename} does not exist")
                label_texts = ann_line.split(' ')[1:]
                assert len(label_texts) == 2
                labels = (float(label_texts[0]), float(label_texts[1]))
                items.append((impath, labels))
            total_imgs += ep_length[f'episode_{n_ep:2d}']
    if n_repeat is not None:
        items *= n_repeat
    return items

@DATASET_REGISTRY.register()
class Duckie(DatasetBase):
    """
    """
    version = '1.0.0'
    dataset_dir = f'duckie/{version}'


    def __init__(self, cfg, train=0.8, test=0.1):
        assert train + test <= 1.0
        self.input_domains = ['base_small', 'colors', 'shapes', 'textures', 'blurred', 'colors2', 'shapes2', 'textures2', 'blurred2']
        self.target_domains = ['base_small', 'colors', 'shapes', 'textures', 'blurred', 'colors2', 'shapes2', 'textures2', 'blurred2', 'real_small']
        self.domains = self.target_domains
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        
        nb_train = 8960
        nb_test = 1040
        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, n=nb_train)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, n=nb_train)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, n=nb_test)

        super().__init__(train_x=train_x, train_u=train_u, test=test, outputs=2)
    
    
    def _read_data(self, input_domains, n):
        items = [] # Datum list
        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            items_d = read_duckie_image_list(domain_dir, dname, n) # (path, label) list
            for impath, label in items_d:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)
        return items
