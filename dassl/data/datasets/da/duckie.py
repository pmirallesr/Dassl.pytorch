import random
import os
import os.path as osp
from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


total_data = 10000

def read_duckie_image_list(path, domain, n, num_stack=4, n_repeat=None):
        
    #Get Images
    items: List[Tuple[List[Path], Tuple[float, float]]] = []
    c = 0 #Image counter
    for episode in sorted(os.listdir(path)):
        ep_path = osp.join(path, episode)
        #Skip file if needed
        if episode.endswith('.yaml') or episode.endswith('.png'):
            continue
        if c <= n:
            with open(osp.join(ep_path, "annotation.txt"), "r") as fp:
                annotations = [line.split() for line in fp.readlines()]
            for idx, ann in enumerate(annotations):
                if c <= n:
                    indexes = [max(0, i) for i in range(idx - num_stack + 1, idx + 1)]
                    images = [osp.join(ep_path,annotations[i][0]) for i in indexes]
                    actions = float(ann[1]), float(ann[2])
                    items.append((images, actions))
                c += 1

    if n_repeat is not None:
        items *= n_repeat
    return items

@DATASET_REGISTRY.register()
class Duckie(DatasetBase):

    def __init__(self, cfg, nb_train=8960, nb_test=1040):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        version = cfg.DATASET.VERSION
        self.dataset_dir = f'duckie/{version}'
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.domains = ['base_small', 'colors', 'colors2', 'shapes', 'shapes2', 'blurred', 'blurred2', 'textures', 'textures2','combined', 'combined2', 'real_small', 'extended_base_small', 'patchy_base_small', 'objects_base_small', 'empty_base_small']
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        
        nb_train = nb_train
        nb_test = nb_test
        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, n=nb_train, num_stack=cfg.DATASET.NUM_STACK, n_repeat = None)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, n=nb_train, num_stack=cfg.DATASET.NUM_STACK, n_repeat = None)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, n=nb_test, num_stack=cfg.DATASET.NUM_STACK, n_repeat = None)

        super().__init__(train_x=train_x, train_u=train_u, test=test, outputs=2)
    
    
    def _read_data(self, input_domains, n, num_stack, n_repeat):
        items = [] # Datum list
        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            items_d = read_duckie_image_list(domain_dir, dname, n, num_stack, n_repeat) # ([paths], label) list
            for impath, label in items_d:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)
        return items
