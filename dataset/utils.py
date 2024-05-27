from dataset.PatchWSI import WSIPatchClf
from dataset.PatchMNIST import MnistBags
from dataset.PatchCIFAR import CifarBags
from dataset.PatchGaussian import GaussianBags

from utils.func import parse_str_dims


def prepare_clf_dataset(origin:str, cfg, **kws):
    """
    Interface for preparing MIL datasets.
    
    origin: 'mnist' or 'wsi'.
    cfg: a dict where 'path_patch', 'path_table', and 'feat_format' are included.
    """
    if origin == 'wsi':
        path_patch = cfg['path_patch']
        path_table = cfg['path_table']
        feat_format = cfg['feat_format']
        if 'path_label' in cfg:
            path_label = cfg['path_label']
        else:
            path_label = None
        if 'ratio_sampling' in kws:
            ratio_sampling = kws['ratio_sampling']
        else:
            ratio_sampling = None
        if 'ratio_mask' in kws:
            if cfg['test']: # only used in a test mode
                ratio_mask = kws['ratio_mask']
            else:
                ratio_mask = None
        else:
            ratio_mask = None
        if 'filter_slide' in kws:
            if_remove_slide = kws['filter_slide']
        else:
            if_remove_slide = None

        if 'ood_origin' in kws:
            ood_origin = kws['ood_origin']
        else:
            ood_origin = None
        
        patient_ids = kws['patient_ids']
        dataset = WSIPatchClf(
            patient_ids, path_patch, path_table, label_path=path_label, read_format=feat_format, ratio_sampling=ratio_sampling, 
            ratio_mask=ratio_mask, coord_path=cfg['path_coord'], filter_slide=if_remove_slide, ood_origin=ood_origin, 
        )
    elif origin in ['mnist', 'cifar10', 'gau']:
        name_prefix = origin
        if cfg[name_prefix + '_id_labels'] is None or len(cfg[name_prefix + '_id_labels']) == 0:
            id_labels = None
        else:
            id_labels = parse_str_dims(cfg[name_prefix + '_id_labels'])
        
        if cfg[name_prefix + '_target_number'] is None:
            target_labels = None
        else:
            target_labels = parse_str_dims(cfg[name_prefix + '_target_number'])

        if name_prefix == 'mnist':
            BagDataset = MnistBags
        elif name_prefix == 'cifar10':
            BagDataset = CifarBags
        elif name_prefix == 'gau':
            BagDataset = GaussianBags
        else:
            BagDataset = None
         
        dataset = BagDataset(
            target_number=target_labels,
            mean_bag_length=cfg[name_prefix + '_mean_bag_length'],
            var_bag_length=cfg[name_prefix + '_var_bag_length'],
            mean_pos_ratio=cfg[name_prefix + '_mean_pos_ratio'],
            num_bag=kws['data_size'],
            seed=kws['data_seed'],
            data_name=kws['data_name'],
            label_allowed_in_train=id_labels,
            ood_origin=cfg[name_prefix + '_ood_origin'],
            ood_ins_ratio=cfg[name_prefix + '_ood_ratio']
        )
    return dataset
