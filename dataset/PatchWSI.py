"""
Pytorch Dataset class that loads WSI (pathology image) MIL dataset.
"""
from typing import Union
import os.path as osp
import torch
import numpy as np
import random
from torch.utils.data import Dataset

from utils.io import retrieve_from_table_clf
from utils.io import read_patch_data, read_patch_coord
from utils.func import sampling_data, random_mask_instance


class WSIPatchClf(Dataset):
    r"""A patch dataset class for classification tasks (slide-level in general).
    
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        label_path (string): The path of patch-level labels, non-existing if it is None.
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, table_path: str, label_path:Union[None,str]=None,
        read_format:str='pt', ratio_sampling:Union[None,float,int]=None, ratio_mask=None, mode='patch', **kws):
        patient_ids = list(set(patient_ids)) # avoid possible duplicates
        super(WSIPatchClf, self).__init__()
        if ratio_sampling is not None:
            assert ratio_sampling > 0 and ratio_sampling < 1.0
            print("[dataset] Patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))
        if ratio_mask is not None and ratio_mask > 1e-5:
            assert ratio_mask <= 1, 'The argument ratio_mask must be not greater than 1.'
            assert mode == 'patch', 'Only support a patch mode for instance masking.'
            self.ratio_mask = ratio_mask
            print("[dataset] Masking instances with ratio_mask = {}".format(ratio_mask))
        else:
            self.ratio_mask = None

        self.read_path = patch_path
        self.label_path = label_path
        self.has_patch_label = (label_path is not None) and len(label_path) > 0
        
        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table_clf(
            patient_ids, table_path, ret=info, level='slide')
        self.uid = self.sids
        
        assert mode in ['patch', 'cluster', 'graph']
        self.mode = mode
        self.read_format = read_format
        self.kws = kws
        self.SUFFIX_OOD = "OOD_"
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        if self.mode == 'graph':
            assert 'graph_path' in kws

        self.prepare_ood_data()
        self.summary()

    def prepare_ood_data(self):
        if 'ood_origin' in self.kws and self.kws['ood_origin'] is not None:
            self.ood_origin = self.kws['ood_origin']
        else:
            self.ood_origin = None
            return False

        # For distribution-shifted datasets: C16-Lighter, C16-Light, and C16-Strong
        if 'c16_synth' in self.ood_origin:
            shift_severity = self.ood_origin.split("c16_synth_")[-1]
            assert shift_severity in ['lighter', 'light', 'strong']
            print("[info] Domain shift is set to {} for `c16_synth` dataset.".format(shift_severity))

            self.ood_read_path = [
                f'/NAS02/ExpData/camelyon16/feats-RN18-SimCL-Test-set-tiles-l1-s256-synth-shift/blur_{shift_severity}/pt_files',
                f'/NAS02/ExpData/camelyon16/feats-RN18-SimCL-Test-set-tiles-l1-s256-synth-shift/hed_{shift_severity}/pt_files',
            ]
            self.ood_read_format = 'pt'
            self.ood_coord_path = '/NAS02/ExpData/camelyon16/tiles-l1-s256/patches'
            self.ood_table_path = '/NAS02/ExpData/camelyon16/table/camelyon16_testset_insclf_label.csv'
        # For OOD dataset: TCGA-PRAD
        elif self.ood_origin == 'prostate':
            self.ood_read_path = '/NAS02/ExpData/tcga_prad/feats-RN18-SimCL-tiles-l1-s256-ood-det/pt_files'
            self.ood_read_format = 'pt'
            self.ood_coord_path = '/NAS02/ExpData/tcga_prad/tiles-l1-s256/patches'
            self.ood_table_path = '/NAS02/ExpData/tcga_prad/table/tcga_prad_slide.csv'
        else:
            raise ValueError(f"OOD dataset {self.ood_origin} is not expected.")

        info = ['sid', 'sid2pid', 'sid2label']
        self.ood_sids, self.ood_sid2pid, self.ood_sid2label = retrieve_from_table_clf(
            None, self.ood_table_path, ret=info, level='slide')

        # append the OOD data which will be loaded
        for sid in self.ood_sids:
            new_sid = self.SUFFIX_OOD + sid # avoid the same ID
            assert new_sid not in self.sid2pid
            assert new_sid not in self.sid2label
            self.sids.append(new_sid)
            self.sid2pid[new_sid] = self.ood_sid2pid[sid]
            self.sid2label[new_sid] = -1 * (1 + self.ood_sid2label[sid]) # neg. value to indicate OOD
        self.uid = self.sids

        print(f"[info] finished the preparation for {len(self.ood_sids)} OOD samples from {self.ood_origin}")
        return True

    def summary(self):
        print(f"Dataset WSIPatchClf for {self.mode}: avaiable WSIs count {self.__len__()}")
        if not self.has_patch_label:
            print("[info] the patch-level label is not avaiable, derived by slide label.")
        else:
            print("[info] the patch-level label is read from {}".format(self.label_path))

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid]

        ood_indicator = True if label < 0 and self.ood_origin is not None else False
        if ood_indicator:
            if isinstance(self.ood_read_path, list):
                cur_path_patch = self.ood_read_path[random.randint(0, len(self.ood_read_path) - 1)]
            else:
                cur_path_patch = self.ood_read_path
            cur_read_format = self.ood_read_format
            cur_path_coord  = self.ood_coord_path
            has_patch_label = False
            cur_path_label  = None
            slide_name = sid[len(self.SUFFIX_OOD):]
        else:
            cur_path_patch  = self.read_path
            cur_read_format = self.read_format
            cur_path_coord  = self.kws['coord_path']
            has_patch_label = self.has_patch_label
            cur_path_label  = self.label_path
            slide_name = sid

        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        if self.mode == 'patch':
            full_path = osp.join(cur_path_patch, slide_name + '.' + cur_read_format)
            feats = read_patch_data(full_path, dtype='torch').to(torch.float)
            # if masking patches
            if self.ratio_mask:
                feats = random_mask_instance(feats, self.ratio_mask, scale=1, mask_way='mask_zero')
            full_coord = osp.join(cur_path_coord, slide_name + '.h5')
            coors = read_patch_coord(full_coord, dtype='torch')
            if has_patch_label:
                path = osp.join(cur_path_label, slide_name + '.npy')
                patch_label = read_patch_data(path, dtype='torch', key='label').to(torch.float)
            else:
                patch_label = label * torch.ones(feats.shape[0]).to(torch.float)
            assert patch_label.shape[0] == feats.shape[0]
            assert coors.shape[0] == feats.shape[0]
            return index, (feats, coors), (label, patch_label)
        else:
            pass
            return None
