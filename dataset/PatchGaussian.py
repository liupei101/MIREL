"""
Pytorch Dataset class that loads synthesis MIL dataset
"""
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data_utils
import sklearn.datasets


class GaussianBags(data_utils.Dataset):
    """
    There are 10 clusters in 2D plane. Each cluster contains Gaussian-distributed 2D points.
    - Some clusters (only one maybe) are selected as the cluster of interest, i.e., the origin of positive instances.
    - All the remaining clusters are negative ones, i.e., the origin of negative instances.
    """
    def __init__(self, target_number=1, mean_bag_length=10, var_bag_length=1, mean_pos_ratio=0.5, num_bag=1000, seed=7, 
            data_name='train', label_allowed_in_train=None, ood_origin=None, ood_ins_ratio=-1, ood_num_bag=None, **kwargs):
        assert data_name in ['train', 'val', 'test']
        self.target_number = target_number if isinstance(target_number, list) else [int(target_number)]
        self.target_number = list(set(self.target_number)) # get rid of duplicates
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.low_pos_ratio = 2 * mean_pos_ratio - 1
        self.num_bag = num_bag
        self.seed = seed
        self.data_name = data_name

        self.r = np.random.RandomState(seed)

        self.data_ID = {"gaussian": {"num_classes": 3}}
        self.name_ID = 'gaussian'
        self.num_classes = self.data_ID[self.name_ID]['num_classes']
        
        # OOD settings
        all_classes = [i for i in range(self.num_classes)]
        if label_allowed_in_train is None or len(label_allowed_in_train) == 0:
            self.label_ID = all_classes # all available numbers
            self.label_OOD = []
            # self.mean_ood_bag_length = 0
            # self.var_ood_bag_length = 0
        else:
            self.label_ID  = [i for i in label_allowed_in_train if i in all_classes] # e.g., [5, 6, 7, 8, 9]
            self.label_OOD = [i for i in all_classes if i not in self.label_ID]
            # self.mean_ood_bag_length = min(1, int(self.mean_bag_length * ood_ins_ratio))
            # self.var_ood_bag_length = 0
        self.ood_ins_ratio = ood_ins_ratio
        self.ood_num_bag = ood_num_bag if ood_num_bag is not None else num_bag

        if ood_origin is None:
            self.ood_origin = 'Gaussian'
        else:
            self.ood_origin = ood_origin
        
        for number in self.target_number:
            assert number in self.label_ID
        
        if self.data_name == 'train':
            self.train_bags_list, self.train_labels_list = self._form_bags()
            self.uid = ["train_{}".format(i) for i in range(self.num_bag)]
        elif self.data_name == 'val':
            self.val_bags_list, self.val_labels_list = self._form_bags()
            self.uid = ["val_{}".format(i) for i in range(self.num_bag)]
        elif self.data_name == 'test':
            self.test_bags_list, self.test_labels_list = self._form_bags()
            self.uid = ["test_{}".format(i) for i in range(self.num_bag + self.ood_num_bag)]
            
        print("[PatchGaussian] {}: num_bag={}, seed={}, data_name={}, target_number={}, low_pos_ratio={}, ID labels={}, ood_ins_ratio={}, ood_num_bag={}.".format(
            self.data_name, num_bag, seed, data_name, self.target_number, self.low_pos_ratio, self.label_ID, ood_ins_ratio, self.ood_num_bag))

    def _get_ID_data(self):
        if self.data_name in ['train', 'val', 'test']:
            # train and val will use different seeds to form bags
            size_points = 1000
            if self.name_ID == 'gaussian':
                assert self.num_classes == 3
                train_examples, train_labels = [], []
                centers = [[0, 1.5], [-1.5, -0.5], [1.5, -0.5]]
                for i, c in enumerate(centers):
                    c_points = self.r.multivariate_normal(
                        c, cov=np.diag((0.1, 0.1)), size=size_points
                    ) 
                    train_examples.append(c_points)
                    train_labels.append(np.ones(size_points) * i)
                train_examples = np.concatenate(train_examples, axis=0)
                train_labels = np.concatenate(train_labels, axis=0)
            else:
                pass
            points = train_examples # [N, 2]
            labels = torch.LongTensor(train_labels) # [N, ]
        else:
            pass
        # filter unallowed data
        sel_idx = np.isin(labels.numpy(), np.array(self.label_ID))
        sel_points = points[sel_idx]
        sel_labels = labels[sel_idx]
        print("[info] there are {} points reserved in the dataset after keeping the numbers {}".format(
               len(sel_idx), self.label_ID))
        
        return sel_points, sel_labels
    
    def _get_OOD_data(self):
        if self.ood_origin == 'Gaussian':
            size_points = 1000
            ood_points = self.r.multivariate_normal(
                (5, -5), cov=np.diag((0.1, 0.1)), size=size_points
            ) # [500, 2]
            ood_labels = torch.ones((size_points,)).to(torch.long)
        else:
            pass
        # we use neg. values (<= -1) to indicate OOD 
        ood_labels = -1 * (1 + ood_labels) 
        print("[info] there are {} OOD points from {}.".format(len(ood_points), self.ood_origin))
        return ood_points, ood_labels
    
    def _get_class_indices(self, given_labels):
        if_target_number = np.isin(given_labels.numpy(), np.array(self.target_number))
        pos_indices = np.where(if_target_number)[0]
        neg_indices = np.where(~if_target_number)[0]
        return pos_indices, neg_indices
    
    def _generate_subset_indices(self, given_indices, ret_length:int):
        indices = self.r.choice(given_indices, ret_length, replace=False)
        return indices
    
    def _form_bags(self):
        bags_list, labels_list = [], []
        
        if self.data_name in ['train', 'val']:
            num_ID_bag = self.num_bag
        else:
            num_ID_bag = self.num_bag + self.ood_num_bag # last ones will be replaced later
        
        # generate ID bags
        all_points, all_labels = self._get_ID_data()
        pos_indices, neg_indices = self._get_class_indices(all_labels)
        for i in range(num_ID_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            bag_length = max(1, bag_length)
            if i % 2 == 0:
                pos_ratio = self.r.uniform(self.low_pos_ratio, 1.0) # in [low_pos_ratio, 1)
                num_pos = max(1, int(bag_length * pos_ratio)) # in [1, bag_length), at least one neg instance
                assert num_pos < bag_length
                sel_idx_pos = self._generate_subset_indices(pos_indices, num_pos)
                sel_idx_neg = self._generate_subset_indices(neg_indices, bag_length - num_pos)
                sel_idx = np.concatenate([sel_idx_pos, sel_idx_neg])
                sel_idx = sel_idx[self.r.permutation(bag_length)]
            else:
                sel_idx = self._generate_subset_indices(neg_indices, bag_length)
            bags_list.append(all_points[sel_idx]) # fetch points
            labels_list.append(all_labels[sel_idx])
        
        if self.data_name in ['train', 'val']:
            print("[info] A total of {} bags is generated.".format(num_ID_bag))
            return bags_list, labels_list
        
        if abs(self.ood_ins_ratio) < 1e-5:
            print("[warning] ood_ins_ratio is set to 0, so skipping the OOD replacement process and returing all ID bags.")
            print(f"[info] A total of {num_ID_bag} bags is generated; the bags with OOD instances count 0.")
            return bags_list, labels_list
            
        # generate OOD bags (or partial OOD) for test by replacing the last `self.ood_num_bag` bags in `bags_list`
        ood_points, ood_labels = self._get_OOD_data()
        ood_indices = np.array([_ for _ in range(len(ood_labels))])
        for i in range(self.ood_num_bag):
            cur_idx = num_ID_bag - 1 - i
            last_bag = bags_list[cur_idx]
            last_label = labels_list[cur_idx]
            size_last_bag = len(last_bag)

            # ood_bag_length \in [1, len(last_bag)]
            ood_ins_ratio  = self.r.rand() if self.ood_ins_ratio < 0 or self.ood_ins_ratio > 1 else self.ood_ins_ratio
            ood_bag_length = max(1, int(self.ood_ins_ratio * size_last_bag))

            cur_ood_indices = self._generate_subset_indices(ood_indices, ood_bag_length)

            replace_indices = self.r.permutation(size_last_bag)[:ood_bag_length]
            last_bag[replace_indices] = ood_points[cur_ood_indices] # replace ID points with OOD ones
            last_label[replace_indices] = ood_labels[cur_ood_indices] # Neg label values to indicate OOD instances

            bags_list[cur_idx] = last_bag
            labels_list[cur_idx] = last_label
            
        print(f"[info] A total of {num_ID_bag} bags is generated; the bags with OOD instances count {self.ood_num_bag}.")

        return bags_list, labels_list 
    
    def _has_target_number(self, given_labels):
        for number in self.target_number:
            if number in given_labels:
                return True
        return False
    
    def __len__(self):
        if self.data_name == 'train':
            return len(self.train_labels_list)
        elif self.data_name == 'val':
            return len(self.val_labels_list)
        elif self.data_name == 'test':
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.data_name == 'train':
            bag = self.train_bags_list[index]
            label = self.train_labels_list[index]
        elif self.data_name == 'val':
            bag = self.val_bags_list[index]
            label = self.val_labels_list[index]
        elif self.data_name == 'test':
            bag = self.test_bags_list[index]
            label = self.test_labels_list[index]

        tensor_bag = torch.FloatTensor(bag) # [N, 2]
        coord = torch.IntTensor([i for i in range(len(label))])
        bag_label = torch.LongTensor([1 if self._has_target_number(label) else 0])

        return index, (tensor_bag, coord), (bag_label, label)
