import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from types import SimpleNamespace
import wandb
from functools import partial

from .model_utils import load_model, general_init_weight
from loss.utils import load_loss, loss_reg_l1
from loss.utils import load_edl_neg_instance_loss
from eval.utils import load_evaluator, load_uncertainty_evaluator
from dataset.utils import prepare_clf_dataset
from optim import create_optimizer

from utils.func import seed_everything, parse_str_dims, print_metrics
from utils.func import add_prefix_to_filename, rename_keys, rename_unc_keys
from utils.func import fetch_kws, print_config, EarlyStopping
from utils.func import seed_generator, seed_worker
from utils.func import convert_instance_output, has_no_ood_instance
from utils.func import to_instance_label, quick_ood_test, quick_IFB_test
from utils.io import read_datasplit_npz, read_maxt_from_table
from utils.io import save_prediction_clf, save_prediction_clf_ins


# Active quick OOD test, i.e., skipping the model training 
# when testing the same model on another OOD datasets
ENABLE_QUICK_OOD_TEST = True

# Active quick IFB test, i.e., skipping the model training 
# when testing the same model by obtaining instance prediction from the bag-level network
ENABLE_QUICK_IFB_TEST = True


class ClfHandler(object):
    """
    Handling the initialization, training, and testing 
    of general MIL-based classification models.
    """
    def __init__(self, cfg):
        # check args
        assert cfg['task'] == 'clf', 'Task must be clf.'

        torch.cuda.set_device(cfg['cuda_id'])
        seed_everything(cfg['seed'])

        # path setup
        if cfg['test']: # if in a test mode
            cfg['test_save_path'] = cfg['test_save_path'].format(cfg['data_split_seed'])
            cfg['test_load_path'] = cfg['test_load_path'].format(cfg['data_split_seed'])
            if not osp.exists(cfg['test_save_path']):
                os.makedirs(cfg['test_save_path'])
            run_name = cfg['test_save_path'].split('/')[-1]
            self.last_ckpt_path = osp.join(cfg['test_load_path'], 'model-last.pth')
            self.best_ckpt_path = osp.join(cfg['test_load_path'], 'model-best.pth')
            self.last_metrics_path = osp.join(cfg['test_save_path'], 'metrics-last.txt')
            self.best_metrics_path = osp.join(cfg['test_save_path'], 'metrics-best.txt')
            self.config_path = osp.join(cfg['test_save_path'], 'print_config.txt')
            # wandb writter
            wandb.init(project=cfg['test_wandb_prj'], name=run_name, dir=cfg['wandb_dir'], config=cfg, reinit=True)
        else:
            if not osp.exists(cfg['save_path']):
                os.makedirs(cfg['save_path'])
            run_name = cfg['save_path'].split('/')[-1]
            self.last_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
            self.best_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
            self.last_metrics_path = osp.join(cfg['save_path'], 'metrics-last.txt')
            self.best_metrics_path = osp.join(cfg['save_path'], 'metrics-best.txt')
            self.config_path = osp.join(cfg['save_path'], 'print_config.txt')
            # wandb writter
            wandb.init(project=cfg['wandb_prj'], name=run_name, dir=cfg['wandb_dir'], config=cfg, reinit=True)

        # model setup
        dims = parse_str_dims(cfg['net_dims'])
        self.net = load_model(
            cfg['task'], cfg['backbone'], dims, drop_rate=cfg['drop_rate'], use_feat_proj=cfg['use_feat_proj'], 
            edl_output=cfg['loss_edl'], edl_evidence_func=cfg['edl_evidence_func'], edl_pred_head=cfg['edl_pred_head'], 
            pooling=cfg['abmil_pooling'], ins_pred_from=cfg['ins_pred_from'], eins_frozen_feat=cfg['eins_frozen_feat'] 
        )
        if hasattr(self.net, 'relocate'):
            self.net.relocate()
        else:
            self.net = self.net.cuda()

        if cfg['init_wt']:
            self.net.apply(general_init_weight)

        # loss setup
        kws_loss = fetch_kws(cfg, prefix='loss')
        self.loss = load_loss(cfg['task'], **kws_loss)
        
        if kws_loss['bce']:
            if dims[-1] != 2:
                print("[warning] there is a conflit between the configs 'bce_loss' and 'net_dims'.")
        else:
            assert dims[-1] > 2, "conflit between the configs 'bce_loss' and 'net_dims'."
        
        if kws_loss['edl']:
            self.loss = partial(self.loss, 
                num_classes=dims[-1], annealing_step=kws_loss['annealing_steps'], 
                c_fisher=kws_loss['mse_fisher_coef'], use_kl_div=kws_loss['use_kl_div'], 
                red_type=kws_loss['red_type'], branch='bag'
            )

            if cfg['edl_evidence_sum']:
                self.aux_loss = load_loss(cfg['task'], edl=True, edl_type=kws_loss['edl_type']) 
                
                self.coef_aux_loss = cfg['loss_aux_coef']
                self.aux_separate = cfg['edl_evidence_sum_separate']
                self.aux_loss = partial(self.aux_loss, 
                    num_classes=dims[-1], annealing_step=kws_loss['annealing_steps'], 
                    c_fisher=kws_loss['mse_fisher_coef'], use_kl_div=kws_loss['use_kl_div'], 
                    red_type=kws_loss['ins_red_type'], branch='instance', separate=self.aux_separate, 
                    aggregate=cfg['edl_evidence_sum_aggregate']
                )
            else:
                self.coef_aux_loss = 0
                self.aux_loss = None
                self.aux_separate = None

        # optimizer and lr_scheduler
        cfg_optimizer = SimpleNamespace(opt=cfg['opt_name'], weight_decay=cfg['opt_weight_decay'], lr=cfg['opt_lr'], 
            opt_eps=None, opt_betas=None, momentum=None)
        self.optimizer = create_optimizer(cfg_optimizer, self.net)
        # LR scheduler
        kws_lrs = fetch_kws(cfg, prefix='lrs')
        self.steplr = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
            factor=kws_lrs['factor'], patience=kws_lrs['patience'], verbose=True)

        # evaluator (bag)
        self.evaluator = load_evaluator(cfg['task'], 'bag', binary_clf=cfg['loss_bce'], edl=cfg['loss_edl'], dataset=cfg['dataset_origin'])
        if cfg['loss_bce']: # binary
            self.metrics_list = ['auc', 'loss', 'acc@mid', 'recall@mid', 'precision@mid', 'f1_score@mid', 'ece', 'mce']
        else: # multi-class
            self.metrics_list = None # ['auc', 'loss', 'acc', 'macro_f1_score', 'micro_f1_score']
        self.ret_metrics = ['auc', 'acc@mid', 'loss']

        # evaluator (instance)
        ins_edl = (cfg['ins_pred_from'] == 'ins' and cfg['abmil_pooling'] == 'edl') \
                    or (cfg['ins_pred_from'] in ['bag', 'eins'] and cfg['loss_edl'])
        self.ins_evaluator = load_evaluator(cfg['task'], 'instance', binary_clf=cfg['loss_bce'], edl=ins_edl, dataset=cfg['dataset_origin'])
        if cfg['loss_bce']: # binary
            self.ins_metrics_list = ['auc', 'loss', 'loss_soft', 'loss_soft_pos', 'loss_soft_neg', 'loss_ID', 'loss_OOD', 
                'acc@mid', 'recall@mid', 'precision@mid', 'f1_score@mid', 'ece', 'mce']
        else: # multi-class
            self.ins_metrics_list = None # ['auc', 'loss', 'acc', 'macro_f1_score', 'micro_f1_score']
        self.ret_ins_metrics = ['auc', 'acc@mid', 'loss']
        
        # uncertainty evaluator (bag)
        self.unc_evaluator = load_uncertainty_evaluator(cfg['task'], 'bag', binary_clf=cfg['loss_bce'], edl=cfg['loss_edl'], dataset=cfg['dataset_origin'])        
        # uncertainty evaluator (instance)
        self.unc_ins_evaluator = load_uncertainty_evaluator(cfg['task'], 'instance', binary_clf=cfg['loss_bce'], edl=ins_edl, dataset=cfg['dataset_origin'])
        
        self.task = cfg['task']
        self.bin_clf = cfg['loss_bce']
        self.edl_clf = cfg['loss_edl']
        self.ins_pred = True
        self.backbone = cfg['backbone']
        self.uid = dict()
        self.pseb_ind = dict()
        self.cfg = cfg
        print_config(cfg, print_to_path=self.config_path)
    
    def exec(self):
        print('[exec] setting: task = {}, backbone = {}.'.format(self.task, self.backbone))
        
        if self.cfg['dataset_origin'] == 'wsi':
            # Prepare data spliting 
            if "{}" in self.cfg['data_split_path']:
                if 'data_split_fold' in self.cfg:
                    path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'], self.cfg['data_split_fold'])
                else:
                    path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'])
            else:
                path_split = self.cfg['data_split_path']
            pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
            print('[exec] finished reading patient IDs from {}'.format(path_split))

            # Prepare datasets 
            train_set  = prepare_clf_dataset('wsi', self.cfg, patient_ids=pids_train)
            self.uid.update({'train': train_set.uid})
            val_set    = prepare_clf_dataset('wsi', self.cfg, patient_ids=pids_val)
            self.uid.update({'validation': val_set.uid})
            train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=True,  worker_init_fn=seed_worker, collate_fn=default_collate
            )
            val_loader = DataLoader(val_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
            )
            if pids_test is not None:
                test_set = prepare_clf_dataset('wsi', self.cfg, patient_ids=pids_test, ood_origin=self.cfg['wsi_ood_origin'])
                self.uid.update({'test': test_set.uid})
                test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
                    generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                    shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
                )
            else:
                test_set = None
                test_loader = None
        elif self.cfg['dataset_origin'] in ['mnist', 'cifar10', 'gau']:
            name_prefix = self.cfg['dataset_origin']
            # load the configuration of dataset size
            size_train, size_val, size_test = parse_str_dims(self.cfg[name_prefix + '_dataset_size'])
            
            # Prepare datasets 
            train_set  = prepare_clf_dataset(name_prefix, self.cfg, data_name='train', data_size=size_train, data_seed=98)
            self.uid.update({'train': train_set.uid})
            val_set    = prepare_clf_dataset(name_prefix, self.cfg, data_name='val', data_size=size_val, data_seed=97)
            self.uid.update({'validation': val_set.uid})
            train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=True,  worker_init_fn=seed_worker, collate_fn=default_collate
            )
            val_loader = DataLoader(val_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
            )
            test_set    = prepare_clf_dataset(name_prefix, self.cfg, data_name='test', data_size=size_test, data_seed=96)
            self.uid.update({'test': test_set.uid})
            test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
            )
            
        run_name = 'train'
        # Train
        if (
            ('force_to_skip_training' in self.cfg and self.cfg['force_to_skip_training']) or 
            (ENABLE_QUICK_OOD_TEST and quick_ood_test(self.cfg)) or 
            (ENABLE_QUICK_IFB_TEST and quick_IFB_test(self.cfg))
        ):
            print(f"[warning] your training for {self.cfg['save_path'].split('/')[-1]} is skipped...")
        else:
            val_name = 'validation'
            val_loaders = {'validation': val_loader, 'test': test_loader}
            self._run_training(self.cfg['epochs'], train_loader, 'train', val_loaders=val_loaders, val_name=val_name, 
                measure_training_set=True, save_ckpt=True, early_stop=True, run_name=run_name)

        # Evals using the best ckpt
        evals_loader = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
        metrics = self._eval_all(evals_loader, ckpt_type='best', run_name=run_name, if_print=True)
        
        return metrics

    def exec_test(self):
        print('[exec] test under task = {}, backbone = {}.'.format(self.task, self.backbone))
        mode_name = 'test_mode'
        
        # Prepare datasets 
        path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        if self.cfg['test_path'] == 'train':
            pids = pids_train
        elif self.cfg['test_path'] == 'val':
            pids = pids_val
        elif self.cfg['test_path'] == 'test':
            pids = pids_test
        else:
            pass
        print('[exec] test patient IDs from {}'.format(self.cfg['test_path']))

        # Prepare datasets 
        test_set = prepare_clf_dataset(pids, self.cfg, ratio_mask=self.cfg['test_mask_ratio'])
        self.uid.update({'exec-test': test_set.uid})
        test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
        )

        # Evals
        evals_loader = {'exec-test': test_loader}
        metrics = self._eval_all(evals_loader, ckpt_type='best', if_print=True, test_mode=True, test_mode_name=mode_name)
        return metrics

    def _run_training(self, epochs, train_loader, name_loader, val_loaders=None, val_name=None, 
        measure_training_set=True, save_ckpt=True, early_stop=False, run_name='train', **kws):
        """Traing model.

        Args:
            epochs (int): Epochs to run.
            train_loader ('DataLoader'): DatasetLoader of training set.
            name_loader (string): name of train_loader, used for infering patient IDs.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, which gives the datasets
                to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            measure_training_set (bool): If measure training set at each epoch.
            save_ckpt (bool): If save models.
            early_stop (bool): If early stopping according to validation loss.
            run_name (string): Name of this training, which would be used as the prefixed name of ckpt files.
        """
        # setup early_stopping
        if early_stop and self.cfg['es_patience'] is not None:
            self.early_stop = EarlyStopping(
                warmup=self.cfg['es_warmup'], patience=self.cfg['es_patience'], 
                start_epoch=self.cfg['es_start_epoch'], verbose=self.cfg['es_verbose']
            )
        else:
            self.early_stop = None

        if val_name is not None and self.early_stop is not None:
            assert val_name in val_loaders.keys(), "Not specify a dataloader to enable early stopping."
            print("[{}] {} epochs w early stopping on {}.".format(run_name, epochs, val_name))
        else:
            print("[{}] {} epochs w/o early stopping.".format(run_name, epochs))
        
        # iterative training
        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch + 1
            train_cltor = self._train_each_epoch(epoch + 1, train_loader, name_loader)
            cur_name = name_loader

            if measure_training_set:
                for k_cltor, v_cltor in train_cltor.items():
                    self._eval_and_print(k_cltor, v_cltor, name=cur_name+'/'+k_cltor, at_epoch=epoch+1)

            # val/test
            early_stopping_metrics = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    if val_loaders[k] is None:
                        continue
                    val_cltor = self.test_model(self.net, val_loaders[k], loader_name=k)
                    for k_cltor, v_cltor in val_cltor.items():
                        met_auc, met_acc, met_loss = self._eval_and_print(k_cltor, v_cltor, name=k+'/'+k_cltor, at_epoch=epoch+1)
                        if k == val_name:
                            if k_cltor == 'pred': 
                                if self.cfg['monitor_metrics'] == 'auc':
                                    early_stopping_metrics = met_auc
                                elif self.cfg['monitor_metrics'] == 'loss':
                                    early_stopping_metrics = met_loss
                                elif self.cfg['monitor_metrics'] == 'error+loss':
                                    early_stopping_metrics = met_loss + (1 - met_acc)
                                else:
                                    early_stopping_metrics = met_loss
                        # uncertainty evaludation
                        self._eval_uncertainty_and_print(k_cltor, v_cltor, name=k+'/'+k_cltor, at_epoch=epoch+1)
            
            # early_stop using VAL_METRICS
            if early_stopping_metrics is not None and self.early_stop is not None:
                self.steplr.step(early_stopping_metrics)
                self.early_stop(epoch, early_stopping_metrics)
                if self.early_stop.save_ckpt():
                    self.save_model(epoch+1, ckpt_type='best', run_name=run_name)
                    print("[train] {} best model saved at epoch {}".format(run_name, epoch+1))
                if self.early_stop.stop():
                    break
        
        if save_ckpt:
            self.save_model(last_epoch, ckpt_type='last', run_name=run_name) # save models and optimizers
            print("[train] {} last model saved at epoch {}".format(run_name, last_epoch))

    def _train_each_epoch(self, idx_epoch, train_loader, name_loader):
        print("[train] train one epoch using train_loader={}".format(name_loader))
        self.net.train()
        bp_every_batch = self.cfg['bp_every_batch']
        idx_collector, x_collector, y_collector, ins_collector = [], [], [], []

        all_pred, all_gt = [], []
        all_ins_pred, all_ins_gt = [], []
        i_batch = 0
        for data_idx, data_x, data_y in train_loader:
            # data_x = (feats, coords) | data_y = (label_slide, label_patch)
            i_batch += 1
            # 1. read data (mini-batch)
            data_x = data_x[0] # only use the first item
            data_label = data_y[0] # [B, 1]
            patch_label = data_y[1] # [B, N]

            data_x = data_x.cuda()
            data_label = data_label.cuda()

            x_collector.append(data_x)
            y_collector.append(data_label)
            idx_collector.append(data_idx)
            ins_collector.append(patch_label)

            # in a mini-batch
            if i_batch % bp_every_batch == 0:
                # 1. update network
                cur_pred, cur_ins_pred = self._update_network(idx_epoch, i_batch, x_collector, y_collector, ins_collector)
                
                all_pred.append(cur_pred)
                all_gt.append(torch.cat(y_collector, dim=0).detach().cpu())
                
                # append the instance-level data in this batch
                all_ins_pred += cur_ins_pred 
                all_ins_gt += [ins_x.squeeze(0).tolist() for ins_x in ins_collector]

                # 2. reset mini-batch
                idx_collector, x_collector, y_collector, ins_collector = [], [], [], []
                torch.cuda.empty_cache()

        all_pred = torch.cat(all_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0) # [B, 1]
        all_gt = all_gt.squeeze(-1) # [B, ]

        train_cltor = dict()
        train_cltor['pred'] = {'y': all_gt, 'y_hat': all_pred}
        if self.ins_pred:
            # elements are in python list style
            train_cltor['pred_ins'] = {'y': all_ins_gt, 'y_hat': all_ins_pred}

        return train_cltor

    def _update_network(self, idx_epoch, i_batch, xs, ys, ys_ins):
        """
        Update network using one batch data
        """
        n_sample = len(xs)
        y_hat, y_hat_ins, weight_ins = [], [], []

        for i in range(n_sample):
            # [B, num_cls]
            logit_bag, ins_res = self.net(xs[i], ret_ins_res=True)
            y_hat.append(logit_bag)
            if isinstance(ins_res, tuple):
                init_ins_res, ins_res = ins_res
                weight_ins.append(init_ins_res)
            y_hat_ins.append(ins_res)

        # 3.1 zero gradients buffer
        self.optimizer.zero_grad()

        # 3.2 loss
        # loss of bag clf
        bag_preds = torch.cat(y_hat, dim=0) # [B, num_cls]
        bag_label = torch.cat(ys, dim=0).squeeze(-1) # [B, ]

        if self.cfg['loss_edl']:
            bag_label = bag_label.unsqueeze(-1) # [B, 1]
            clf_loss = self.loss(output_alpha=bag_preds, target=bag_label, epoch_num=idx_epoch)
            
            if self.aux_loss is not None:
                # ONLY used as a fully-supervised baseline for the comparison to WS counterparts
                if self.aux_separate == 'F': 
                    ins_label = [to_instance_label(x, self.cfg) for x in ys_ins] 
                else: # MIL 
                    ins_label = None
                
                weight = weight_ins if self.cfg['edl_evidence_sum_aggregate'] == 'diweight' else None
                clf_loss += self.coef_aux_loss * self.aux_loss(
                    output_alpha=y_hat_ins, target=bag_label, 
                    epoch_num=idx_epoch, ins_target=ins_label, weight=weight
                )
        else:
            clf_loss = self.loss(bag_preds, bag_label)

        print("[training {}-th epoch] {}-th batch: clf_loss = {:.6f}".format(idx_epoch, i_batch, clf_loss.item()))
        wandb.log({'train_batch/clf_loss': clf_loss.item()})

        # 3.3 backward gradients and update networks
        clf_loss.backward()
        self.optimizer.step()

        val_preds = bag_preds.detach().cpu()
        val_preds_ins = [ins_pred.detach().cpu().squeeze(0).tolist() for ins_pred in y_hat_ins]

        return val_preds, val_preds_ins

    def _eval_all(self, evals_loader, ckpt_type='best', run_name='train', task='bag_clf', if_print=True,
        test_mode=False, test_mode_name='test_mode'):
        """
        test_mode = True only if run self.exec_test(), indicating a test mode.
        """
        if test_mode:
            print('[warning] you are in test mode now.')
            ckpt_run_name = 'train'
            wandb_group_name = test_mode_name
            metrics_path_name = test_mode_name
            csv_prefix_name = test_mode_name
            save_pred_path = self.cfg['test_save_path']
        else:
            ckpt_run_name = run_name
            wandb_group_name = run_name
            metrics_path_name = run_name
            csv_prefix_name = run_name
            save_pred_path = self.cfg['save_path']
        
        if ckpt_type == 'best':
            ckpt_path = add_prefix_to_filename(self.best_ckpt_path, ckpt_run_name)
            wandb_group = 'bestckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.best_metrics_path, metrics_path_name)
            csv_name = '{}_{}_best'.format(task, csv_prefix_name)
        elif ckpt_type == 'last':
            ckpt_path = add_prefix_to_filename(self.last_ckpt_path, ckpt_run_name)
            wandb_group = 'lastckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.last_metrics_path, metrics_path_name)
            csv_name = '{}_{}_last'.format(task, csv_prefix_name)
        else:
            pass

        metrics = dict()
        for k, loader in evals_loader.items():
            if loader is None:
                continue
            metrics[k] = []

            cltor = self.test_model(self.net, loader, loader_name=k, ckpt_path=ckpt_path)
            k_cltor = 'pred'
            v_cltor = cltor[k_cltor]
            auc, acc, loss = self._eval_and_print(k_cltor, v_cltor, name='{}/{}/{}'.format(wandb_group, k, k_cltor))
            metrics[k].append(('auc_'+k_cltor, auc))
            metrics[k].append(('acc_'+k_cltor, acc))
            metrics[k].append(('loss_'+k_cltor, loss))
            # uncertainty evaludation (bag-level)
            self._eval_uncertainty_and_print(k_cltor, v_cltor, name='{}/{}/{}'.format(wandb_group, k, k_cltor))

            if self.cfg['save_prediction']:
                path_save_pred = osp.join(save_pred_path, '{}_BagClf_pred_{}.csv'.format(csv_name, k))
                uids = self._get_unique_id(k, v_cltor['idx'])
                save_prediction_clf(
                    uids, v_cltor['y'], v_cltor['y_hat'], path_save_pred, 
                    binary=self.bin_clf, edl_output=self.edl_clf
                )

            if self.ins_pred:
                k_cltor = 'pred_ins' # instance-level cltor, only for EDL models or the models with an instance branch
                assert k_cltor in cltor
                v_cltor = cltor[k_cltor] 
                # evaluation
                self._eval_and_print(k_cltor, v_cltor, name='{}/{}/{}'.format(wandb_group, k, k_cltor))
                # uncertainty evaludation (instance-level)
                self._eval_uncertainty_and_print(k_cltor, v_cltor, name='{}/{}/{}'.format(wandb_group, k, k_cltor))
                # save instance-level results
                path_save_pred = osp.join(save_pred_path, '{}_InsClf_pred_{}.csv'.format(csv_name, k))
                uids = self._get_unique_id(k, v_cltor['idx'])
                save_prediction_clf_ins(
                    uids, v_cltor['y'], v_cltor['y_hat'], path_save_pred, 
                    binary=self.bin_clf, edl_output=self.edl_clf
                )

        if if_print:
            print_metrics(metrics, print_to_path=print_path)

        return metrics

    def _eval_and_print(self, key, cltor, name='', ret_metrics=None, at_epoch=None):
        if 'ins' in key:
            ret_metrics = self.ret_ins_metrics
            eval_metrics = self.ins_metrics_list
            evaluator = self.ins_evaluator
            cltor = convert_instance_output(cltor, self.cfg)
        else:
            ret_metrics = self.ret_metrics
            eval_metrics = self.metrics_list
            evaluator = self.evaluator
        
        eval_results = evaluator.compute(cltor, eval_metrics)
        eval_results = rename_keys(eval_results, name, sep='/')

        print("[{}] At epoch {}:".format(name, at_epoch), end=' ')
        print(' '.join(['{}={:.6f},'.format(k, v) for k, v in eval_results.items()]))
        wandb.log(eval_results)

        return [eval_results[name+'/'+k] for k in ret_metrics]
    
    def _eval_uncertainty_and_print(self, key, cltor, name='', at_epoch=None):
        if 'ins' in key:
            evaluator = self.unc_ins_evaluator
            cltor = convert_instance_output(cltor, self.cfg)
        else:
            evaluator = self.unc_evaluator
        
        eval_results = evaluator.compute(cltor)
        eval_results = rename_unc_keys(eval_results, name, sep='/')

        print("[{}] At epoch {}:".format(name, at_epoch), end=' ')
        print(' '.join(['{}={:.6f},'.format(k, v) for k, v in eval_results.items()]))
        wandb.log(eval_results)

    def _get_unique_id(self, k, idxs, concat=None):
        if k not in self.uid:
            raise KeyError('Key {} not found in `uid`'.format(k))
        uids = self.uid[k]
        idxs = idxs.squeeze().tolist()
        if concat is None:
            return [uids[i] for i in idxs]
        else:
            return [uids[v] + "-" + str(concat[i].item()) for i, v in enumerate(idxs)]

    def test_model(self, model, loader, loader_name=None, ckpt_path=None):
        if ckpt_path is not None:
            net_ckpt = torch.load(ckpt_path)
            model.load_state_dict(net_ckpt['model'])
        model.eval()

        all_idx, all_pred, all_gt, all_gt_dist = [], [], [], []
        all_pred_ins, all_gt_ins = [], []
        for data_idx, data_x, data_y in loader:
            # data_x = (feats, coords) | data_y = (label_slide, label_patch)
            X = data_x[0].cuda() 
            data_label = data_y[0] 
            ins_label  = data_y[1]
            
            with torch.no_grad():
                if self.ins_pred:
                    logit_bag, ins_pred = model(X, ret_ins_res=True)
                    if isinstance(ins_pred, tuple):
                        ins_pred = ins_pred[1] # fetch the last one, enhanced instace predictions
                else:
                    logit_bag = model(X)
                                
            all_gt.append(data_label)
            all_pred.append(logit_bag.detach().cpu())
            all_idx.append(data_idx)
            all_gt_dist.append(has_no_ood_instance(ins_label, self.cfg)) # [1, N] -> [1, 1]
            
            if self.ins_pred:
                # print(ins_label.shape, ins_pred.shape)
                all_gt_ins.append(ins_label.squeeze(0).tolist())
                all_pred_ins.append(ins_pred.squeeze(0).tolist())
                assert len(all_gt_ins[-1]) == len(all_pred_ins[-1]), "Please check the instance-level dimension."

        all_pred = torch.cat(all_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0).squeeze() # [B, ]
        all_idx = torch.cat(all_idx, dim=0).squeeze() # [B, ]
        all_gt_dist = torch.cat(all_gt_dist, dim=0).squeeze() # [B, ]

        cltor = dict()
        cltor['pred'] = {'y': all_gt, 'y_soft': all_gt_dist,  'y_hat': all_pred, 'idx': all_idx}
        if self.ins_pred:
            # the first two elements are in python list style
            cltor['pred_ins'] = {'y': all_gt_ins, 'y_hat': all_pred_ins, 'idx': all_idx}

        return cltor

    def _get_state_dict(self, epoch):
        return {
            'epoch': epoch,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def save_model(self, epoch, ckpt_type='best', run_name='train'):
        net_ckpt_dict = self._get_state_dict(epoch)
        if ckpt_type == 'last':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.last_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.best_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))

    def resume_model(self, ckpt_type='best', run_name='train'):
        if ckpt_type == 'last':
            net_ckpt = torch.load(add_prefix_to_filename(self.last_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            net_ckpt = torch.load(add_prefix_to_filename(self.best_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))
        self.net.load_state_dict(net_ckpt['model']) 
        self.optimizer.load_state_dict(net_ckpt['optimizer']) 
        print('[model] resume the network from {}_{} at epoch {}...'.format(ckpt_type, run_name, net_ckpt['epoch']))

