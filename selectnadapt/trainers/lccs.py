import os
import os.path as osp
import json
import time
import random
import copy
import numpy as np

import torch
import torch.optim as optim
from torch.nn import functional as F
import gc
from dassl.utils import set_random_seed
from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY
from dassl.evaluation import build_evaluator
from dassl.utils import load_checkpoint
from dassl.engine.dg import Vanilla
from dassl.engine.selfsupervision.byol import OnlineNet, MLP,loss_fn
from dassl.modeling.network.csg_builder import CSG
import lccs.imcls.trainers.lccs_utils.lccs_svd as optms
from sklearn import manifold
from sklearn import cluster
import matplotlib.pyplot as plt
import swav.src.resnet50 as resnet_models
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

class AbstractLCCS(Vanilla):
    """Abstract class for LCCS trainer.
    """

    def __init__(self, cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10,
        user_support_coeff_init=None, classifier_type='linear', finetune_classifier=False, svd_dim=1,random=True,resnet="resnet18"):
        """
        Args:
            cfg: configurations
            batch_size: batch size
            ksupport: number of support samples per class
            init_epochs: number of epochs in initialization stage
            grad_update_epochs: number of epochs in gradient update stage
            user_support_coeff_init: user-specified value for support LLCS parameter
            classifier_type: type of classifier
            finetune_classifier: updates classifier by gradient descent if True
            svd_dim: number of support statistics basis vectors
        """
        super().__init__(cfg)

        self.cfg = cfg
        self.batch_size = batch_size
        self.ksupport = ksupport
        self.init_epochs = init_epochs
        self.grad_update_epochs = grad_update_epochs
        self.user_support_coeff_init = user_support_coeff_init
        self.classifier_type = classifier_type
        self.finetune_classifier = finetune_classifier
        self.svd_dim = svd_dim
        self.eps = 1e-5
        self.random = True
        self.evaluator = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        model_copy = copy.deepcopy(self.model)
        self.byol_model = OnlineNet(model_copy.backbone,
                                     projection_size=256,#256
                                     projection_hidden_size=4096,
                                     hidden_layer='global_avgpool',
                                     use_simsiam_mlp=False)
        if resnet =="resnet18":
            self.byol_model.online_encoder.projector = MLP(512, 256)
            self.byol_model = resnet_models.__dict__[resnet](
            normalize=True,
            hidden_mlp=512,
            output_dim=256,
            nmb_prototypes=7)
        elif resnet =="resnet50":
            self.byol_model.online_encoder.projector = MLP(2048,256)  # change dim based on ResNet dim
            self.byol_model = resnet_models.__dict__[resnet](
            normalize=True,
            hidden_mlp=2048,
            output_dim=256,
            nmb_prototypes=7)
        #self.target_byol_model = copy.deepcopy(self.byol_model.online_encoder)
        #self.register_model("byol_model", self.byol_model, self.optim, self.sched)
        self.source_state_dict = None
        self.ss_state_dict = None
    def info_nce_loss(self, features):
        batch_size = features.shape[0] / 2
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / 0.07  # self.temperature
        return logits, labels
    def load_model_nostrict(self, directory, epoch=None):
        """Non-strict loading of model state dict, since LCCS parameters added.
        """
        names = self.get_model_names()
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for name in names:
            if name =='model':
                model_path = osp.join(directory, name, model_file)

                if not osp.exists(model_path):
                    raise FileNotFoundError(
                        'Model not found at "{}"'.format(model_path)
                    )

                checkpoint = load_checkpoint(model_path)
                state_dict = checkpoint['state_dict']
                epoch = checkpoint['epoch']

                print(
                    'Loading weights to {} '
                    'from "{}" (epoch = {})'.format(name, model_path, epoch)
                )
                self._models[name].load_state_dict(state_dict, strict=False)

    def get_ksupport_loaders(self):
        """Obtain support set.
        """
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # val and test loader with sample shuffling
        # follows dassl.data.data_manager
        dataset = build_dataset(self.cfg)
        tfm_train = build_transform(self.cfg, is_train=True)
        tfm_test = build_transform(self.cfg, is_train=False)

        # extract support samples
        np.random.seed(self.cfg.SEED)
        n = len(dataset.test)
        #n = len(dataset.train_x)
        self.num_classes = dataset._num_classes
        support_idx = []
        big_idx = []
        for i in range(dataset._num_classes):
            idx_i = [j for j in range(n) if dataset.test[j]._label == i]
            support_idx += list(np.random.choice(idx_i, self.ksupport, replace=False))
            #big_idx += list(np.random.choice(idx_i,30, replace=False))
        dataset.ksupport = [dataset.test[i] for i in support_idx]
        #dataset.big = [dataset.test[i] for i in big_idx]
        #dataset.ksupport = [dataset.train_x[i] for i in support_idx]
        dataset.eval = [dataset.test[i] for i in range(n) if i not in support_idx]
        #dataset.eval = [dataset.test[i] for i in range(n) if i not in big_idx]
        #dataset.eval = [dataset.test[i] for i in range(len(dataset.test))]

        # support set for finetuning
        self.support_loader_train_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=min(self.batch_size, self.ksupport*dataset._num_classes),
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )
        self.support_loader_test_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=dataset._num_classes*self.ksupport,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
        # evaluation set
        self.eval_loader = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.eval,
            batch_size=self.batch_size,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
        """
        self.big_data = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.big,
            batch_size=min(self.batch_size, self.ksupport * dataset._num_classes),
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )

        self.big_data_test = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.big,
            batch_size=min(self.batch_size, self.ksupport * dataset._num_classes),
            tfm=tfm_test,
            is_train=True,
            dataset_wrapper=None
        )
        """
    def get_ksupport_loaders_ours_sup(self):
        """Obtain support set.
        """
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.cfg.SEED)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.byol_model.to(self.device)
        self.byol_model.eval()
        #self.target_byol_model.to(self.device)
        #self.ss_model.to(self.device)
        # val and test loader with sample shuffling
        # follows dassl.data.data_manager
        dataset = build_dataset(self.cfg)
        tfm_train = build_transform(self.cfg, is_train=True)
        tfm_test = build_transform(self.cfg, is_train=False)
        #tfm_select = build_transform(self.cfg,is_train=False,choices=['gaussian_noise','colorjitter'],is_select=True)#['gaussian_noise','colorjitter','cutout','randomgrayscale','random_flip']
        #tfm_select.transforms[3].p=1
        #tfm_select.transforms[4].p = 1
        #tfm_select = build_transform(self.cfg, is_train=False, choices=['gaussian_noise', 'colorjitter'],
         #                            is_select=True)
        #tfm_select = build_transform(self.cfg, is_train=True, choices=['colorjitter','randomgrayscale','random_flip','gaussian_blur','random_resized_crop'],
         #                                                        is_select=False)

        tfm_select = build_transform(self.cfg, is_train=True, choices=['randaugment','normalize'],
                                     is_select=False)
        #tfm_select.transforms.remove(tfm_select.transforms[0])
        #tfm_select.transforms[0].p = 1
        #tfm_select.transforms[2].p = 1
        if False:
            # extract support samples
            np.random.seed(self.cfg.SEED)
            n = len(dataset.test)
            self.num_classes = dataset._num_classes
            support_idx = []
            for i in range(dataset._num_classes):
                idx_i = [j for j in range(n) if dataset.test[j]._label == i]
                support_idx += list(np.random.choice(idx_i, self.ksupport, replace=False))
            dataset.ksupport = [dataset.test[i] for i in support_idx]
            dataset.eval = [dataset.test[i] for i in range(n) if i not in support_idx]
        else:
            # systematic selection of support samples
            np.random.seed(self.cfg.SEED)
            names = self.get_model_names()
            #n = len(dataset.train_x)
            n = len(dataset.test)
            for name in names:
                #self._models[name].eval()
                #self.model.eval()
                #self.target_byol_model.eval()
                #self.model = self.model.to(torch.double)
                select_set = {}
                support_idx = {}
                ksupport = {}
                eval = {}
                dataset.ksupport = []
                dataset.eval = []
                for i in range(dataset._num_classes):

                    #idx_i = [j for j in range(n) if dataset.train_x[j]._label == i]
                    #all_indicies = [dataset.train_x[i] for i in idx_i]
                    idx_i = [j for j in range(n) if dataset.test[j]._label == i]
                    all_indicies = [dataset.test[j] for j in idx_i]
                    self.select_loader = build_data_loader(
                        self.cfg,
                        sampler_type='RandomSampler',
                        data_source=all_indicies,
                        batch_size=self.batch_size,
                        tfm=tfm_select,#.transforms,
                        #tfm=tfm_select,
                        is_train=True,
                        dataset_wrapper=None,
                        drop_last=False
                )
                    num_imgs = len(self.select_loader.dataset)
                    select_set[i] = torch.zeros((num_imgs),dtype=torch.float32)
                    img_indices = torch.zeros((num_imgs),dtype=torch.int64)
                    len_select_loader = len(self.select_loader)
                    select_loader_iter = iter(self.select_loader)
                    #self.model = self.model.to(torch.double)
                    features = torch.zeros((len(idx_i),256),dtype=torch.float32)#512*7*7

                    for iterate in range(len_select_loader):
                        batch = next(select_loader_iter)
                        #input, label, indices, img_path,orig_img = self.parse_batch_select_per_image_aug(batch,tfm_select.transforms)
                        input, label, indices, img_path, orig_img = self.parse_batch_select_pipeline_aug(batch)
                        img_indices[indices] = indices
                        #label = label.to(torch.double)
                        # order by label
                        #idx = np.argsort(label.cpu())
                        #label = label[idx]
                        loss = 0
                        with torch.no_grad():
                            #output_orig = self.model(orig_img)
                            #output = F.softmax(output,dim=1)
                            #prob,pred = torch.max(F.softmax(output,dim=1),dim=1)
                            #output_orig_feat = self.model.backbone.featuremaps(orig_img)
                            output_orig_feat,_ = self.byol_model.online_encoder(orig_img)
                            #proj_byol, rep_byol = self.byol_model.online_encoder(orig_img)
                            #proj_source = self.byol_model.online_encoder.projector(self.model.backbone(orig_img))
                            #output_orig_feat = torch.cat([proj_byol,proj_source],dim=1)
                            #output_orig_feat = self.byol_model.online_predictor(online_proj)

                            #output_orig_feat,_ = self.target_byol_model(orig_img)
                            features[indices] = output_orig_feat.detach().cpu() #torch.flatten(output_orig_feat.detach().cpu(),start_dim=1,end_dim=3)
                            #features[indices] = F.normalize(features[indices], dim=1)
                        if isinstance(input,list):
                            #output_aug = []
                            for aug in input:
                                #input_temp = input
                                #input_temp.remove(input_temp(input.index(aug1))
                                #with torch.no_grad():
                                    #output_orig_feat, _ = self.target_byol_model(aug1)
                                #for aug2 in input_temp:
                                #aug = aug.to(torch.double)
                                #aug = aug[idx]
                                with torch.no_grad():
                                    output_aug,_ = self.byol_model.online_encoder(aug)#self.model.backbone.featuremaps(aug)
                                    #output_aug= self.byol_model.online_predictor(online_proj)
                                    #select_set[i][indices] += torch.cosine_similarity(torch.flatten(output_orig_feat.detach().cpu(),start_dim=1,end_dim=3)
                                     #                                                ,torch.flatten(output_aug.detach().cpu(),start_dim=1,end_dim=3),dim=1)
                                    #output_aug = torch.flatten(output_aug.detach().cpu(),start_dim=1,end_dim=3)
                                    #features[indices] += F.normalize(output_aug.detach().cpu(),dim=1)
                                    #features[indices] += output_aug.detach().cpu()
                                    #select_set[i][indices] += loss_fn(output_aug,output_orig_feat).detach().cpu()
                                    #output = self.model(aug)
                                    #output = F.softmax(output,dim=1)
                                    #output_aug.append(output)
                                #loss+= F.cross_entropy(output, label.long(),reduction='none').detach().cpu()
                            #one_hot = torch.zeros(1, 7, device='cuda')
                            #one_hot[0, label[0].item()] = 1
                            #for out in output_aug:
                                #if torch.eq(target,input).sum().item() != (target.shape[0]*target.shape[1]):
                                #loss+= F.kl_div(F.log_softmax(input,dim=1),F.softmax(output,dim=1),reduction='none').sum(dim=1).detach().cpu()
                                #loss += F.kl_div(F.log_softmax(out, dim=1), one_hot,reduction='none').sum(dim=1).detach().cpu()
                            #loss += F.kl_div(F.log_softmax(output_orig, dim=1), one_hot, reduction='none').sum(dim=1).detach().cpu()
                        else:
                            with torch.no_grad():
                                output = self.model(input)
                            loss += F.cross_entropy(output, label.long(), reduction='none').detach().cpu()

                        #select_set[i][indices] = loss/len(input)
                        #del input,orig_img,label
                        gc.collect()
                        torch.cuda.empty_cache()
                    #mean_feat_aug = (features)#len(tfm_select.transforms))
                    #mean_feat_aug = mean_feat_aug/(torch.max(torch.norm(mean_feat_aug,dim=1)))
                    #mean_feat_cls = (mean_feat_aug.sum(dim=0) / len(idx_i)).unsqueeze(0)
                    #norm_mean_feat_cls = torch.norm(mean_feat_cls, dim=1)
                    #norms = torch.norm(mean_feat_aug, dim=1)
                    mean_feat_aug = (features)
                    #mean_feat_cls = (mean_feat_aug.sum(dim=0) / len(idx_i)).unsqueeze(0)
                    #mean_feat_aug = torch.nn.functional.normalize(mean_feat_aug,dim=0)
                    #tsne = manifold.TSNE(n_components=256,random_state=self.cfg.SEED,method='exact').fit_transform(mean_feat_aug)
                    #tx = tsne[:, 0]
                    #ty = tsne[:, 1]
                    #tx = scale_to_01_range(tx)
                    #ty = scale_to_01_range(ty)
                    #norm_space = np.concatenate((np.expand_dims(tx,axis=1),np.expand_dims(ty,1)),axis=1)
                    #kmeans = cluster.KMeans(n_clusters=5,random_state=self.cfg.SEED).fit(mean_feat_aug)
                    kmeans = cluster.KMeans(n_clusters=self.ksupport, random_state=self.cfg.SEED).fit(mean_feat_aug)
                    labels = kmeans.labels_
                    candidate = []
                    for label in range(0,self.ksupport):
                        res = list(np.where(labels==label))[0]
                        dist = np.zeros((len(res), 1))
                        centroid = kmeans.cluster_centers_[label]
                        for d in range(0,len(dist)):
                            dist[d] += np.linalg.norm(centroid-np.array(mean_feat_aug[res][d]))#np.linalg.norm(centroid-norm_space[res][d])
                            #dist[d] += np.linalg.norm(centroid - tsne[res][d])
                        best = np.argmin(dist)
                        candidate.append(res[best])
                    

                    val, pos = torch.topk(select_set[i],k=self.ksupport,dim=0,largest=True)
                    pos = candidate
                    support_idx[i] = list(img_indices[pos].detach())
                    support_idx[i] = [item.item() for item in support_idx[i]]
                    ksupport[i] = [all_indicies[p] for p in support_idx[i]]
                    eval[i] = [all_indicies[p] for p in range(len(all_indicies)) if p not in support_idx[i]]

                for key in ksupport.keys():
                    dataset.ksupport += ksupport[key]
                    dataset.eval += eval[key]
                #dataset.eval = [dataset.test[i] for i in range(len(dataset.test)) ]
                #dataset.eval = [dataset.test[i] for i in range(n) if i not in support_idx]
        assert (len(dataset.ksupport) == dataset._num_classes*self.ksupport)
        assert (len(dataset.ksupport) + len(dataset.eval) == n)
        # support set for finetuning
        self.cfg['DATALOADER']['K_TRANSFORMS'] = 1
        self.support_loader_train_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=min(self.batch_size, self.ksupport * dataset._num_classes),
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )
        self.support_loader_test_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=dataset._num_classes * self.ksupport,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
        # evaluation set
        self.eval_loader = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.eval,
            batch_size=self.batch_size,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )

    def get_ksupport_loaders_ours_unsup(self):
        """Obtain support set.
        """
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.cfg.SEED)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.byol_model.to(self.device)
        self.byol_model.eval()
        #self.model.to(self.device)
        self.model.eval()

        # val and test loader with sample shuffling
        # follows dassl.data.data_manager
        dataset = build_dataset(self.cfg)
        tfm_train = build_transform(self.cfg, is_train=True)
        tfm_test = build_transform(self.cfg, is_train=False)


        tfm_select = build_transform(self.cfg, is_train=True, choices=['randaugment','normalize'],
                                     is_select=False)


        # systematic selection of support samples
        np.random.seed(self.cfg.SEED)
        names = self.get_model_names()
        #n = len(dataset.train_x)
        n = len(dataset.test)
        select_set = {}
        support_idx = {}
        ksupport = {}
        support_imgpth = []
        idx_all = [j for j in range(n)]
        all_indicies = [dataset.test[j] for j in idx_all]
        self.select_loader = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=all_indicies,
            batch_size=512,
            tfm=tfm_select,  # .transforms,
            # tfm=tfm_select,
            is_train=True,
            dataset_wrapper=None,
            drop_last=False
        )
        len_select_loader = len(self.select_loader)
        select_loader_iter = iter(self.select_loader)
        sample_pl = torch.zeros((len(all_indicies)))
        ## get pseudo-labels
        for iterate in range(len_select_loader):
            batch = next(select_loader_iter)
            # input, label, indices, img_path,orig_img = self.parse_batch_select_per_image_aug(batch,tfm_select.transforms)
            input, label, indices, img_path, orig_img = self.parse_batch_select_pipeline_aug(batch)
            with torch.no_grad():

                output_orig_source = self.model(orig_img)
                #output_orig_byol = self.model.classifier(self.byol_model.online_encoder.net(orig_img))
                output_orig_byol = self.model.classifier(self.byol_model.forward_backbone(orig_img))
                output_source = F.softmax(output_orig_source, dim=1)
                output_byol = F.softmax(output_orig_byol, dim=1)
                prob, pred = torch.max((output_byol+output_source)/2, dim=1)
                #prob, pred = torch.max((output_source), dim=1)

                #output = F.softmax(output_orig, dim=1)
                #entropy = -torch.sum(output * torch.log(output + 1e-10), dim=1)

                #thresh = np.percentile(
                 #   entropy.detach().cpu().numpy().flatten(), 80
                #)
                #prob, pred = torch.max(output, dim=1)
                #pred[prob < 0.8] = -1
              

            sample_pl[indices] +=  pred.detach().cpu()

        del self.select_loader
        del select_loader_iter
        all_indicies_correct = []
        dataset.ksupport = []
        dataset.eval = []
        for i in range(dataset._num_classes):
            idx_i = [j for j in range(n) if sample_pl[j] == i]
            #temp = [j for j in range(n) if dataset.test[j]._label == i]
            all_indicies = [dataset.test[j] for j in idx_i]
            for j in range(n):
                if dataset.test[j]._label == i:
                    all_indicies_correct.append(dataset.test[j])
            self.select_loader = build_data_loader(
                self.cfg,
                sampler_type='RandomSampler',
                data_source=all_indicies,
                batch_size=self.batch_size,
                tfm=tfm_select,#.transforms,
                #tfm=tfm_select,
                is_train=True,
                dataset_wrapper=None,
                drop_last=False
        )
            num_imgs = len(self.select_loader.dataset)
            select_set[i] = torch.zeros((num_imgs),dtype=torch.float32)
            img_indices = torch.zeros((num_imgs),dtype=torch.int64)
            len_select_loader = len(self.select_loader)
            select_loader_iter = iter(self.select_loader)
            #self.model = self.model.to(torch.double)
            features = torch.zeros((len(idx_i),256),dtype=torch.float32)#512*7*7

            for iterate in range(len_select_loader):
                batch = next(select_loader_iter)
                #input, label, indices, img_path,orig_img = self.parse_batch_select_per_image_aug(batch,tfm_select.transforms)
                input, label, indices, img_path, orig_img = self.parse_batch_select_pipeline_aug(batch)
                img_indices[indices] = indices
                #label = label.to(torch.double)
                # order by label
                #idx = np.argsort(label.cpu())
                #label = label[idx]
                loss = 0
                with torch.no_grad():
                    #output_orig = self.model(orig_img)
                    #output = F.softmax(output,dim=1)
                    #prob,pred = torch.max(F.softmax(output,dim=1),dim=1)
                    #output_orig_feat = self.model.backbone(orig_img)
                    ########
                    #output_orig_feat, _ = self.byol_model.online_encoder(orig_img)
                    output_orig_feat, _ = self.byol_model(orig_img)
                    ######
                    #output_orig_feat,_ = self.byol_model.online_encoder(orig_img)
                    #output_orig_feat = 0.5*output_orig_feat
                    #output_orig_feat += 0.5*self.byol_model.online_encoder.projector(self.model.backbone(orig_img))
                    #output_orig_feat,_ = self.target_byol_model(orig_img)
                    features[indices] = output_orig_feat.detach().cpu() #torch.flatten(output_orig_feat.detach().cpu(),start_dim=1,end_dim=3)
                    #features[indices] = F.normalize(features[indices], dim=1)
                #if isinstance(input,list):
                    #output_aug = []
                    #for aug in input:
                        #input_temp = input
                        #input_temp.remove(input_temp(input.index(aug1))
                        #with torch.no_grad():
                            #output_orig_feat, _ = self.target_byol_model(aug1)
                        #for aug2 in input_temp:
                        #aug = aug.to(torch.double)
                        #aug = aug[idx]
                        #with torch.no_grad():
                            #output_aug,_ = self.byol_model.online_encoder(aug)#self.model.backbone.featuremaps(aug)
                            #output_aug= self.byol_model.online_predictor(online_proj)
                            #select_set[i][indices] += torch.cosine_similarity(torch.flatten(output_orig_feat.detach().cpu(),start_dim=1,end_dim=3)
                             #                                                ,torch.flatten(output_aug.detach().cpu(),start_dim=1,end_dim=3),dim=1)
                            #output_aug = torch.flatten(output_aug.detach().cpu(),start_dim=1,end_dim=3)
                            #features[indices] += F.normalize(output_aug.detach().cpu(),dim=1)
                            #features[indices] += output_aug.detach().cpu()
                            #select_set[i][indices] += loss_fn(output_aug,output_orig_feat).detach().cpu()
                            #output = self.model(aug)
                            #output = F.softmax(output,dim=1)
                            #output_aug.append(output)
                        #loss+= F.cross_entropy(output, label.long(),reduction='none').detach().cpu()
                    #one_hot = torch.zeros(1, 7, device='cuda')
                    #one_hot[0, label[0].item()] = 1
                    #for out in output_aug:
                        #if torch.eq(target,input).sum().item() != (target.shape[0]*target.shape[1]):
                        #loss+= F.kl_div(F.log_softmax(input,dim=1),F.softmax(output,dim=1),reduction='none').sum(dim=1).detach().cpu()
                        #loss += F.kl_div(F.log_softmax(out, dim=1), one_hot,reduction='none').sum(dim=1).detach().cpu()
                    #loss += F.kl_div(F.log_softmax(output_orig, dim=1), one_hot, reduction='none').sum(dim=1).detach().cpu()
                #else:
                    #with torch.no_grad():
                        #output = self.model(input)
                    #loss += F.cross_entropy(output, label.long(), reduction='none').detach().cpu()

                #select_set[i][indices] = loss/len(input)
                #del input,orig_img,label
                gc.collect()
                torch.cuda.empty_cache()
            mean_feat_aug = (features)#len(tfm_select.transforms))
            mean_feat_aug = mean_feat_aug/(torch.max(torch.norm(mean_feat_aug,dim=1)))
            mean_feat_cls = (mean_feat_aug.sum(dim=0) / len(idx_i)).unsqueeze(0)
            norm_mean_feat_cls = torch.norm(mean_feat_cls, dim=1)
            norms = torch.norm(mean_feat_aug, dim=1)
            mean_feat_aug = (features)
            mean_feat_cls = (mean_feat_aug.sum(dim=0) / len(idx_i)).unsqueeze(0)
           
            if len(mean_feat_aug) >1:
                kmeans = cluster.KMeans(n_clusters=self.ksupport,random_state=self.cfg.SEED).fit(mean_feat_aug)
                labels = kmeans.labels_
                candidate = []
                for label in range(0,self.ksupport):
                    res = list(np.where(labels==label))[0]
                    dist = np.zeros((len(res), 1))
                    centroid = kmeans.cluster_centers_[label]
                    for d in range(0,len(dist)):
                        dist[d] += np.linalg.norm(centroid-np.array(mean_feat_aug[res][d]))#np.linalg.norm(centroid-norm_space[res][d])
                    best = np.argmin(dist)
                    candidate.append(res[best])
            else:
                candidate=[0]

            pos = candidate
            support_idx[i] = list(img_indices[pos].detach())
            support_idx[i] = [item.item() for item in support_idx[i]]
            #ksupport[i] = [all_indicies[p] for p in support_idx[i]]
            dataset.ksupport += [all_indicies[p] for p in support_idx[i]]


        paths = [img.impath for img in dataset.ksupport]
        for sample in all_indicies_correct:
            if sample.impath not in paths:
                dataset.eval.append(sample)

        #eval = [all_indicies[p] for p in range(len(all_indicies)) if p not in support_idx[i] and all_indicies[p]._label == i]

        #for key in ksupport.keys():
         #   dataset.ksupport += ksupport[key]
                #dataset.eval += eval[key]

        #assert(len(dataset.ksupport) == self.ksupport*dataset._num_classes)
        #assert(len(dataset.ksupport)+len(dataset.eval)==n)
        # support set for finetuning
        self.cfg['DATALOADER']['K_TRANSFORMS'] = 1
        self.support_loader_train_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=min(self.batch_size, self.ksupport * dataset._num_classes),
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )
        self.support_loader_test_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=dataset._num_classes * self.ksupport,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
        # evaluation set
        self.eval_loader = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.eval,
            batch_size=self.batch_size,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None)
    def get_ksupport_loaders_al(self):
        """Obtain support set.
        """
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.cfg.SEED)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.byol_model.to(self.device)
        self.byol_model.eval()
        #self.target_byol_model.to(self.device)
        #self.ss_model.to(self.device)
        # val and test loader with sample shuffling
        # follows dassl.data.data_manager
        dataset = build_dataset(self.cfg)
        tfm_train = build_transform(self.cfg, is_train=True)
        tfm_test = build_transform(self.cfg, is_train=False)
        #tfm_select = build_transform(self.cfg,is_train=False,choices=['gaussian_noise','colorjitter'],is_select=True)#['gaussian_noise','colorjitter','cutout','randomgrayscale','random_flip']
        #tfm_select.transforms[3].p=1
        #tfm_select.transforms[4].p = 1
        #tfm_select = build_transform(self.cfg, is_train=False, choices=['gaussian_noise', 'colorjitter'],
         #                            is_select=True)
        #tfm_select = build_transform(self.cfg, is_train=True, choices=['colorjitter','randomgrayscale','random_flip','gaussian_blur','random_resized_crop'],
         #                                                        is_select=False)

        tfm_select = build_transform(self.cfg, is_train=True, choices=['randaugment','normalize'],
                                     is_select=False)
        #tfm_select.transforms.remove(tfm_select.transforms[0])
        #tfm_select.transforms[0].p = 1
        #tfm_select.transforms[2].p = 1
        if False:
            # extract support samples
            np.random.seed(self.cfg.SEED)
            n = len(dataset.test)
            self.num_classes = dataset._num_classes
            support_idx = []
            for i in range(dataset._num_classes):
                idx_i = [j for j in range(n) if dataset.test[j]._label == i]
                support_idx += list(np.random.choice(idx_i, self.ksupport, replace=False))
            dataset.ksupport = [dataset.test[i] for i in support_idx]
            dataset.eval = [dataset.test[i] for i in range(n) if i not in support_idx]
        else:
            # systematic selection of support samples
            np.random.seed(self.cfg.SEED)
            names = self.get_model_names()
            #n = len(dataset.train_x)
            n = len(dataset.test)
            for name in names:
                #self._models[name].eval()
                #self.model.eval()
                #self.target_byol_model.eval()
                #self.model = self.model.to(torch.double)
                select_set = {}
                support_idx = {}
                ksupport = {}
                eval = {}
                dataset.ksupport = []
                dataset.eval = []
                for i in range(dataset._num_classes):

                    #idx_i = [j for j in range(n) if dataset.train_x[j]._label == i]
                    #all_indicies = [dataset.train_x[i] for i in idx_i]
                    idx_i = [j for j in range(n) if dataset.test[j]._label == i]
                    all_indicies = [dataset.test[j] for j in idx_i]
                    self.select_loader = build_data_loader(
                        self.cfg,
                        sampler_type='RandomSampler',
                        data_source=all_indicies,
                        batch_size=self.batch_size,
                        tfm=tfm_select,#.transforms,
                        #tfm=tfm_select,
                        is_train=True,
                        dataset_wrapper=None,
                        drop_last=False
                )
                    num_imgs = len(self.select_loader.dataset)
                    select_set[i] = torch.zeros((num_imgs),dtype=torch.float32)
                    img_indices = torch.zeros((num_imgs),dtype=torch.int64)
                    len_select_loader = len(self.select_loader)
                    select_loader_iter = iter(self.select_loader)

                    loss = torch.zeros((len(idx_i)), dtype=torch.float32)
                    #self.model.backbone.dropout.train()
                    for iterate in range(len_select_loader):
                        batch = next(select_loader_iter)
                        #input, label, indices, img_path,orig_img = self.parse_batch_select_per_image_aug(batch,tfm_select.transforms)
                        input, label, indices, img_path, orig_img = self.parse_batch_select_pipeline_aug(batch)
                        img_indices[indices] = indices
                        #label = label.to(torch.double)
                        # order by label
                        #idx = np.argsort(label.cpu())
                        #label = label[idx]
                        #loss = 0
                        with torch.no_grad():
                            output = 0
                            #for _ in range(10):
                            output_orig = self.model.classifier((self.byol_model.online_encoder.net(orig_img)))
                            output += F.softmax(output_orig,dim=1)
                            #output /=10
                            #prob,pred = torch.max(F.softmax(output,dim=1),dim=1)
                            #output_orig_feat = self.model.backbone.featuremaps(orig_img)
                            #output_orig_feat,_ = self.byol_model.online_encoder(orig_img)
                            loss[indices] = ((-output*torch.log(output)).sum(dim=1)/dataset._num_classes).detach().cpu()#F.cross_entropy(output_orig, label.long(), reduction='none').detach().cpu()

                        #select_set[i][indices] = loss/len(input)
                        #del input,orig_img,label
                        gc.collect()
                        torch.cuda.empty_cache()


                    val, pos = torch.topk(loss,k=self.ksupport,dim=0,largest=True)
                    support_idx[i] = list(img_indices[pos].detach())
                    support_idx[i] = [item.item() for item in support_idx[i]]
                    ksupport[i] = [all_indicies[p] for p in support_idx[i]]
                    eval[i] = [all_indicies[p] for p in range(len(all_indicies)) if p not in support_idx[i]]

                for key in ksupport.keys():
                    dataset.ksupport += ksupport[key]
                    dataset.eval += eval[key]
                #dataset.eval = [dataset.test[i] for i in range(len(dataset.test)) ]
                #dataset.eval = [dataset.test[i] for i in range(n) if i not in support_idx]
        #self.model.backbone.dropout.eval()
        assert (len(dataset.ksupport) == dataset._num_classes*self.ksupport)
        assert (len(dataset.ksupport) + len(dataset.eval) == n)
        # support set for finetuning
        self.cfg['DATALOADER']['K_TRANSFORMS'] = 1
        self.support_loader_train_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=min(self.batch_size, self.ksupport * dataset._num_classes),
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )
        self.support_loader_test_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=dataset._num_classes * self.ksupport,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
        # evaluation set
        self.eval_loader = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.eval,
            batch_size=self.batch_size,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
    def check_parameters(self,params1,params2):
        x=1
        for p1, p2 in zip(params1,params2):
            if p1.data.ne(p2.data).sum() > 0:
                x = 0
        print(x)
    def initialization_stage(self):
        """Initialization stage.

        Find initialization for source and support LCCS parameters.
        """
        """
        if isinstance(self.model, CSG):
            self.model.cuda()
            self.model = self.model.to(torch.double)
            self.model.encoder_k.compute_source_stats()
            self.model.encoder_q.compute_source_stats()

            self.model.encoder_k.set_svd_dim(self.svd_dim)
            self.model.encoder_q.set_svd_dim(self.svd_dim)

            self.model.encoder_k.set_lccs_use_stats_status('initialization_stage')
            self.model.encoder_q.set_lccs_use_stats_status('initialization_stage')

        else:
        """
        self.model = self.model.to(torch.double)
        self.model.backbone.compute_source_stats()
        self.model.backbone.set_svd_dim(self.svd_dim)
        self.model.backbone.set_lccs_use_stats_status('initialization_stage')
        self.set_model_mode('eval')
        candidates_init = np.arange(0, 1.1, 0.1)
        #candidates_init = [0.9]
        if self.user_support_coeff_init is None:
            cross_entropy = {}
            with torch.no_grad():
                for i in candidates_init:
                    print(f'initialization of support LCCS param: {i}')
                    #try:
                     #   self.model.encoder_q.set_coeff(i, 1. - i)
                      #  self.model.encoder_k.set_coeff(i, 1. - i)
                    #except AttributeError:
                    self.model.backbone.set_coeff(i, 1. - i)
                    set_random_seed(self.cfg.SEED)
                    cross_entropy_list = []
                    accuracy_list = []
                    # iterate through support set for init_epochs
                    len_support_loader_train_transform = len(self.support_loader_train_transform)
                    #len_support_loader_train_transform = len(self.big_data)
                    #start = time.time()
                    for j in range(self.init_epochs):
                        #support_loader_train_transform_iter = iter(self.big_data)
                        support_loader_train_transform_iter = iter(self.support_loader_train_transform)
                        for iterate in range(len_support_loader_train_transform):
                            if (j == 0) and (iterate == 0):
                                #try:
                                self.model.backbone.set_lccs_update_stats_status('initialize_support')
                                #except AttributeError:
                                 #   self.model.encoder_q.set_lccs_update_stats_status('initialize_support')
                                  #  self.model.encoder_k.set_lccs_update_stats_status('initialize_support')

                            else:
                                #try:
                                self.model.backbone.set_lccs_update_stats_status('update_support_by_momentum')
                                #except AttributeError:
                                 #   self.model.encoder_q.set_lccs_update_stats_status('update_support_by_momentum')
                                  #  self.model.encoder_k.set_lccs_update_stats_status('update_support_by_momentum')

                            batch = next(support_loader_train_transform_iter)
                            input, label = self.parse_batch_train(batch)
                            input, label = input.to(torch.double), label.to(torch.double)
                            output = self.model(input)
                    #end = time.time()-start
                    #print(end)
                    # evaluate on support set
                    self.model.backbone.set_lccs_update_stats_status('no_update')

                    len_support_loader_test_transform = len(self.support_loader_test_transform)
                    support_loader_train_transform_iter = iter(self.support_loader_test_transform)
                    #len_support_loader_test_transform = len(self.big_data_test)
                    #support_loader_train_transform_iter = iter(self.big_data_test)

                    for iterate in range(len_support_loader_test_transform):
                        batch = next(support_loader_train_transform_iter)
                        input, label = self.parse_batch_test(batch)
                        input, label = input.to(torch.double), label.to(torch.double)
                        if isinstance(self.model,CSG):
                            # compute output
                            output, _ = self.model.encoder_q(input, task='new')
                            output = torch.sigmoid(output)
                            output = (output + torch.sigmoid(self.model.encoder_q(torch.flip(input, dims=(3,)), task='new')[0])) / 2

                        else:
                            output = self.model(input)
                            # cross-entropy, lower the better
                        ce_i = F.cross_entropy(output, label.long())
                        cross_entropy_list.append(float(ce_i))
                    # consolidate cross-entropy
                    cross_entropy[i] = np.mean(cross_entropy_list)

            ce_init = [cross_entropy[i] for i in candidates_init]
            print(f'candidate values: {candidates_init}')
            print(f'cross-entropy: {ce_init}')
            # pick candidate initalization with lowest cross entropy
            user_support_coeff_init = max([v for i, v in enumerate(candidates_init) if ce_init[i] == min(ce_init)])
            print(f'selected initialization of support LCCS param: {user_support_coeff_init}')
        else:
            user_support_coeff_init = self.user_support_coeff_init

        # iterate through support set for init_epochs to initialize model with selected initialization of LCCS parameters
        self.model.backbone.set_coeff(user_support_coeff_init, 1. - user_support_coeff_init)

        set_random_seed(self.cfg.SEED)
        with torch.no_grad():
            support_loader_train_transform_iter = iter(self.support_loader_test_transform)
            #support_loader_train_transform_iter = iter(self.big_data_test)
            #try:
            self.model.backbone.set_lccs_update_stats_status('compute_support_svd')
            #except AttributeError:
             #   self.model.encoder_q.set_lccs_update_stats_status('compute_support_svd')
               # self.model.encoder_k.set_lccs_update_stats_status('compute_support_svd')

            batch = next(support_loader_train_transform_iter)
            input, label = self.parse_batch_train(batch)
            input, label = input.to(torch.double), label.to(torch.double)
            output = self.model(input)

        # initialize LCCS parameters as leanable
        #try:
        self.model.backbone.initialize_trainable(support_coeff_init=user_support_coeff_init, source_coeff_init=1. - user_support_coeff_init)
        #except AttributeError:
         #   self.model.encoder_q.initialize_trainable(support_coeff_init=user_support_coeff_init, source_coeff_init=1. - user_support_coeff_init)
          #  self.model.encoder_k.initialize_trainable(support_coeff_init=user_support_coeff_init, source_coeff_init=1. - user_support_coeff_init)

    def gradient_update_stage(self):
        """Gradient update stage.

        Update trainable parameters.
        """
        print("### Finetuning LCCS params ###")
        self.model_optms = optms.configure_model(self.model, component='LCCS').cuda()
        params, param_names = optms.collect_params(self.model_optms, component='LCCS')
        #self.model_optms = optms.configure_model(self.model, component='backbone').cuda()
        #params, param_names = optms.collect_params(self.model_optms, component='backbone')

        optimizer = torch.optim.Adam(params,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0)
        #optimizer = torch.optim.SGD(params,
         #                            lr=1e-2,
          #                           nesterov=True,
           #                          momentum=0.9,
            #                         weight_decay=1e-6)
        #optimizer = torch.optim.Adagrad(params,
         #                            lr=1e-3,
          #                           weight_decay=1e-6)
        #try:
        self.model_optms.backbone.set_lccs_use_stats_status('gradient_update_stage')
        self.model_optms.backbone.set_lccs_update_stats_status('no_update')
        #except AttributeError:
            #self.model_optms.encoder_k.set_lccs_use_stats_status('gradient_update_stage')
            #self.model_optms.encoder_q.set_lccs_use_stats_status('gradient_update_stage')

            #self.model_optms.encoder_k.set_lccs_update_stats_status('no_update')
            #self.model_optms.encoder_q.set_lccs_update_stats_status('no_update')

        self.model_optms = self.train(self.model_optms, optimizer, self.support_loader_train_transform, self.support_loader_test_transform,
            grad_update_epochs=self.grad_update_epochs, num_classes=self.num_classes, classifier_type='linear',
            initialize_centroid=(self.classifier_type == 'mean_centroid'))

        if self.finetune_classifier:
            print("### Finetuning classifier ###")
            self.model_optms = optms.configure_model(self.model_optms, component='classifier')
            params, param_names = optms.collect_params(self.model_optms, component='classifier')
            optimizer = torch.optim.Adam(params,
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=0)
            self.model_optms = self.train(self.model_optms, optimizer, self.support_loader_train_transform, self.support_loader_test_transform,
                grad_update_epochs=50,#self.grad_update_epochs,
                                          num_classes=self.num_classes, classifier_type=self.classifier_type)

    def train(self, model_optms, optimizer, support_loader_train_transform, support_loader_test_transform, grad_update_epochs, num_classes,
        classifier_type='linear', initialize_centroid=False):
        """Model finetuning.
        """
        len_support_loader_train_transform = len(self.support_loader_train_transform)
        #len_support_loader_train_transform = len(self.big_data)
        for epoch in range(grad_update_epochs):
            support_loader_train_transform_iter = iter(self.support_loader_train_transform)
            #support_loader_train_transform_iter = iter(self.big_data)
            for iterate in range(len_support_loader_train_transform):
                batch = next(support_loader_train_transform_iter)
                input, label = self.parse_batch_test(batch)
                input, label = input.to(torch.double), label.to(torch.double)
                # order by label
                idx = np.argsort(label.cpu())
                input = input[idx]
                label = label[idx]

                if classifier_type == 'linear':
                    if isinstance(model_optms,CSG):
                        output = self.model(input)
                        # synthetic task
                        loss = F.cross_entropy(output['output'], label.long())
                        for idx in range(len(self.model.stages)):
                            _loss = 0
                            acc1 = None
                            # predictions: cosine b/w q and k
                            # targets: zeros
                            _loss = F.cross_entropy(output['predictions_csg'][idx], output['targets_csg'][idx])
                            loss = loss + _loss * 0.1
                    else:
                        output = model_optms(input)
                        loss = F.cross_entropy(output, label.long())
                elif classifier_type == 'mean_centroid':
                    if isinstance(model_optms,CSG):
                        # compute output
                        _, feat = model_optms.encoder_q(input, task='new')
                        feat = model_optms.encoder_q.avgpool(feat['layer4'])
                        feat = feat.view(feat.shape(0),-1)
                        #output = torch.sigmoid(output)
                        #output = (output + torch.sigmoid(model_optms.encoder_q(torch.flip(input, dims=(3,)), task='new')[0])) / 2
                    else:
                        feat = model_optms.backbone(input)

                    uniqlabel = np.unique(label.cpu().numpy())
                    # form cluster centroids
                    newlabel = copy.deepcopy(label)
                    L = len(uniqlabel)
                    centroid_list = []
                    for i in range(L):
                        cluster_i = feat[label == uniqlabel[i]]
                        centroid_i = cluster_i.mean(dim=0)
                        centroid_list.append(centroid_i)
                        # relabel classes to remove missing classes in minibatch
                        newlabel[newlabel == uniqlabel[i]] = i
                    centroid = torch.stack(centroid_list).detach()
                    # obtain probability by cosine similarity
                    cossim = F.cosine_similarity(feat.unsqueeze(1), centroid, dim=-1)
                    # cross-entropy
                    newlabel = torch.tensor(newlabel, dtype=label.dtype).cuda() 
                    loss = F.cross_entropy(cossim, newlabel.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f'Epoch {epoch} Iteration {iterate}: loss {loss.item()}')

        # save centroids at end of finetuning   
        if initialize_centroid:
            #try:
            model_optms.backbone.set_lccs_use_stats_status('evaluation_stage')
            model_optms.backbone.set_lccs_update_stats_status('no_update')
            #except AttributeError:
             #   model_optms.encoder_q.set_lccs_use_stats_status('evaluation_stage')
              #  model_optms.encoder_q.set_lccs_update_stats_status('no_update')

               # model_optms.encoder_k.set_lccs_use_stats_status('evaluation_stage')
                #model_optms.encoder_k.set_lccs_update_stats_status('no_update')
            with torch.no_grad():
                cluster_dict = {i: [] for i in range(num_classes)}
                support_loader_train_transform_iter = iter(self.support_loader_test_transform)
                #support_loader_train_transform_iter = iter(self.big_data_test)
                batch = next(support_loader_train_transform_iter)
                input, label = self.parse_batch_test(batch)
                input, label = input.to(torch.double), label.to(torch.double)
                if isinstance(model_optms, CSG):
                    # compute output
                    _, feat = model_optms.encoder_q(input, task='new')
                    feat = model_optms.encoder_q.avgpool(feat['layer4'])
                    feat = feat.view(feat.size(0), -1)
                    # output = torch.sigmoid(output)
                    # output = (output + torch.sigmoid(model_optms.encoder_q(torch.flip(input, dims=(3,)), task='new')[0])) / 2
                else:
                    feat = model_optms.backbone(input)

                # collect features per class
                for i in range(num_classes):
                    cluster_i = feat[label == i]
                    cluster_dict[i].append(cluster_i)

                # form cluster centroids
                centroid_list = []
                for i in range(num_classes):
                    cluster_i = cluster_dict[i]
                    centroid_i = torch.cat(cluster_i).mean(dim=0)
                    centroid_list.append(centroid_i)
                model_optms.centroid = torch.stack(centroid_list)

        return model_optms

    @torch.no_grad()
    def test(self):
        """Evaluation.
        """

        self.evaluator.reset()
        try:
            self.model_optms = self.model_optms.to(torch.float)
            self.model_optms.backbone.set_lccs_use_stats_status('evaluation_stage')
            self.model_optms.backbone.set_lccs_update_stats_status('no_update')
            #self.model_optms.eval()
        except AttributeError:
            self.model.eval()
            #pass

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.eval_loader

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if self.classifier_type == 'linear':
                output = self.model_optms(input)#F.softmax(self.model(input),dim=1)#self.model_optms(input)
            elif self.classifier_type == 'mean_centroid':
                try:
                    feat = self.model_optms.backbone(input) # n x C
                except AttributeError:
                    _, feat = self.model_optms.encoder_q(input, task='new')
                    feat = self.model_optms.encoder_q.avgpool(feat['layer4'])
                    feat = feat.view(feat.size(0), -1)
                output = F.cosine_similarity(feat.unsqueeze(1), self.model_optms.centroid, dim=-1)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        # save results
        for k, v in results.items():
            tag = '{}/{}'.format(split, k + '_lccs')
            self.write_scalar(tag, v)

        self.save_path = os.path.join(self.output_dir, 'results.jsonl')
        with open(self.save_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

# define trainers

# source classifier
@TRAINER_REGISTRY.register()
class LCCSk1n5(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10, svd_dim=5)

@TRAINER_REGISTRY.register()
class LCCSk1n7(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10, svd_dim=7)
@TRAINER_REGISTRY.register()
class LCCSk5n35(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=35)

@TRAINER_REGISTRY.register()
class LCCSk5n155(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=155,finetune_classifier=True)

@TRAINER_REGISTRY.register()
class LCCSk10n70(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=10, init_epochs=10, grad_update_epochs=10, svd_dim=70)

# mean centroid classifier
@TRAINER_REGISTRY.register()
class LCCSCentroidk1n7(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10, svd_dim=7, classifier_type='mean_centroid')

@TRAINER_REGISTRY.register()
class LCCSCentroidk1n31(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10, svd_dim=310,
                         classifier_type='mean_centroid')

@TRAINER_REGISTRY.register()
class LCCSCentroidk5n10(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=10, classifier_type='mean_centroid')

@TRAINER_REGISTRY.register()
class LCCSCentroidk5n35(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=35, classifier_type='mean_centroid')

@TRAINER_REGISTRY.register()
class LCCSCentroidk5n25(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=25,classifier_type='mean_centroid')

@TRAINER_REGISTRY.register()
class LCCSCentroidk10n50(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=10, init_epochs=10, grad_update_epochs=10, svd_dim=50,classifier_type='mean_centroid')


@TRAINER_REGISTRY.register()
class LCCSCentroidk5n60(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=60, classifier_type='mean_centroid')
@TRAINER_REGISTRY.register()
class LCCSCentroidk10n70(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=10, init_epochs=10, grad_update_epochs=10, svd_dim=70, classifier_type='mean_centroid')