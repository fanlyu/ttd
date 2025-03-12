import numpy as np
import copy

from tqdm import tqdm
import torch, torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from core.models.scale.losses import SupConLoss, IRDLoss, similarity_mask_new, similarity_mask_old
from core.models.scale.memory import *
from core.stream.collate import SCALECollateFunction
from core.utils import smart_dir

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import contingency_matrix

from core.models.pcmc.encoders import get_backbone

class SCALE():

    def __init__(self, config):
        
        # declare properties
        self.name = 'SCALE'

        # extract scenario configs
        self.im_size = config.dataset.img_size
        self.num_c = config.dataset.channels
        self.seed = config.seed
        self.n_workers = config.model.n_workers
        self.load_pretrain = config.model.load_pretrain

        # clustering
        self.k_scale = config.model.k_scale

        # Encoder
        self.backbone = get_backbone(config.model.arch, config.model.pretrained).to('cuda')

        # Training
        self.init_epochs = config.model.epochs
        self.bs = config.model.bs
        self.simil = config.model.simil
        self.losses_distil = []
        self.losses_contrast = []
        self.losses = []
        self.distill_power_1 = 0
        self.distill_power = config.model.distil_power
        self.stream_train = config.model.stream_train

        # Augmentations
        self.base_transform = [
            transforms.RandomResizedCrop(size=self.im_size, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomApply([
            #    transforms.ColorJitter(random.uniform(0.6, 1.4), random.uniform(0.6, 1.4), random.uniform(0.6, 1.4), random.uniform(0.9, 1.1))
            #], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std),
        ]

        self.no_aug_transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std)
        ])

        self.pretrain_transform = transforms.Compose(self.base_transform)
        self.stream_transform = transforms.Compose([transforms.Normalize(mean=[-config.dataset.mean[i] / config.dataset.std[i] for i in range(3)] , std=[1 / config.dataset.std[i] for i in range(3)]), transforms.ToPILImage()] + self.base_transform)
                                                   

        self.pretrain_collate = SCALECollateFunction(self.pretrain_transform)
        self.mem_init_collate = SCALECollateFunction(self.no_aug_transform)

        # Memory
        self.mem = Memory(config.model)
        self.mem_samples = config.model.mem_samples

        # Optim
        self.criterion = SupConLoss(config.model.bs, 'resnet18', config.model.temp, config.model.base_temp).to('cuda')
        self.criterion_reg = IRDLoss(self.criterion.projector, config.model.current_temp, config.model.past_temp).to('cuda')
        self.optimizer = torch.optim.SGD(list(self.backbone.parameters()) + list(self.criterion.projector.parameters()), config.model.lr, config.model.momentum, config.model.wd)


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config.model.lr


        # Logging
        self.dataset = config.dataset.name
        self.log = config.log
        self.pretrain_log = config.pretrain_log
        self.iters = 0
        self.step = 1

        self.batch = []


        # stam init
        self.init_layers()

    def init_layers(self):
        
        # random seed
        np.random.seed(self.seed)

    def pretrain(self, loader):
        self.backbone.train()
        dataset = loader.dataset
        sampler = loader.sampler
        epochs = self.init_epochs

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=self.pretrain_collate,
            drop_last=True,
            sampler=sampler,
            num_workers=self.n_workers,
        )

        log = f'logs/{self.pretrain_log}/{self.dataset}/task_0/saved_models/final_model_backbone.pkl'
        log2 = f'logs/{self.pretrain_log}/{self.dataset}/task_0/saved_models/final_model_projector.pkl'
        if self.load_pretrain and os.path.exists(log):
            print('Loading Pretrain')
            self.backbone.load_state_dict(torch.load(log))
            self.criterion.projector.load_state_dict(torch.load(log2))
        else:
            ite = 0
            loss_history = []
            total_loss = 0
            for batch, t in tqdm(trainloader):

                if ite % len(trainloader) == 0:
                    vutils.save_image(batch[0][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_0') + 'aug_0.png', normalize=True)
                    vutils.save_image(batch[1][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_0') + 'aug_1.png', normalize=True)

                self.optimizer.zero_grad()

                loss = self.criterion(self.backbone, self.backbone, batch[0].to('cuda'), batch[1].to('cuda'), None, None)
                
                loss.backward()

                total_loss += loss.item()

                self.optimizer.step()
                ite += 1

                if ite % (len(trainloader) // self.init_epochs) == 0:
                    avg_loss = total_loss / (len(trainloader) // self.init_epochs)
                    total_loss = 0
                    print(f'Epoch: [{ite}/{len(trainloader)}],   Loss: {avg_loss}')
                    loss_history.append(avg_loss)

                    plt.plot(loss_history)
                    plt.title('Average Loss over Training')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_0/layer_{self.name}/figures') + 'loss_history.png')
                    plt.close()
                    
        torch.save(self.backbone.state_dict(), smart_dir(f'logs/{self.log}/{self.dataset}/task_0/saved_models') + f'final_model_backbone.pkl')
        torch.save(self.criterion.projector.state_dict(), smart_dir(f'logs/{self.log}/{self.dataset}/task_0/saved_models') + f'final_model_projector.pkl')

        

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=self.mem_init_collate,
            drop_last=True,
            sampler=sampler,
            num_workers=self.n_workers,
        )

        it = 0 
        for batch, t in trainloader:
            print(it)
            #if it >= (len(trainloader) // self.init_epochs):
            if it >= 15:
                break
            #vutils.save_image(batch[0], 'memory_batch.png')
            self.mem.update_wo_labels(batch[0], torch.tensor([-1 for _ in range(len(batch[0]))]), self.backbone)
            it += 1

        self.backbone.eval()
        print('Done Initializing')


    def __call__(self, x, t):
        if self.stream_train == False:
            return 
        
        print(self.step)
        if self.step % self.bs != 0:
            print('appending to batch')
            self.batch.append(x)
            self.step += 1
            return
        
        print('executing training step')
        self.batch.append(x)
        
        x = torch.cat(self.batch)
        self.batch = []
        
        past_model = copy.deepcopy(self.backbone)
        past_model.eval()


        aug_main_images_0 = torch.stack([self.stream_transform(ee.cpu())
                                            for ee in x])
        aug_main_images_1 = torch.stack([self.stream_transform(ee.cpu())
                                            for ee in x])

        mem_images, mem_labels = self.mem.get_mem_samples()

        # get augmented streaming and augmented samples and concatenate them
        if self.mem_samples > 0 and mem_images is not None:

            sample_cnt = min(self.mem_samples, mem_images.shape[0])
            select_ind = np.random.choice(mem_images.shape[0], sample_cnt, replace=False)

            # Augment memory samples
            aug_mem_images_0 = torch.stack([self.stream_transform(ee.cpu() * 255)
                                            for ee in mem_images[select_ind]])
            aug_mem_images_1 = torch.stack([self.stream_transform(ee.cpu() * 255)
                                            for ee in mem_images[select_ind]])
            feed_images_0 = torch.cat([aug_main_images_0, aug_mem_images_0], dim=0)
            feed_images_1 = torch.cat([aug_main_images_1, aug_mem_images_1], dim=0)
        else:
            feed_images_0 = aug_main_images_0
            feed_images_1 = aug_main_images_1

        print(feed_images_0.shape, feed_images_1.shape)
        if torch.cuda.is_available():
            feed_images_0 = feed_images_0.cuda(non_blocking=True)
            feed_images_1 = feed_images_1.cuda(non_blocking=True)


        feed_images_all = torch.cat([feed_images_0, feed_images_1], dim=0)

        bsz = feed_images_0.shape[0]

        f0_logits, loss_distill = self.criterion_reg(self.backbone, past_model,
                                                feed_images_0)


        #contrast_mask = similarity_mask_new(opt.batch_size, f0_logits, opt, pos_pairs)
        features_all = self.backbone(feed_images_all)
        contrast_mask = similarity_mask_old(features_all, bsz, self.simil, self.bs)
        loss_contrast = self.criterion(self.backbone, self.backbone, feed_images_0, feed_images_1,
                                  mask=contrast_mask)
        

        self.losses_distil.append(loss_distill.item())
        self.losses_contrast.append(loss_contrast.item())
        loss_contrast_avg = sum(self.losses_contrast) / len(self.losses_contrast)
        loss_distil_avg = sum(self.losses_distil) / len(self.losses_distil)


        if self.distill_power_1 <= 0.0 and loss_distill > 0.0:
            self.distill_power_1 = loss_contrast_avg * self.distill_power / loss_distil_avg

        loss = loss_contrast + self.distill_power_1 * loss_distill

        self.losses.append(loss.item())


        # SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.mem.update_wo_labels(x, torch.tensor([-1 for _ in range(len(x))]), self.backbone)

        self.iters += 1
        self.step += 1


    def eval(self, sup_loader, eval_loader, task, it=None):

        print('Supervising...')
        self.supervise(sup_loader, task)

        print('Classifying...')
        class_acc, class_acc_pc = self.classify(eval_loader, task)

        print('Clustering...')
        clust_acc, clust_acc_pc = self.cluster(eval_loader, task)

        return class_acc, class_acc_pc, clust_acc, clust_acc_pc
    
    def supervise(self, loader, task):
        print('DOING SUPERVISION')
        
        embeddings = []
        labels = []
        for it, (x, y) in enumerate(loader):
            embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            labels.append(y.numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        self.knn_1 = KNeighborsClassifier(n_neighbors=1)
        self.knn_1.fit(embeddings, labels)

        self.knn_3 = KNeighborsClassifier(n_neighbors=3)
        self.knn_3.fit(embeddings, labels)

        self.knn_5 = KNeighborsClassifier(n_neighbors=50)
        self.knn_5.fit(embeddings, labels)

    # call classification function
    def classify(self, loader, task):
        print('CLASSIFYING')
        embeddings = []
        labels = []
        for it, (x, y) in enumerate(loader):
            embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            labels.append(y.numpy())
        
        embeddings = np.concatenate(embeddings)
        y_true = np.concatenate(labels)   


        y_pred_1 = self.knn_1.predict(embeddings)
        y_pred_3 = self.knn_3.predict(embeddings)
        y_pred_5 = self.knn_5.predict(embeddings)

        num_classes = len(np.unique(y_true))
        acc1 = np.sum(np.array(y_true) == np.array(y_pred_1)) / len(y_pred_1)

        num_classes = len(np.unique(y_true))
        acc3 = np.sum(np.array(y_true) == np.array(y_pred_3)) / len(y_pred_3)

        num_classes = len(np.unique(y_true))
        acc5 = np.sum(np.array(y_true) == np.array(y_pred_5)) / len(y_pred_5)

        accs = [acc1, acc3, acc5]
        ind = np.argmax(accs)

        acc = accs[ind]

        if ind == 0:
            y_pred = y_pred_1
        elif ind == 1:
            y_pred = y_pred_3
        else:
            y_pred = y_pred_5

        acc_pc = np.zeros(num_classes)
        for y in range(num_classes):
            inds_y = np.argwhere(y_true == y).flatten()
            acc_pc[y] = np.sum(np.array(y_true[inds_y]) == np.array(y_pred[inds_y])) / len(y_true[inds_y])

        # Visualization
        confusion_matrix = np.zeros((num_classes, num_classes))

        for cf in range(len(y_pred)):
            confusion_matrix[int(y_true[cf]), int(y_pred[cf])] += 1

        plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.xticks(np.arange(num_classes))
        plt.yticks(np.arange(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                text = plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")

        plt.savefig(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{task}/iter_{iter}') + 'class_confusion_matrix.png')
        plt.close()
        
        np.save(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{task}/iter_{iter}') + 'class_confusion_matrix.npz', confusion_matrix)
    
        
        return acc * 100, acc_pc * 100

    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        cm = contingency_matrix(y_true, y_pred)

        num_classes = len(np.unique(y_true))

        cluster_labs = np.argmax(cm, axis=0)

        class_perf = np.zeros(num_classes)
        class_counts = np.zeros(num_classes)

        for i, lab in enumerate(cluster_labs):
            class_perf[lab] += cm[lab, i] / np.sum(cm[:, i])
            class_counts[lab] += 1

        for i in range(len(class_counts)):
            class_perf[i] /= class_counts[i]

        # return purity
        acc, pc_acc = np.sum(np.amax(cm, axis=0)) / np.sum(cm), class_perf
        
        #self.model_stats['class_acc'].append(acc)

        return acc, pc_acc
    
    # cluster
    def cluster(self, loader, task):
        print('DOING CLUSTERING')
        
        embeddings = []
        labels = []
        for it, (x, y) in enumerate(loader):
            embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            labels.append(y.numpy())
        
        embeddings = np.concatenate(embeddings)
        y_true = np.concatenate(labels) 
        num_classes = len(np.unique(y_true)) * self.k_scale

        y_pred = KMeans(num_classes).fit_predict(embeddings)

        acc_total, acc_perclass = self.purity_score(y_true, y_pred)

        print('KMEANS: ', acc_total)

        #y_pred = SpectralClustering(num_classes).fit_predict(embeddings)

        #acc_total, acc_perclass = self.purity_score(y_true, y_pred)

        #print('Spectral: ', acc_total)

        return acc_total * 100, acc_perclass * 100


    def plots(self):
        return



