import os

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('medium')

import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from mylibs.util import em_loss, cr_loss, mi_loss

import lightning as L

import matplotlib.pyplot as plt
import numpy as np

transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    # tv.transforms.RandomHorizontalFlip(0.1), 
    # tv.transforms.RandomVerticalFlip(0.1), 
    tv.transforms.Normalize(mean=[0.1307], std=[0.3081])
])

dataset = tv.datasets.MNIST(
    root="images",
    train=True,
    download=True,
    transform=transforms
)

class UnlabelledDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        image, _ = self.ds[idx]
        return image, -1

    def __len__(self):
        return len(self.ds)

TRAIN_TEST_SPLIT = 0.8
LABELLED_SPLIT = 0.05
BATCH_SIZE = 256

train_ds, test_ds = torch.utils.data.random_split(dataset, [int(len(dataset) * TRAIN_TEST_SPLIT), len(dataset) - int(len(dataset) * TRAIN_TEST_SPLIT)])
train_labelled_ds, train_unlabelled_ds = torch.utils.data.random_split(train_ds, [int(len(train_ds) * LABELLED_SPLIT), len(train_ds) - int(len(train_ds) * LABELLED_SPLIT)])
train_unlabelled_ds = UnlabelledDataset(train_unlabelled_ds)
train_labelled_loader = DataLoader(dataset=train_labelled_ds, batch_size = BATCH_SIZE, shuffle=True)
train_unlabelled_loader = DataLoader(dataset=train_unlabelled_ds, batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size = BATCH_SIZE)

combined_loader = CombinedLoader({
    "supervised": train_labelled_loader,
    "unsupervised": train_unlabelled_loader
}, mode="max_size_cycle")

class SemiSupervisedClassifier(L.LightningModule):
    def __init__(self, num_classes=10, lamb=50, warmup_idx=30, loss_fcn: str = "mi"):
        super().__init__()

        # TODO: need to define the structure of the model here
        self.trainer = None
        self.num_correct=0
        self.loss_fcn = loss_fcn
        self.predictions = 0
        self.warmup_index = warmup_idx
        self.lamb = lamb
        self.sup_losses = []
        self.unsup_losses = []
        self.total_losses = []
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear_model = nn.Sequential(
            nn.Linear(in_features=1600, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=num_classes),
        )
        self.supervised_criterion = nn.CrossEntropyLoss()
    
    def forward(self, input):
        return self.linear_model(self.conv_model(input))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        return [optimizer], [scheduler]
    
    def _calculate_loss_supervised(self, batch_iterator): # calculates cross entropy
        if batch_iterator is None:
            return torch.tensor(0)
        image_batch, label_batch = batch_iterator
        prediction_batch = self(image_batch)
        loss = self.supervised_criterion(prediction_batch, label_batch)
        return loss
    
    def _mi_loss(self, batch_iterator): # calculates mutual information loss
        if batch_iterator is None:
            return torch.tensor(0)
        image_batch, _ = batch_iterator
        output_batch = self.conv_model(image_batch)
        batch_size, num_classes = output_batch.shape
        prediction_batch = nn.Softmax(dim=1)(output_batch)

        marginal_prob = prediction_batch.mean(dim=0)
        H_y = -torch.sum(torch.log(marginal_prob) * marginal_prob)
        H_y_bar = torch.sum(torch.log(prediction_batch) * prediction_batch) / batch_size
        return self.lamb * (H_y - H_y_bar)
    
    def _em_loss(self, batch_iterator): # calculates entropy minimization loss
        if batch_iterator is None:
            return torch.tensor(0)
        image_batch, _ = batch_iterator
        output_batch = self(image_batch)
        prediction_batch = nn.Softmax(dim=1)(output_batch)
        
        loss = -torch.mean(torch.sum(prediction_batch * torch.log(prediction_batch), dim=1))
        return loss
    
    def _cr_loss(self, batch_iterator): # calculates consistency regularization loss
        if batch_iterator is None:
            return torch.tensor(0)
        image_batch, _ = batch_iterator
        transformations = tv.transforms.Compose([
            tv.transforms.RandomRotation(15),
            tv.transforms.RandomAffine(10, translate=(0.1, 0.1)),
        ])
        perturbed = transformations(image_batch)
        pred_batch = nn.functional.softmax(self(image_batch).detach(), dim=1)
        pred_batch_perturbed = nn.functional.softmax(self(perturbed), dim=1)

        loss = nn.functional.mse_loss(pred_batch, pred_batch_perturbed)
        return loss
    
    def _calculate_loss_unsupervised(self, batch_iterator):
        img_batch, label_batch = batch_iterator
        match self.loss_fcn:
            case 'mi':
                # mutual information loss
                loss_unsup = mi_loss(self.conv_model(img_batch))
            case 'em':
                # entropy minimization loss
                loss_unsup = em_loss(self(img_batch))
            # case 'cr':
            #     loss_unsup = cr_loss(self, batch_iterator)
            # This functionality is depricated for this model as it changed after this model was done being developed

            # Kmeans was never implemented for this model
            case _:
                loss_unsup = 0

        return loss_unsup
    
    def training_step(self, batch_iterator, batch_idx):
        supervised_batch = batch_iterator['supervised']
        unsupervised_batch = batch_iterator['unsupervised']

        supervised_loss = self._calculate_loss_supervised(supervised_batch)
        unsupervised_loss = self._calculate_loss_unsupervised(unsupervised_batch)

        total_loss = supervised_loss + unsupervised_loss
        self.sup_losses.append(supervised_loss.item())
        self.unsup_losses.append(unsupervised_loss.item())
        self.total_losses.append(total_loss.item())

        self.log(f'Training loss', total_loss, prog_bar=True)
        return total_loss
    
    def test_step(self, batch_iterator, batch_idx):
        image_batch, label_batch = batch_iterator
        output = self(image_batch)
        output_labels = torch.argmax(output, dim=1)
        diff = output_labels == label_batch
        self.num_correct += diff.sum().item()
        self.predictions+=output_labels.shape[0]
        batch_loss = self._calculate_loss_supervised(batch_iterator)
        self.log('Test loss', batch_loss, on_epoch=True)
    
    def on_train_epoch_end(self):
        # optional, reset any collected metrics here
        pass
    
    def on_test_epoch_end(self):
        self.log('Test accuracy', self.num_correct/self.predictions, on_epoch=True)
        self.num_correct = 0
        self.predictions = 0


    def train_model(self, train_loader: DataLoader):
        self.trainer = L.Trainer(max_epochs=5, fast_dev_run=False)
        self.trainer.fit(model=self, train_dataloaders=train_loader)
    
    def test_model(self, test_loader: DataLoader, plot_loss: bool = False):
        self.trainer.test(model=self, dataloaders=test_loader)
        if plot_loss:
            steps = np.arange(0, len(self.total_losses))
            plt.plot(steps,np.array(self.sup_losses), label = "supervised")
            plt.plot(steps,np.array(self.unsup_losses), label = "unsupervised")
            plt.plot(steps,np.array(self.total_losses), label = "total")
            plt.legend()
            plt.show()



        
# Training
if __name__ == '__main__':
    FAST_DEV_RUN = False # tests model on only one single input if set to true
    LOAD_FROM_CHECKPOINT = False
    VERSION_NUM = 50
    PLOT_LOSSES = True

    loss_fcn = "mi"
    
    checkpoint_path = f"lightning_logs/version_{VERSION_NUM}/checkpoints"
    trainer = L.Trainer(max_epochs=5, fast_dev_run=FAST_DEV_RUN)

    if LOAD_FROM_CHECKPOINT:
        checkpoint_name = os.listdir(checkpoint_path)[0]
        classifier = SemiSupervisedClassifier.load_from_checkpoint(os.path.join(checkpoint_path, checkpoint_name))
    else:
        classifier = SemiSupervisedClassifier(loss_fcn=loss_fcn)
        trainer.fit(model=classifier, train_dataloaders=combined_loader) 

    trainer.test(model=classifier, dataloaders=test_loader) 

    if PLOT_LOSSES:
        steps = np.arange(0, len(classifier.total_losses))
        plt.plot(steps,np.array(classifier.sup_losses), label = "supervised")
        plt.plot(steps,np.array(classifier.unsup_losses), label = "unsupervised")
        plt.plot(steps,np.array(classifier.total_losses), label = "total")
        plt.legend()

        plt.show()
