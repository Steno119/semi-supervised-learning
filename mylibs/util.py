import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def kmeans_clustering_loss(cluster_centers, prediction):
    dists = torch.cdist(prediction, cluster_centers)
    assign_u = dists.argmin(dim=1)
    centers_u = cluster_centers[assign_u]
    loss_km = ((prediction - centers_u) ** 2).sum(dim=1).mean()
    return loss_km

def mi_loss(deep_features): # mutual information loss
    batch_size, num_classes = deep_features.shape
    df_probs = nn.Softmax(dim=1)(deep_features)

    marginal_prob = df_probs.mean(dim=0)
    H_y = -torch.sum(torch.log(marginal_prob) * marginal_prob)
    H_y_bar = torch.sum(torch.log(df_probs) * df_probs) / batch_size
    return (H_y - H_y_bar)

def em_loss(logits): # entropy minimization loss
    preds = nn.Softmax(dim=1)(logits)
    
    loss = -torch.mean(torch.sum(preds * torch.log(preds), dim=1))
    return loss

def cr_loss(model, sample_batch):
    _, logits = model(sample_batch)
    preds = nn.Softmax(dim=1)(logits.detach())
    transformations = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(10, translate=(0.1, 0.1)),
    ])
    perturbed_samples = transformations(sample_batch)
    _, p_logits = model(perturbed_samples)
    perturbed_preds = nn.Softmax(dim=1)(p_logits)
    return nn.MSELoss()(preds, perturbed_preds)

class UnlabelledDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        image, _ = self.ds[idx]
        return image, -1

    def __len__(self):
        return len(self.ds)


def split_training_data(dataset, unlabeled_split = 0.95, batch_size=128):
    unlabeled_set_size = 1 if unlabeled_split == 0 else int(len(dataset) * unlabeled_split)
    labeled_set_size = len(dataset) - unlabeled_set_size
    labeled_ds, unlabeled_ds = torch.utils.data.random_split(dataset, [labeled_set_size, unlabeled_set_size])
    unlabeled_ds = UnlabelledDataset(unlabeled_ds)
    labelled_loader = DataLoader(dataset=labeled_ds, batch_size = batch_size, shuffle=True)
    unlabelled_loader = DataLoader(dataset=unlabeled_ds, batch_size = batch_size, shuffle=True)
    return labelled_loader, unlabelled_loader
