import os
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from mylibs.util import split_training_data, kmeans_clustering_loss, mi_loss, em_loss
from mylibs.resnet import resnet18_c
import matplotlib.pyplot as plt


def train_model(learning_rate, epochs, lam, train_loader, test_loader, model_file, unsup_loss_fn = "km", force_train=False):
    # Set device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Initialize model, loss function, and optimizer
    model = resnet18_c(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    log_dir = f"runs/{model_file}"[:-4]
    writer = SummaryWriter(log_dir=log_dir)

    # Check if model exists
    if os.path.exists(model_file) and not force_train:
        model.load_state_dict(torch.load(model_file))
        model.to(device)
        print("Loaded saved model")
    else:
        # Training loop
        num_batches = len(iter(train_loader))

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            ce_loss = 0
            kmeans_loss = 0
            combined_loss = 0
            correct = 0
            total = 0

            model.train()

            for batch, batch_idx, _ in train_loader:
                # unpack into labelled and unlabelled
                Xl, yl = batch["supervised"]
                Xu, _ = batch["unsupervised"]

                Xl, yl, Xu = Xl.to(device), yl.to(device), Xu.to(device)

                # forward pass on labeled
                zl, logits = model(Xl)
                loss_ce = loss_fn(logits, yl)

                # forward pass on unlabeled
                zu, logits_u = model(Xu)

                match unsup_loss_fn:
                    case 'km':
                        # k-means clustering loss
                        loss_unsup = kmeans_clustering_loss(model.cluster_centers, zu)
                    case 'mi':
                        # mutual information loss
                        loss_unsup = mi_loss(zu)
                    case 'em':
                        # entropy minimization loss
                        loss_unsup = em_loss(logits_u)
                    case 'cr':
                        loss_unsup = 0
                    case _:
                        loss_unsup = 0

                # combined loss
                loss = loss_ce + lam * loss_unsup

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                kmeans_loss += loss_unsup.item()
                ce_loss += loss_ce.item()
                combined_loss += loss.item()
                # Supervised accuracy
                _, predicted = logits.max(1)
                total += yl.size(0)
                correct += predicted.eq(yl).sum().item()
                if batch_idx % 100 == 0:
                    print(f"Batch: {batch_idx + 1}/{num_batches}, "
                          f"CE Loss: {loss_ce.item():.4f}, "
                          f"Unsupervised Loss: {loss_unsup.item():.4f}, "
                          f"Combined Loss: {loss.item():.4f}, "
                          f"Supe Acc: {100 * correct / total:.2f}%")

            # Log training metrics
            train_loss = combined_loss / num_batches
            train_acc = correct / total
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)

            # Test
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    x, pred = model(X)
                    loss = loss_fn(pred, y)

                    test_loss += loss.item()
                    _, predicted = pred.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

            # Log test metrics
            test_loss = test_loss / len(test_loader)
            test_acc = correct / total
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)

            print(f"Test Loss: {test_loss:.4f}, Test Acc: {100 * test_acc:.2f}%")

            # Log model weights and gradients
            for name, param in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

            # Save model
            torch.save(model.state_dict(), model_file)
            writer.flush()

        writer.close()
        print("Training completed!")


def visualize_predictions(model, dataset, device, num_samples=9):
    # CIFAR-10 label names
    label_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # Create DataLoader with batch_size=1
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get samples
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 3, 3

    # CIFAR-10 normalization parameters - need these to denormalize
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    for i in range(min(num_samples, len(loader))):
        X, y_true = next(iter(loader))
        X, y_true = X.to(device), y_true.to(device)

        with torch.no_grad():
            # Forward pass
            deep_features, pred = model(X)
            y_pred = pred.argmax(1).item()

        # Plot
        ax = figure.add_subplot(rows, cols, i + 1)
        pred_label = label_names[y_pred]
        true_label = label_names[y_true.item()]
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}")

        # Denormalize the image for display
        img = X.cpu().squeeze()  # Get the image tensor
        img = img * std + mean  # Reverse the normalization
        img = torch.clamp(img, 0, 1)  # Clamp values to valid range [0,1]
        img = img.permute(1, 2, 0)  # Change from [C,H,W] to [H,W,C] for display

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    learning_rate = 1e-3
    BATCH_SIZE = 128
    epochs = 20
    TRAIN_TEST_SPLIT = 0.9
    LABELLED_SPLIT = 0.5
    lam = 0.1
    unsup_loss_fn = "mi"
    model_file = f"resnet18_cifar10_{unsup_loss_fn}_{int(LABELLED_SPLIT * 100)}.pth"

    # Train or load the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    dataset = datasets.CIFAR10(
        root="images",
        train=True,
        download=True,
        transform=transform
    )

    # split dataset into training and test set
    train_set_size = int(len(dataset) * TRAIN_TEST_SPLIT)
    test_set_size = len(dataset) - train_set_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_set_size, test_set_size])
    test_loader = DataLoader(dataset=test_ds, batch_size = BATCH_SIZE)

    # 5% labeled vs unlabeled DataLoader
    labeled_loader, unlabeled_loader = split_training_data(dataset=train_ds, unlabeled_split=1 - LABELLED_SPLIT)
    train_loader = CombinedLoader({
        "supervised": labeled_loader,
        "unsupervised": unlabeled_loader,
    }, mode="max_size_cycle")

    # Train or load the model
    train_model(learning_rate, epochs, lam, train_loader, test_loader, model_file, unsup_loss_fn=unsup_loss_fn, force_train=False)

    # Load the model for visualization
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = resnet18_c(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Setup test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Visualize predictions
    visualize_predictions(model, test_data, device)

