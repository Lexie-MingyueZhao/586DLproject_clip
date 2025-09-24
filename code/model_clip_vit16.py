import torch
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import json

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
#define functions
def build_dataloader(dataset_name, batch_size=64):
    """Load dataset by name."""
    transform = preprocess

    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        classnames = dataset.classes
    elif dataset_name == "food101":
        dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)
        classnames = dataset.classes
    elif dataset_name == "eurosat":
        dataset = datasets.EuroSAT(root="./data", download=True, transform=transform)
        classnames = dataset.classes
    else:
        # For datasets like ImageNet using folder-based structure
        data_root = f"./data/{dataset_name}"
        dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=transform)
        classnames = dataset.classes

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, classnames

def generate_text_features(classnames,dataset_name):
    """Tokenize and encode class names."""
    if dataset_name == "imagenet-mini":
        with open("./data/imagenet-mini/imagenet_class_index.json", "r") as f:
            class_index = json.load(f)
        prompts = [f"a photo of a {class_index[str(i)][1].replace('_', ' ')}" for i in range(len(class_index))]
    else:
        prompts = [f"a photo of a {c.replace('_', ' ')}" for c in classnames]

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def evaluate_zero_shot(loader, classnames,dataset_name):
    """Evaluate zero-shot top-1 and top-5 accuracy."""
    text_features = generate_text_features(classnames, dataset_name)
    top1, top5, total = 0, 0, 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T

            top1_preds = similarity.argmax(dim=1)
            top5_preds = similarity.topk(5, dim=1).indices

            top1 += (top1_preds == labels).sum().item()
            top5 += sum([label in top5_preds[i] for i, label in enumerate(labels)])
            total += labels.size(0)

    top1_acc = top1 / total * 100
    top5_acc = top5 / total * 100
    print(f"[Zero-shot] Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"[Zero-shot] Top-5 Accuracy: {top5_acc:.2f}%")
    return top1_acc, top5_acc

def evaluate_linear_probe(loader, dataset_name):
    """Evaluate linear probe using logistic regression with train/test split."""
    X, y = [], []
    for images, labels in tqdm(loader):
        images = images.to(device)
        with torch.no_grad():
            features = model.encode_image(images)
            X.append(features.cpu())
            y.append(labels)

    X = torch.cat(X).numpy()
    y = torch.cat(y).numpy()

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear classifier
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100

    print(f"[Linear Probe] Train Accuracy: {train_acc:.2f}%")
    print(f"[Linear Probe] Test Accuracy:  {test_acc:.2f}%")

    return train_acc, test_acc

def evaluate_dataset(dataset_name, method="zero"):
    """Main entry point for evaluation."""
    loader, classnames = build_dataloader(dataset_name)

    if method == "zero":
        return evaluate_zero_shot(loader, classnames, dataset_name)
    elif method == "linear":
        return evaluate_linear_probe(loader, dataset_name)
    else:
        raise ValueError("Method must be 'zero' or 'linear'")
