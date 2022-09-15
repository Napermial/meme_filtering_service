import torch.nn as nn
from torch import load, cat, device
from os.path import isfile
from os import getcwd
import gdown
from os.path import join
from PIL import Image
from torchvision import transforms
import numpy as np

categories = [
    "Affection",
    "Anger",
    "Annoyance",
    "Anticipation",
    "Aversion",
    "Confidence",
    "Disapproval",
    "Disconnection",
    "Disquietment",
    "Doubt/Confusion",
    "Embarrassment",
    "Engagement",
    "Esteem",
    "Excitement",
    "Fatigue",
    "Fear",
    "Happiness",
    "Pain",
    "Peace",
    "Pleasure",
    "Sadness",
    "Sensitivity",
    "Suffering",
    "Surprise",
    "Sympathy",
    "Yearning",
]


class Emotic(nn.Module):
    def __init__(self, num_context_features, num_body_features):
        super(Emotic, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((self.num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        fuse_features = cat((context_features, body_features), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


def download_dependencies(files: list[str]):
    dependencies = [
        "https://drive.google.com/uc?id=1-1HcXR9AE6LvJb4Bssa7wvIDLCgHTUzW",
        "https://drive.google.com/uc?id=1HJP0PnqK4H--jjZG1fR8m9hXf7dhsVYg",
        "https://drive.google.com/uc?id=1rPg4IH8DIPuCZ2sfsRwAZ_rshB9F7SNq",
    ]

    if all([isfile(join(getcwd(), "processing", file)) for file in files]):
        return

    for dep, file in zip(dependencies, files):
        gdown.download(dep, join(getcwd(), "processing", file), quiet=False)


def predict_emotion(path: str):
    files = [
        "models\\model_context1.pth",
        "models\\model_body1.pth",
        "models\\model_emotic1.pth",
    ]

    download_dependencies(files)
    ind2cat = {}
    for idx, emotion in enumerate(categories):
        ind2cat[idx] = emotion

    transform = transforms.ToTensor()
    input = transform(Image.open(path))
    input = input.unsqueeze(0)
    emotic_model = Emotic(1, 1)

    model_context = load(join(getcwd(), "processing", files[0]))
    model_body = load(join(getcwd(), "processing", files[1]))
    emotic_model = load(
        join(getcwd(), "processing", files[2]), map_location=device("cpu")
    )

    model_context.eval()
    model_body.eval()
    emotic_model.eval()

    pred_body = model_body(input)
    pred_context = model_context(input)
    pred_category, _ = emotic_model(pred_body, pred_context)

    vibes = {}
    for tensor in pred_category.tolist():
        for i, emotion in enumerate(tensor):
            vibes[emotion] = ind2cat[i]

    return [vibes[key] for key in sorted(vibes, reverse=True)[:3] if key > 0.4]


if __name__ == "__main__":
    pass
