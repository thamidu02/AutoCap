import torch
import torchvision.transforms.v2 as T

toTensor=T.Compose([
    T.Resize((224,224)),
    T.ToImage(),
    T.ToDtype(torch.float32,scale=True),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
