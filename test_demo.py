"""
Test Demo
    ```bash
    python test_demo.py --im_path=data/I03_01_1.bmp
    ```
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from IQADataset import NonOverlappingCropPatches


class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        """data: channel 1 => gray image patch"""
        super(CNNIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)

    def forward(self, x):
        # batch, channel, width, height -> 165,1,32,32
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))

        # 32-3-3 -> 165,50,26,26
        h = self.conv1(x)

        # to 1 -> 165,50,1,1
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        # min pool -> to 1 -> 165,50,1,1
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))

        # stack -> new dim, cat -> no new dim => 165,100,1,1
        h = torch.cat((h1, h2), 1)  # max-min pooling

        # remove 2,3 -> 165,100
        h = h.squeeze(3).squeeze(2)

        # 800 -> 165,800
        h = F.relu(self.fc1(h))
        h = F.dropout(h)

        # 800 -> 165,800
        h = F.relu(self.fc2(h))

        # 1 -> 165,1
        q = self.fc3(h)
        return q


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch CNNIQA test demo")
    parser.add_argument(
        "--im_path", type=str, default="data/I03_01_1.bmp", help="image path"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/CNNIQA-LIVE",
        help="model file (default: models/CNNIQA-LIVE)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNIQAnet(ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800).to(device)

    model.load_state_dict(torch.load(args.model_file, map_location=device))

    # rgb->gray
    im = Image.open(args.im_path).convert("L")
    # im->batch*32*32
    patches = NonOverlappingCropPatches(im, 32, 32)

    model.eval()
    with torch.no_grad():
        patch_scores = model(torch.stack(patches).to(device))
        # patch scores mean -> image quality
        print(patch_scores.mean().item())
