import os
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset


class dataset(Dataset):
    def __init__(
        self,
        metadata_dir,
        image_dir,
        split
    ):
        super().__init__()
        self.data = load_dataset(metadata_dir, split=split)
        self.image_dir = image_dir
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __getitem__(self, index):
        d = self.data[index]
        image = Image.open(os.path.join(self.image_dir, d['filename'])).convert("RGB")
        return {
            'sentid': str(100000000+d['sentid'])[1:],
            'image': self.preprocess(image),
            'sentence': d['sentence']# + ' on white background.'
        }
    
    def __len__(self):
        return len(self.data)

