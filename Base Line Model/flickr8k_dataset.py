import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from vocabulary import Vocabulary

class Flickr8kDataset(Dataset):
    def __init__(self,root_dir,captions_file,images_file,
                 transform=None,freq_threshold=5):
        
        self.root_dir=root_dir
        self.transform=transform
        
        # splitting image filenames 
        with open (images_file,"r") as f:
            self.images=set(f.read().strip().split("\n"))
            
        self.images_caption_pairs=[]
        captions=[]
        
        with open(captions_file,"r") as f:
            for line in f:
                img,caption=line.strip().split(",",1)
                if img in self.images:
                    caption=caption.lower().strip()
                    self.images_caption_pairs.append((img,caption))
                    captions.append(caption)
                    
        self.vocab=Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(captions)
        
    
    def __len__(self):
        return len(self.images_caption_pairs)
    
    def __getitem__(self,idx):
        img_name,caption=self.images_caption_pairs[idx]
        
        img_path=os.path.join(self.root_dir,img_name)
        image=Image.open(img_path).convert("RGB")
        
        if self.transform:
            image=self.transform(image)
            
        
        numericalized_caption=(
            [self.vocab.stoi["<start>"]]+
            self.vocab.numericalize(caption)+
            [self.vocab.stoi["<end>"]]
        )
        

        return image,torch.tensor(numericalized_caption)
        