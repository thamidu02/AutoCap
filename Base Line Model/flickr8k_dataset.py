import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from vocabulary import Vocabulary

class Flickr8kDataset(Dataset):
    def __init__(self,root_dir,captions_file,
                 transform=None,freq_threshold=5):
        
        self.root_dir=root_dir
        self.transform=transform
        
        self.image_captions_pairs=[]
        captions=[]
        
        # splitting image filenames 
        with open (captions_file,"r") as f:
            for line in f:
                line=line.strip()
                
                if not line or "," not in line:
                    continue
                
                img,caption=line.split(",",1)
                caption=caption.lower().strip()
                
                self.image_captions_pairs.append((img,caption))
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
        