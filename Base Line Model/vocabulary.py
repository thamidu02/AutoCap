from collections import Counter

class Vocabulary:
    """A class to create a vocabulary builder for captions in the dataset"""
    def __init__(self,freq_threshold=5):
        """Here we are considering the freq_threshold=5 from default : meaning if word appear above 5 times we add it to our vocabulary"""
        self.freq_threshold=freq_threshold
        
        self.itos={
            0:"<pad>",
            1:"<start>",
            2:"<end>",
            3:"<unk>",
        }
        
        self.stoi={v:k for k,v in self.itos.items()}
        
        
    def __len__(self):
        return len(self.itos)
    
    
    def build_vocabulary(self,captions):
        frequencies=Counter()
        idx=4
        
        for caption in captions:
            for word in caption.split():
                frequencies.update(word)
                
                if frequencies[word]==self.freq_threshold:
                    self.stoi[word]=idx
                    self.itos[idx]=word
                    idx+=1
                    
                    
    def numericalize(self,text):
        return [
            self.stoi.get(word,self.stoi["<unk>"])
            for word in text
        ]