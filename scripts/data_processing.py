import numpy as np
import random
import soundfile as sf

from torch.utils.data import Dataset
from egg_exp import signal_processing

class TrainSet(Dataset):
    def __init__(self, args, items, p_gaussian_noise):
        self.items = items
        self.p_gaussian_noise = p_gaussian_noise
        self.crop_size = args['num_train_frames']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample
        item = self.items[index]
        
        # read wav
        audio = signal_processing.rand_crop_read(item.path, self.crop_size)
        
        if len(np.shape(audio)) == 1:
            audio = np.repeat(audio, 2).reshape(-1, 2)
            
        if audio.shape[0] < self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, ((0, shortage), (0, 0)), 'wrap')
        
        audio = np.transpose(audio) 
        assert audio.shape[0] == 2 and audio.shape[1] == self.crop_size, f'{np.shape(audio)}'
        
        return audio.astype(np.float), item.label

class ValidationSet(Dataset):
    def __init__(self, args, items):
        self.items = items
        self.crop_size = args['num_train_frames']
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        audio, _ = sf.read(item.path)
        
        if len(np.shape(audio)) == 1:
            audio = np.repeat(audio, 2).reshape(-1, 2)
        
        if audio.shape[0] < self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, ((0, shortage), (0, 0)), 'wrap')
        
        audio = audio[:self.crop_size]
        audio = np.transpose(audio)
        assert audio.shape[0] == 2 and audio.shape[1] == self.crop_size, f'{np.shape(audio)}'
            
        return audio.astype(np.float), item.label
    
class EvaluationSet(Dataset):
    def __init__(self, args, items):
        self.items = items
        self.crop_size = args['num_train_frames']
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        # sample
        item = self.items[index]

        # read wav
        audio, _ = sf.read(item.path)
        
        if len(np.shape(audio)) == 1:
            audio = np.repeat(audio, 2).reshape(-1, 2)
            
        audio = np.transpose(audio)
            
        return audio.astype(np.float), item.label