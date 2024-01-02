

from torch.utils.data import Dataset, DataLoader
import h5py
import torch
class EdDSA(Dataset):
    def __init__(self, path, img_size, transform=None):
        self.transform = transform



        db = self.load_database(path)



        a_label = list(db["Attack_traces"]["label"])

        a_traces = list(db["Attack_traces"]["traces"])


        p_label =  list(db["Profiling_traces"]["label"])
        p_traces = list(db["Profiling_traces"]["traces"])




        self.traces = a_traces+p_traces
        self.labels = a_label + p_label

        print('Dataset size:', len(self.traces ))


    def load_database(self, path):
        return h5py.File(path, "r")
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]


        trc = self.traces[idx]
        trc = torch.tensor(trc).to(torch.float)
        label = torch.tensor(label).to(torch.int)    
        if self.transform:
            trc = self.transform(trc)
        
        return trc, label