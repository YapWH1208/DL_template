class CustomDataset():
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(self.data[idx]) if self.transform else self.data[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __add__(self, other):
        return CustomDataset(self.data + other.data, transform=self.transform)
