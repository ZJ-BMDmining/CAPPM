# from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
# import soundfile as sf
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler

def make_img(t_img):
    img = pd.read_pickle(t_img)
    img_l = []
    for i in range(len(img)):
        img_l.append(img.values[i][0])
    
    return np.array(img_l)


class ADNI(torch.utils.data.Dataset):
    def __init__(self, split='train', datapath='../../processed_data/ADNI/overlap/'):
        standard_scaler = StandardScaler()
        self.base_path = datapath
        self.split = split

        self.csv_data = pd.DataFrame(standard_scaler.fit_transform(pd.read_csv(self.base_path +'X_'+self.split + '_plasma.csv').drop("ID_Visit", axis=1)))
        # self.csv_data = pd.read_csv(self.base_path +'X_'+self.split + '_plasma.csv').drop("ID_Visit", axis=1)

        self.pkl_data = make_img(self.base_path +'X_'+self.split + '_img.pkl')

        self.label=pd.read_csv(self.base_path +'y_'+self.split + '.csv').drop("ID_Visit", axis=1)


    def __getitem__(self, index):
        # print()
        features = self.csv_data.iloc[index, :].values.astype(np.float32)
        image = self.pkl_data[index]
        label=self.label.iloc[index, :].values.astype("int").flatten().squeeze()
        # print(f"Index: {index}, Image shape: {image.shape if image is not None else 'None'}, Features: {features.shape}, Label: {label}")

        return torch.tensor(image, dtype=torch.float),torch.tensor(features, dtype=torch.float), torch.tensor(label, dtype=torch.long)

            

    def __len__(self):
        return len(self.csv_data)
    
    def get_csvshape(self):
        return self.csv_data.shape[1]
    
    def get_true_label(self):
        return self.label.values.astype("int").flatten().squeeze()
