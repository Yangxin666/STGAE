from sklearn.impute import KNNImputer
import random
import copy
import pandas as pd
import torch

def augment(D_0):
    imputer = KNNImputer(n_neighbors=5)
    df = copy.deepcopy(D_O)
    df = pd.DataFrame(df)
    imputer.fit(df)
    D_A = imputer.transform(df)
    D_A = pd.DataFrame(D_A)
    return D_A

# D_A.to_csv('gdrive/My Drive/STD-GAE/data/Data_Augmented.csv', index = False)
# D_A

def mask(D_0, missing_type, missing_severity):
    m = D_O.shape[0]
    n = D_O.shape[1]
    if missing_type == "MCAR":
        corruption_mask = torch.FloatTensor(m, n).uniform_() > missing_severity
    else:
        length = missing_severity
        corruption_mask = torch.full((m, n), True)
        for i in range(int(m/288)):
            for j in range(n):
                number = random.randint(288*i+1,288*(i+1)-length-1)
                corruption_mask[number:number+length, j] = False
    return corruption_mask

# pd.DataFrame(mask.numpy()).to_csv('gdrive/My Drive/STD-GAE/data/2hr_BM.csv',index=False)


#Choose the mask you want to corrupt D_A: here we choose 12hrs BM
def corrupt(corruption_mask, D_A):
    # mask = pd.read_csv('gdrive/My Drive/STD-GAE/data/12hr_BM.csv')
    mask = torch.tensor(corruption_mask.values)
    D_C = copy.deepcopy(D_A)
    D_C[mask.numpy()==False] = -1
    return D_C


# D_A = pd.read_csv('gdrive/My Drive/STD-GAE/data/Data_Augmented.csv')