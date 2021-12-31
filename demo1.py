import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self._cols_int = ['I{}'.format(idx) for idx in range(1,14)]
        self._cols_category = ['C{}'.format(idx) for idx in range(1,27)]
        
        df = pd.read_csv("criteo_sampled_data.csv",sep=',')
        df.columns = ['label'] + self._cols_int + self._cols_category
        
        # 建立一个dict,将指定列中的所有NaN元素分别替换为对应值
        # values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # df.fillna(value=values)
        df.fillna(value={c:'' for c in self._cols_category},inplace=True)
        
        self._y = df['label']
        self._X = df.loc[:,df.columns != 'label']
        self._proproced_df = None
        
    def _normalize_numerics(self, upper_bound):
        numeric_features = self._X.loc[:, self._cols_int].copy()
        # axis=1,按列应用upper_bound,大于upper_bound，这将该值截断为upper_bound
        numeric_features.clip_upper(upper_bound, axis=1, inplace=True)

        # I2有小于0的值
        # -1    204968(占10%左右)
        # -2      1229
        # -3         1
        numeric_features['I2'] = (numeric_features['I2'] + 1).clip_lower(0)
        
        numeric_features = np.log1p(numeric_features)

        # 既然不能用zero mean,unit variance scaling，因为那样会破坏数据的稀疏性
        # 最简单的就是用min-max-scaling
        # 理论做法，应该先split train and test，再做scaling
        # 这里就不那么讲究了，差别也没有那么大
        # (因为numeric_features不是ndarray，没有被minmax_scale inplace modify的可能性，也就没设置copy=False)
        col_min = numeric_features.min()
        col_max = numeric_features.max()
        return (numeric_features - col_min) / (col_max - col_min)
    
    def _build_catval_vocab(self,min_occur):    #对于多个field构建统一的vocab
        vocab = []
        for c in self._cols_category:
            cat_counts = self._X[c].value_counts()
            valid_catcounts = cat_counts.loc[cat_counts>=min_occur] 
            
            vocab.extend('{}/{}'.format(c,tag) for tag in valid_catcounts.index)  
        
        return {tag:idx for idx,tag in enumerate(vocab,start=1)}
    
df = pd.read_csv("./criteo_sampled_data.csv")
# cat_counts = df['label'].value_counts()
# print(cat_counts.loc[df['label'].value_counts()>=100].values)

