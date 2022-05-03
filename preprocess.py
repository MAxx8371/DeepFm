import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json

class DataPreprocessor:
  def __init__(self):
    self._cols_int = ['I{}'.format(idx) for idx in range(1,14)]
    # self._cols_int = ['I1']
    self._multi_cols_category = ['C{}'.format(idx) for idx in range(1,4)]   #category前三列组成多值特征
    self._single_cols_category = ['C{}'.format(idx) for idx in range(4,27)]
    # self._single_cols_category = ['C5']
    
    df = pd.read_csv("./criteo_sampled_data.csv")
    # df = pd.read_csv("E:\DeepFM\demo.csv")
    # df.columns = ['label'] + self._cols_int + self._multi_cols_category + self._single_cols_category
    
    # 建立一个dict,将指定列中的所有NaN元素分别替换为对应值
    # values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # df.fillna(value=values)
    df.fillna(value={c:'' for c in self._single_cols_category},inplace=True)
    df.fillna(value={c:'' for c in self._multi_cols_category},inplace=True)
      
    self._y = df['label']
    self._X = df.loc[:,df.columns != 'label']
    self._proproced_df = pd.DataFrame()
      
  def _normalize_numerics(self, upper_bound):
    numeric_features = self._X.loc[:, self._cols_int].copy()
    # axis=1,按列应用upper_bound,大于upper_bound，这将该值截断为upper_bound
    numeric_features.clip(upper=upper_bound, axis=1, inplace=True)

    # I2有小于0的值
    # -1    204968(占10%左右)
    # -2      1229
    # -3         1
    numeric_features['I2'] = (numeric_features['I2'] + 1).clip(lower=0)
    
    numeric_features = np.log1p(numeric_features)

    # 既然不能用zero mean,unit variance scaling，因为那样会破坏数据的稀疏性
    # 最简单的就是用min-max-scaling
    # 理论做法，应该先split train and test，再做scaling
    # (因为numeric_features不是ndarray，没有被minmax_scale inplace modify的可能性，也就没设置copy=False)
    # col_min = numeric_features.min()
    # col_max = numeric_features.max()
    # numeric_features = (numeric_features - col_min) / (col_max - col_min)
    
    #对数值特征进行离散化
    labels = ['{}'.format(i) for i in range(1,11)]
    for name in self._cols_int:
      _, binedge = pd.cut(numeric_features[name].dropna(),10,retbins=True)
      # discreted_feat = pd.cut(numeric_features[name],binedge,labels=labels).cat.add_categories([0])
      discreted_feat = pd.cut(numeric_features[name],binedge,labels=labels)
      # numeric_features[name].to_csv("dataset\demo.csv")
      # discreted_feat.to_csv("dataset\demo.csv")
      numeric_features[name] = discreted_feat
    
    return numeric_features
  
  def _build_catval_vocab(self,min_occur):    #构建vocab [multi_cal, ...single_cal...]
    tag2idx = []
    dict_size = {}

    vocab = []
    for c in self._multi_cols_category:
        cat_counts = self._X[c].value_counts()        #记录在一列中出现过的tag以及tag出现的次数
        valid_catcounts = cat_counts.loc[cat_counts>=min_occur] 
        
        vocab.extend('{}/{}'.format(c,tag) for tag in valid_catcounts.index)  
        # idx从1开始计数，1号位置为'罕见的tag'(出现不足min_occur次)预留, 0号位置留给Nan值
    tag2idx.append({tag:idx for idx,tag in enumerate(vocab,start=2)})
    dict_size["multi_feats"] = len(vocab)
        
    for c in self._single_cols_category:
      vocab = []
      cat_counts = self._X[c].value_counts()
      valid_catcounts = cat_counts.loc[cat_counts>=min_occur]
      vocab.extend('{}/{}'.format(c,tag) for tag in valid_catcounts.index)
      
      tag2idx.append({tag:idx for idx,tag in enumerate(vocab,start=2)})
      dict_size[c] = len(vocab)

    #dict_size写入json
    with open('./list.json','w') as f:
      json.dump(dict_size,f)
      f.close()

    return tag2idx
  
  # def _transform_numerical_row(self, row, idx):
  #   txts = []
  #   value = row[idx]
    
  #   if ~np.isnan(value) and abs(value)>1e-6:
  #       txts.append("{}:{:.6f}".format(idx,value)) 
        
  #   return ",".join(txts)
  
  def _transform_numerical_row(self, row, idx):
    txts = []
    value = row[idx]
    
    if value is '':
      return ",".join(txts)
    
    # cat = "{}/{}".format(idx,value)
    txts.append("{:.0f}:1".format(value))
    
    return ",".join(txts)
  
  def _transform_categorical_row(self,row,tag2idx,*args):
    txts = []
    index = args[0]
    if(isinstance(index,str)):
        index = [index]
    for c in index:
        tag = row[c]
        if len(tag)==0:
            continue
        
        idx = tag2idx.get("{}/{}".format(c,tag),1)
        txts.append("{}:1".format(idx))
    
    return ",".join(txts)
  
  def run(self,int_upper_bound,cat_min_occur):
    #qcut后为categorical类型,导致处理速度缓慢
    normed_numeric_feats = self._normalize_numerics(int_upper_bound)
    self._proproced_df['label'] = self._y

    tag2idx = self._build_catval_vocab(cat_min_occur)
    cnt = []
    for dict in tag2idx:
      cnt.append(len(dict))
      
    print(cnt)
    
    # normed_numeric_feats.to_string()
    normed_numeric_feats.to_csv("duplicate.csv")
    normed_numeric_feats = pd.read_csv("duplicate.csv").fillna('')
    os.remove("duplicate.csv")
        
    for idx, _col_category in enumerate(self._cols_int):
        self._proproced_df[_col_category] = normed_numeric_feats.progress_apply(lambda row:self._transform_numerical_row(row,_col_category),axis=1)
    
    print(cnt,len(cnt))
    self._proproced_df['multi_feats'] = self._X.progress_apply(lambda row:self._transform_categorical_row(row,tag2idx[0],self._multi_cols_category),
                                      axis=1)
    
    for (index,_col_category) in enumerate(self._single_cols_category,start=1):  
        self._proproced_df[_col_category] = self._X.progress_apply(lambda row:self._transform_categorical_row(row,tag2idx[index],_col_category),
                            axis=1)
      
  def split_save(self,test_ratio=0.2,sample_ratio=1.0):
    df = self._proproced_df
    if sample_ratio<1:
        df = self._proproced_df.sample(frac=sample_ratio)
    
    train_df,test_df = train_test_split(df,test_size=test_ratio)
    
    out_dir = "./dataset/_train.csv"
    train_df.to_csv(out_dir,sep='\t',index=False)
    # df.to_csv(out_dir,sep='\t',index=False)
    
    out_dir = "./dataset/_test.csv"
    test_df.to_csv(out_dir,sep='\t',index=False)
        
if __name__ == "__main__":
  tqdm.pandas()
  preproc = DataPreprocessor()
  
  # int_upper_bound=[20]
  int_upper_bound=[20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
  cat_min_occur = 200
  
  preproc.run(int_upper_bound=int_upper_bound,cat_min_occur=cat_min_occur)
  preproc.split_save()
    
# df = pd.read_csv("./criteo_sampled_data.csv")
# cat_counts = df['label'].value_counts()
# print(cat_counts.loc[df['label'].value_counts()>=100].values)