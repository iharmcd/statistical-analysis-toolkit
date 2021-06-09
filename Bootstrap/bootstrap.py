#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
import matplotlib.pyplot as plt

class Bootstrap:
    
    def __init__(self, alpha=0.05, boot_samples=1000, statistic=np.mean,):
        self.bootstrap_samples = boot_samples
        self.statistic = statistic
        self.bootstrap_conf_level = 1 - alpha
        self.left_quant = (1 - self.bootstrap_conf_level) / 2
        self.right_quant = 1 - (1 - self.bootstrap_conf_level) / 2
        
    def __str__(self):
        return f'{type(self)}\n\
bootstrap_samples: {self.bootstrap_samples}\n\
statistic: {self.statistic}\n\
confidence_level: {self.bootstrap_conf_level}\n\
left_quant: {self.left_quant}\n\
right_quant: {self.right_quant}'
        
    def fit(self, sample_a, sample_b):
        boot_len = max([len(sample_a), len(sample_b)])
        self.boot_data = []
        
        state = np.random.RandomState(42)
        for i in tqdm(range(self.bootstrap_samples)): # извлечение подвыборок 
            sub_a = sample_a.sample(boot_len, replace = True, random_state=state).to_numpy()
            sub_b = sample_b.sample(boot_len, replace = True, random_state=state).to_numpy()
            self.boot_data.append(self.statistic(sub_a-sub_b)) 
        
        self.quants = (pd.DataFrame(self.boot_data).quantile([self.left_quant, self.right_quant])
                      .reset_index().rename(index={0:'left', 1:'right'}, columns={'index':'quantiles', 0:'values'}))
    
    def compute(self):
        p_1 = st.norm.cdf(x = 0, loc = np.mean(self.boot_data), scale = np.std(self.boot_data))
        p_2 = st.norm.cdf(x = 0, loc = -np.mean(self.boot_data), scale = np.std(self.boot_data))
        p_value = min(p_1, p_2) * 2
        return p_value
    
    def get_graph(self, title=None):
        _, _, bars = plt.hist(self.boot_data, bins = 50)
        for bar in bars:
            if bar.get_x() <= self.quants['values'][0] or bar.get_x() >= self.quants['values'][1]:
                bar.set_facecolor('red')
            else: 
                bar.set_facecolor('grey')
                bar.set_edgecolor('black')
        
        plt.vlines(self.quants['values'],ymin=0,ymax=len(bars),linestyle='--')
        plt.xlabel('differences')
        plt.ylabel('frequency')
        plt.title(title)


def confidence_interval(data, conf_level=0.95,boot_samples=1000,random_state=42,statistic=np.mean,**kwargs):
    state = np.random.RandomState(random_state)
    left_quant, right_quant = (1 - conf_level) / 2, 1 - (1-conf_level) / 2

    values = []
    for i in (range(boot_samples)):
        subsample = data.sample(frac=1, replace=True, random_state=state).to_numpy()
        values.append(statistic(subsample, **kwargs))

    return np.quantile(values, [left_quant, right_quant])
    

def corr_ratio(categories, values):
    cat = pd.factorize(categories)[0]
    values = np.array(values)
    
    ssw = 0
    ssb = 0
    for i in np.unique(cat):
        subgroup = values[np.argwhere(cat == i).flatten()]
        ssw += np.sum((subgroup-np.mean(subgroup))**2)
        ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    return (ssb / (ssb + ssw))**.5


