#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from tqdm import tqdm
import matplotlib.pyplot as plt

class Bootstrap:
    
    def __init__(self, alpha=0.05, boot_samples=1000, statistic=np.mean, random_state=42):
        self.bootstrap_samples = boot_samples
        self.statistic = statistic
        self.random_state = random_state
        self.bootstrap_conf_level = 1 - alpha
        self.left_quant = (1 - self.bootstrap_conf_level) / 2
        self.right_quant = 1 - (1 - self.bootstrap_conf_level) / 2
        
    def __str__(self):
        return f'{type(self)}\n\
bootstrap_samples: {self.bootstrap_samples}\n\
statistic: {self.statistic}\n\
confidence_level: {self.bootstrap_conf_level}\n\
left_quant: {self.left_quant}\n\
right_quant: {self.right_quant}\n\
random_state: {self.random_state}'
        
    def fit(self, sample_a, sample_b):
        np.random.seed(self.random_state)
        boot_len = max([len(sample_a), len(sample_b)])
        self.boot_data = []
    
        for i in tqdm(range(self.bootstrap_samples)): 
            sub_a = np.random.choice(sample_a, size=boot_len, replace = True)
            sub_b = np.random.choice(sample_b, size=boot_len, replace = True)
            self.boot_data.append(self.statistic(sub_a-sub_b)) 

        self.quants = np.quantile(self.boot_data,[self.left_quant, self.right_quant])
    
    def compute(self):
        p_1 = st.norm.cdf(x = 0, loc = np.mean(self.boot_data), scale = np.std(self.boot_data))
        p_2 = st.norm.cdf(x = 0, loc = -np.mean(self.boot_data), scale = np.std(self.boot_data))
        p_value = min(p_1, p_2) * 2
        return p_value
    
    def get_graph(self, title=None):
        _, _, bars = plt.hist(self.boot_data, bins = 50)
        for bar in bars:
            bar.set_edgecolor('white')
            if bar.get_x() <= self.quants[0] or bar.get_x() >= self.quants[1]:
                bar.set_facecolor('#EF553B')
            else: 
                bar.set_facecolor('#636EFA')

        plt.vlines(self.quants,ymin=0,ymax=len(bars),linestyle='--')
        plt.xlabel('differences')
        plt.ylabel('frequency')
        plt.title(title)


def confidence_interval(data, conf_level=0.95,boot_samples=1000,random_state=42,statistic=np.mean,**kwargs):
    np.random.seed(random_state)
    left_quant, right_quant = (1 - conf_level) / 2, 1 - (1-conf_level) / 2
    
    values = []
    for i in (range(boot_samples)):
        subsample = np.random.choice(data, size=len(data), replace=True)
        values.append(statistic(subsample, **kwargs))

    return np.quantile(values, [left_quant, right_quant])
    

def correlation_ratio(categories, values):
    cat = np.unique(categories, return_inverse=True)[1]
    values = np.array(values)
    
    ssw = 0
    ssb = 0
    for i in np.unique(cat):
        subgroup = values[np.argwhere(cat == i).flatten()]
        ssw += np.sum((subgroup-np.mean(subgroup))**2)
        ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    return (ssb / (ssb + ssw))**.5


def cramers_v(rc_table, correction=False):
    
    '''Calculating cramers_v correlation for categorical data. 
       correction=True - Yates' correction
       p_value = chi2_stats[1]
       chi_square_statistic = chi2_stats[0] '''
    
    rc_table = np.array(rc_table)
    n = rc_table.sum()
    chi2_stats = st.chi2_contingency(rc_table, correction=correction)
    cramers_v = (chi2_stats[0]/(n*min(rc_table.shape[0]-1, rc_table.shape[1]-1)))**.5
    return cramers_v, chi2_stats[1], chi2_stats[0]


def robust_mean(data, trunc_level=.2, type_='truncated'):
    data = np.array(data)
    q = np.quantile(data, q=[trunc_level / 2, 1 - trunc_level / 2])
    trunc_data = data[(data > q[0]) & (data < q[1])]
    if type_ == 'truncated':
        return trunc_data.mean()
    elif type_ == 'winsorized':
        return np.clip(data, trunc_data.min(), trunc_data.max()).mean()
