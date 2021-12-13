#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

class Bootstrap:
    
    '''https://en.wikipedia.org/wiki/Bootstrapping_(statistics)'''

    def __init__(self, alpha=0.05, boot_samples=1000, statistic=np.mean, random_state=42):
        self.bootstrap_samples = boot_samples
        self.statistic = statistic
        self.random_state = random_state
        self.bootstrap_conf_level = 1 - alpha
        self.left_quant = (1 - self.bootstrap_conf_level) / 2
        self.right_quant = 1 - (1 - self.bootstrap_conf_level) / 2
                
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
    
    '''https://en.wikipedia.org/wiki/Correlation_ratio
       ssw: sum of squares within groups
       ssb: sum of squares between groups'''
    
    cat = np.unique(categories, return_inverse=True)[1]
    values = np.array(values)
    
    ssw = 0
    ssb = 0
    for i in np.unique(cat):
        subgroup = values[np.argwhere(cat == i).flatten()]
        ssw += np.sum((subgroup-np.mean(subgroup))**2)
        ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    return (ssb / (ssb + ssw))**.5


def cramers_v(rc_table):
    
    '''Calculating cramers_v correlation for categorical data. 
       https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
       correction: True - Yates' correction'''
    
    rc_table = np.array(rc_table)
    
    if rc_table.min() < 5:
        raise ValueError('Not enough observations')
    else:
        if rc_table.min() < 10:
            correction = True
        else:
            correction = False
            
        n = rc_table.sum()
        chi2_stats = st.chi2_contingency(rc_table, correction=correction)
        cramers_v = (chi2_stats[0]/(n*min(rc_table.shape[0]-1, rc_table.shape[1]-1)))**.5
        stat = namedtuple('CramersvResult', ('corr', 'pvalue','chi2'))
        return stat(cramers_v, chi2_stats[1], chi2_stats[0])


def robust_mean(data, trunc_level=.2, type_='truncated'):
    
    '''https://en.wikipedia.org/wiki/Truncated_mean
       https://en.wikipedia.org/wiki/Winsorized_mean'''
    
    data = np.array(data)
    q = np.quantile(data, q=[trunc_level / 2, 1 - trunc_level / 2])
    trunc_data = data[(data > q[0]) & (data < q[1])]
    if type_ == 'truncated':
        return trunc_data.mean()
    elif type_ == 'winsorized':
        return np.clip(data, trunc_data.min(), trunc_data.max()).mean()


def events_sum(probas: list) -> float:
    if probas[0] < 0 or probas[0] > 1:
        raise ValueError("probability can't take negative values or values greater than one")
    if len(probas) == 1:
        return probas[0]
    else:
        second_element = events_sum(probas[1:])
        return probas[0] + second_element - (probas[0] * second_element)

class PowerAnalysis:
    def __init__(self, alpha=0.05, power_thresh=0.8, stat_test=st.ttest_ind, bootsize=1000, 
                 random_state=None, **kwargs):
        self._bootsize = bootsize
        self._random_state=random_state
        self._alpha = alpha
        self._power = power_thresh
        self._stat_test = stat_test
        self._kwargs = kwargs
        
    def fit(self, *samples, lift=None, min_sample_size=None):
        if not samples or len(samples) > 2:
            raise ValueError('use one or two samples')
        else:
            self._samples = [np.array(i) for i in samples]
        if not lift:
            lift = 1
        if len(self._samples) == 1:
            self._control = self._samples[0]
            self._test = self._control * lift
        else:
            self._control, self._test = self._samples
        if not min_sample_size:
            min_sample_size = max(len(self._control), len(self._test))
                
        p_values = []
        means = []
        np.random.seed(self._random_state)
        for i in range(self._bootsize):
                sample_1 = np.random.choice(self._control, size=min_sample_size, replace=True)
                sample_2 = np.random.choice(self._test, size=min_sample_size, replace=True)
                stat_result = self._stat_test(sample_1, sample_2, **self._kwargs)
                p_values.append(stat_result[1] if isinstance(stat_result, (tuple,set,list)) else stat_result)
                means.append([np.mean(sample_1), np.mean(sample_2)])
        self._p_values = np.array(p_values)
        self._means = np.array(means)
        
    def compute_fpr(self, weighted=False):
        fpr = (self._p_values <= self._alpha).sum() / self._bootsize
        return fpr * max(self._p_values) if weighted else fpr
    
    def get_charts(self, figsize=(18,6), bins=10, alpha=0.7):
        plt.figure(figsize=figsize)
        plt.subplot(1,3,1)
        sns.histplot(self._control, bins=bins, label=f'sample_1')
        sns.histplot(self._test, bins=bins, label=f'sample_2', color='C1', alpha=alpha)
        plt.title('Distribution of Samples')
        plt.legend()
        plt.subplot(1,3,2)
        sns.histplot(self._means[:,0], bins=bins, label=f'sample_1')
        sns.histplot(self._means[:,1], bins=bins, label=f'sample_2', color='C1', alpha=alpha)
        plt.title('Bootstrap Means Distribution')
        plt.legend()
        plt.subplot(1,3,3)
        p_val_bins = 100 if self._alpha <= 0.01 else int(1/self._alpha)
        sns.histplot(self._p_values, color='C3', stat='probability', bins=p_val_bins)
        plt.axvline(self._alpha, color='black', linewidth=2, label=f'alpha={self._alpha}', ls='--')
        hist = np.histogram(self._p_values, bins=p_val_bins)
        if (hist[0] / hist[0].sum()).max() >= self._power: 
            plt.axhline(0.8, color='blue', linewidth=2, label=f'power_threshold={self._power}', ls='--')
        plt.legend()
        plt.title('P-values Distribution')
        plt.show()
        
    def perform_chisquare(self, bins=None):
        if not bins:
            len_ = len(np.arange(0, max(self._p_values), self._alpha))
            self.bins = 20 if len_ > 20 else len_
        else:
            self.bins = bins
        return st.chisquare(np.histogram(self._p_values, bins=self.bins)[0])
        

def cohens_d(control,test):
    
    """https://en.wikipedia.org/wiki/Effect_size"""
    
    n_control, n_test = len(control), len(test)
    diff = np.mean(test) - np.mean(control)
    df = n_control + n_test - 2
    sd_pooled = (((n_control-1) * np.std(control) ** 2 + (n_test-1) * np.std(test) ** 2) / df)**.5
    return diff / sd_pooled
