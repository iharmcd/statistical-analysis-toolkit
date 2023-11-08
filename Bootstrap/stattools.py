#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

class BootstrapAB:
    
    '''https://en.wikipedia.org/wiki/Bootstrapping_(statistics)'''

    def __init__(self, stat=np.mean, confidence_level=0.95, boot_size=10_000, random_state=None, **kwargs):
        self._boot_size = boot_size
        self._statistic = stat
        self._random_state = random_state
        self._confidence_level = confidence_level
        self.left_quant = (1 - confidence_level) / 2
        self.right_quant = 1 - (1 - confidence_level) / 2
        self._kwargs = kwargs
                
    def fit(self, sample_a, sample_b, progress_bar=False):
        sample_a, sample_b = np.array(sample_a), np.array(sample_b)
        np.random.seed(self._random_state)
        size = max(len(sample_a),len(sample_b))
        bootstrap_data = []
        rng = tqdm(range(self._boot_size)) if progress_bar else range(self._boot_size)
        for i in rng: 
            sub_a = np.random.choice(sample_a, size=size, replace = True)
            sub_b = np.random.choice(sample_b, size=size, replace = True)
            bootstrap_data.append([self._statistic(sub_a,**self._kwargs),self._statistic(sub_b,**self._kwargs)])
        
        self.lift = (self._statistic(sample_b,**self._kwargs) - self._statistic(sample_a,**self._kwargs)) / self._statistic(sample_a,**self._kwargs)
        self._bootstrap_data = np.array(bootstrap_data)
        self.diffs = self._bootstrap_data[:,1] - self._bootstrap_data[:,0]
        self.a_ci = self._get_ci(self._bootstrap_data[:,0])
        self.b_ci = self._get_ci(self._bootstrap_data[:,1])
        self.diff_ci = self._get_ci(self.diffs)
        self.uplift = self.diffs / self._bootstrap_data[:,0]
        self.uplift_ci = self._get_ci(self.uplift)
        
    def compute(self):
        p = np.mean(self.diffs > 0)
        pvalue = min(p*2, 2-p*2)
        stats = namedtuple('BootstrapResult', ('pvalue', 'uplift','uplift_ci','diff_ci'))
        return stats(pvalue, self.lift, self.uplift_ci, self.diff_ci)
    
    def _get_ci(self, data):
        self.quants = np.quantile(data, [self.left_quant, self.right_quant])
        return (self.quants[0].round(3),
                self.quants[1].round(3))
    
    def get_charts(self, figsize=(22,6), bins=50, stat='probability'):
    
        plt.figure(figsize=figsize)
        plt.subplot(1,3,1)
        sns.histplot(self._bootstrap_data[:,0], bins=bins,  stat=stat, color='#19D3F3',
                    label=f'A Sample. {self._confidence_level:.0%} CI: {self.a_ci[0]} - {self.a_ci[1]}')
        sns.histplot(self._bootstrap_data[:,1],bins=bins,  stat=stat, color='C1',
                    label=f'B Sample. {self._confidence_level:.0%} CI: {self.b_ci[0]} - {self.b_ci[1]}')
        plt.legend()
        plt.title(f'Distribution of {self._statistic.__name__}(s) for each group')
        plt.subplot(1,3,2)
        bar = sns.histplot(self.diffs, bins=bins, stat=stat, color='#DAA520',
                           #label=f'{self._confidence_level:.0%} CI: {self.diff_ci[0]} - {self.diff_ci[1]}'
                          )
        #plt.legend()
        plt.title(f'Distribution of {self._statistic.__name__}(s) differences (B-A)')
        plt.subplot(1,3,3)
        h = sns.histplot(x=self.uplift,bins=bins,stat='probability',cumulative=True, color='#636EFA')
        for i in h.patches:
            if i.get_x() <= 0:
                i.set_facecolor('#EF553B')
        plt.axvline(x=self.lift, color='black', linestyle='--')
        #plt.axhline(ratio, color='black',linestyle='--')
        plt.yticks(np.arange(0,1.1,0.1))
        plt.title('Uplift')
        plt.show()
        
        #bar = sns.histplot(self.diffs, bins=bins, stat=stat, color='#636EFA',
        #                   label=f'{self._confidence_level:.0%} CI: {self.diff_ci[0]} - {self.diff_ci[1]}')
        #counts = []
        #for i in bar.patches:
        #    counts.append(i.get_height())
        #    if i.get_x() <= self.diff_ci[0] or i.get_x() >= self.diff_ci[1]:
        #        i.set_facecolor('#EF553B')
        ##plt.vlines(self.quants,ymin=0,ymax=max(counts),linestyle='--') 
        #plt.legend()
        #plt.title(f'Distribution of {self._statistic.__name__}(s) differences (B-A)')
        #plt.show()
        
        

def efron_tibshirani(sample_a, sample_b, bootsize=10000, random_state=None):
    
    "https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bootstrap_hypothesis_testing"
    sample_a, sample_b = np.array(sample_a), np.array(sample_b)
    def t_stat(sample_a, sample_b):
        lift = sample_a.mean() - sample_b.mean()
        return lift / (np.var(sample_a)/len(sample_a) + np.var(sample_b)/len(sample_b)) ** .5
    true_t = t_stat(sample_a, sample_b)
    combined_mean = ((sample_a.mean() * len(sample_a) + sample_b.mean() * len(sample_b)) / 
                                                        (len(sample_b) + len(sample_a)))
    bias_sample_a = sample_a - sample_a.mean() + combined_mean
    bias_sample_b = sample_b - sample_b.mean() + combined_mean
    
    max_len = max([len(sample_a), len(sample_b)])
    
    np.random.seed(random_state)
    t_values = []
    for i in range(bootsize):
        sub_a=np.random.choice(bias_sample_a, size=max_len, replace=True)
        sub_b=np.random.choice(bias_sample_b, size=max_len, replace=True)
        t_values.append(t_stat(sub_a,sub_b))
    t_values = np.array(t_values)
    
    p_ = (t_values >= true_t).sum() / bootsize
    return min(2*p_, 2-2*p_)


def ratio_bootstrap(sample_a, sample_b, statistic=np.mean, bootsize=10000, random_state=None, **kwargs):
    ab_diff = statistic(sample_a, **kwargs) - statistic(sample_b, **kwargs)
    
    np.random.seed(random_state)
    count=0
    for _ in range(bootsize):
        stack = np.random.choice(np.hstack([sample_a,sample_b]), size=len(sample_a) + len(sample_b), replace=True)
        sub_a = stack[:len(sample_a)]
        sub_b = stack[len(sample_b):]
        boot_diff = statistic(sub_a, **kwargs) - statistic(sub_b, **kwargs)
        if boot_diff >= ab_diff:
            count += 1
    p_ = count / bootsize
    return min(2*p_, 2-2*p_)


def percentile_ci(data, conf_level=0.95, bootsize=10000, random_state=None, statistic=np.mean, **kwargs):
    np.random.seed(random_state)
    left_quant, right_quant = (1 - conf_level) / 2, 1 - (1-conf_level) / 2
    
    values = []
    for i in range(bootsize):
        subsample = np.random.choice(data, size=len(data), replace=True)
        values.append(statistic(subsample, **kwargs))
    stat = namedtuple('ConfidenceInterval', ('level', 'low','high'))
    return stat(conf_level, *np.quantile(values, [left_quant, right_quant]))

   
def ttest_ci(sample_a, sample_b, confidence_level=0.95):
    t_alpha = abs(st.norm(0,1).ppf((1-confidence_level)/2))
    mean_a,mean_b = np.mean(sample_a), np.mean(sample_b)
    var_a,var_b = np.var((sample_a), ddof=1),np.var((sample_b), ddof=1)
    se = ((var_a)/len(sample_a) + (var_b)/len(sample_b))**.5
    low, high = ((mean_a-mean_b) - t_alpha*se).round(4), ((mean_a-mean_b) + t_alpha*se).round(4)
    stat = namedtuple('ConfidenceInterval', ('level', 'low','high'))
    return stat(confidence_level, low, high)


def correlation_ratio(categories, values):
    
    '''https://en.wikipedia.org/wiki/Correlation_ratio
       ssw: sum of squares within groups
       ssb: sum of squares between groups'''
    
    values = np.array(values)
    categories = np.array(categories)
    
    ssw = 0
    ssb = 0
    for category in set(categories):
        subgroup = values[np.where(categories == category)[0]]
        ssw += sum((subgroup-np.mean(subgroup))**2)
        ssb += len(subgroup)*(np.mean(subgroup)-np.mean(values))**2

    return (ssb / (ssb + ssw))**.5


def cramers_v(rc_table, observations='raise'):
    
    '''Calculating cramers_v correlation for categorical data. 
       https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
       correction: True - Yates' correction'''
    
    rc_table = np.array(rc_table)
    
    def get_corr(rc_table):
        
        if rc_table.min() < 10:
            correction = True
        else:
            correction = False
            
        n = rc_table.sum()
        chi2_stats = st.chi2_contingency(rc_table, correction=correction)
        cramers_v = (chi2_stats[0]/(n*min(rc_table.shape[0]-1, rc_table.shape[1]-1)))**.5
        stat = namedtuple('CramersvResult', ('corr', 'pvalue','chi2'))
        return stat(cramers_v, chi2_stats[1], chi2_stats[0])    
    
    if observations == 'raise':
        if rc_table.min() < 5:
            raise ValueError('Not enough observations')
        else:
            return get_corr(rc_table)
    elif observations == 'ignore':
        return get_corr(rc_table)


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

class TestAnalyzer:
    def __init__(self, stat=st.ttest_ind, alpha=0.05, boot_size=10_000, random_state=None, **kwargs):
        self._boot_size = boot_size
        self._random_state=random_state
        self._alpha = alpha
        self._stat = stat
        self._kwargs = kwargs
        
    def fit(self, sample, samples_ratio=0.5, progress_bar=False):
        np.random.seed(self._random_state)
        p_values = []
        self._means = []
        size = len(sample)
        size_per_sample = int(samples_ratio * size)
        rng = tqdm(range(self._boot_size)) if progress_bar else range(self._boot_size)
        for i in rng:
                sample_data = np.random.choice(sample, size=size, replace=True)
                a,b = sample_data[:size_per_sample], sample_data[size_per_sample:] 
                stat_result = self._stat(a,b, **self._kwargs)
                p_values.append(stat_result.pvalue if isinstance(stat_result, tuple) else stat_result)
                self._means.append(np.mean(a)-np.mean(b))
        self._p_values = np.array(p_values)
        
    def compute_fpr(self, weighted=False):
        fpr = np.mean(self._p_values <= self._alpha)
        return fpr * max(self._p_values) if weighted else fpr

    def perform_chisquare(self, bins=None):
        if not bins:
            len_ = len(np.arange(0, max(self._p_values), self._alpha))
            self.bins = 20 if len_ > 20 else len_
        else:
            self.bins = bins
        return st.chisquare(np.histogram(self._p_values, bins=self.bins)[0])
    
    def get_charts(self, figsize=(20,6), bins=20):
        with sns.axes_style("whitegrid"): 
            plt.figure(figsize=figsize)
            plt.subplot(1,2,1)
            sns.histplot(self._means, bins=bins, color='#19D3F3', stat='density', 
                         label=f'AVG: {round(np.mean(self._means),3)}\nSTD: {round(np.std(self._means),3)}')
            plt.legend()
            plt.title('Bootstrap Mean(s) Distribution')
            plt.subplot(1,2,2)
            # p_val_bins = 100 if self._alpha <= 0.01 else int(1/self._alpha)
            # sns.histplot(self._p_values, color='C3', stat='probability', bins=p_val_bins,label='p-value')
            # plt.axvline(self._alpha, color='black', linewidth=2, label=f'alpha={self._alpha}', ls='--')
            # hist = np.histogram(self._p_values, bins=p_val_bins)
            # plt.legend()
            # plt.title('P-values Distribution')
            plt.plot([0, 1], [0, 1], linestyle='dashed', color='black', linewidth=2)  # Рисование пунктирной линии с заданной шириной
            plt.vlines(x=0.05,ymin=0,ymax=1,linestyle='dotted', color='black', linewidth=2)
            plt.plot(np.array(sorted(self._p_values)), np.array(sorted(np.linspace(0,1,self._boot_size))))
            plt.title('P-values Distribution Estimate')
            plt.ylabel('p-value')
            plt.show()

        

def cohens_d(control,test):
    
    """https://en.wikipedia.org/wiki/Effect_size"""
    
    n_control, n_test = len(control), len(test)
    lift = np.mean(test) - np.mean(control)
    df = n_control + n_test - 2
    sd_pooled = (((n_control-1) * np.std(control) ** 2 + (n_test-1) * np.std(test) ** 2) / df)**.5
    return lift / sd_pooled

def lift(before, after):
    return (after - before) / before


class BayesAB:
    '''
    https://marketing.dynamicyield.com/bayesian-calculator/
    
    required libs:
    import numpy as np
    from collections import namedtuple
    import matplotlib.pyplot as plt
    import seaborn as sns
    '''
    
    def __init__(self, size=100_000, conf_level=0.95, random_state=None):
        self.size = size
        self.random_state = random_state
        self.left_quant, self.right_quant = (1 - conf_level) / 2, 1 - (1-conf_level) / 2

    def fit(self, nobs, counts, prior=()):
    
        if len(nobs) != 2 or len(counts) !=2:
            raise ValueError('You must have 2 elements in each list')
            
        control_a, test_a = nobs
        control_total, test_total = counts
        control_b, test_b = control_total - control_a, test_total - test_a

        np.random.seed(self.random_state)
        
        self.cr_control = control_a / control_total
        self.cr_test = test_a / test_total
        self.lift = self.uplift(self.cr_control,self.cr_test)
        
        if not isinstance(prior, (list,tuple)):
            raise TypeError(f'You can use for prior only list or tuple. Passed {type(prior).__name__}')
        elif not prior:
            pr = (1,) * 4
        elif len(prior) == 2:
            pr = prior * 2
        elif len(prior) in [1,3] or len(prior) > 4:
            raise ValueError('You can pass only two or four values')
        else:
            pr = prior
  
        self.beta_control = np.random.beta(a=control_a+pr[0],b=control_b+pr[1],size=self.size)
        self.beta_test = np.random.beta(a=test_a+pr[2],b=test_b+pr[3],size=self.size)
        
    def uplift(self, before, after):
        return (after - before) / before
        
    def compute(self):
        proba = np.mean(self.beta_test > self.beta_control)
        #loss_control =  np.mean(np.maximum(self.beta_test - self.beta_control, 0))
        #loss_test =  np.mean(np.maximum(self.beta_control - self.beta_test, 0))
        loss_control = np.mean(np.maximum(self.uplift(self.beta_control, self.beta_test),0)) #uplift_loss_c
        loss_test = np.mean(np.maximum(self.uplift(self.beta_test, self.beta_control),0)) #uplift_loss_t
        ci = np.quantile(self.uplift(self.beta_control, self.beta_test), q=[self.left_quant, self.right_quant]).tolist()
        stats = namedtuple('BayesResult', ('proba', 'uplift','uplift_ci','control_loss','test_loss'))
        return stats(proba, self.lift, ci, loss_control, loss_test)
    
    def get_charts(self, figsize=(22,6), bins=50):
        thresh = 0
        diff = self.uplift(self.beta_control, self.beta_test)
        #diff = self.beta_test / self.beta_control
        min_xy, max_xy = np.min([self.beta_control, self.beta_test]), np.max([self.beta_control, self.beta_test])
        #ratio = (diff <= thresh).sum() / self.size
        
        plt.figure(figsize=figsize)
        plt.subplot(1,3,1)
        sns.histplot(self.beta_control, label='control', bins=bins, stat='probability', color='#19D3F3')
        sns.histplot(self.beta_test, label='test',bins=bins, stat='probability', color='C1')
        plt.title('Beta Distributions for CR')
        plt.legend()
        
        plt.subplot(1,3,2)
        sns.histplot(x=self.beta_control,y=self.beta_test,bins=bins, color='#3366CC')
        plt.xlabel('control')
        plt.ylabel('test')
        plt.axline(xy1=[min_xy, min_xy], xy2=[max_xy,max_xy], color='black', linestyle='--')
        plt.title('Joint Distribution')
        
        plt.subplot(1,3,3)
        h = sns.histplot(x=diff,bins=bins,stat='probability',cumulative=True, color='#636EFA')
        for i in h.patches:
            if i.get_x() <= thresh:
                i.set_facecolor('#EF553B')
        plt.axvline(x=self.lift, color='black', linestyle='--')
        #plt.axhline(ratio, color='black',linestyle='--')
        plt.yticks(np.arange(0,1.1,0.1))
        plt.title('Uplift')
        plt.show()

        
        
def bayes_duration_estimator(cr_baseline, 
                             uplift, 
                             total_daily_size, 
                             power_level=0.8, 
                             boot_size=10_000, 
                             beta_size=10_000,
                             significance_level=0.95,
                             random_state=None):
    '''
    one-sided duration_estimator
    https://marketing.dynamicyield.com/ab-test-duration-calculator/
    
    required libs:
    import numpy as np
    from collections import namedtuple
    from tqdm import tqdm
    '''
    
    np.random.seed(random_state)
    
    p_control = cr_baseline
    p_test = cr_baseline*(1+uplift)

    power = 0
    days = 0
    sample_size = 0
    while power < power_level:
        sample_size += total_daily_size
        size_per_sample = int(sample_size/2)
        probas = []
        for i in tqdm(range(boot_size)):
            a_control = np.random.binomial(n=size_per_sample, p=p_control)
            a_test = np.random.binomial(n=size_per_sample, p=p_test)
            c = np.random.beta(a_control,size_per_sample-a_control, size=beta_size)
            t = np.random.beta(a_test,size_per_sample-a_test, size=beta_size)
            probas.append((t>c).mean())
            
        days += 1
        power = (np.array(probas) >= significance_level).mean()

    result = namedtuple('EstimatorResult',('days', 'total_sample_size', 'power'))    
    return result(days, sample_size, power)


def bayesian_continuous(mean: np.array, 
                     std: np.array, 
                     n: np.array, 
                     cofidence_level=0.95, 
                     size=100_000, 
                     random_state=None):
    
    if len(mean) != 2 or len(std) != 2 or len(n) != 2:
        raise ValueError('Len of all collections must be equal to 2')
    else:
        mean,std,n = np.array(mean),np.array(std),np.array(n)
        sem = std / n**.5
        lift = (mean[1] - mean[0]) / mean[0]
    q_l, q_h = (1 - cofidence_level) / 2, 1 - (1 - cofidence_level) / 2
        
    np.random.seed(random_state)
    a = np.random.normal(mean[0],sem[0], size=10_000)
    b = np.random.normal(mean[1],sem[1], size=10_000)
    diff_ci = np.quantile(b-a,q=[q_l,q_h])
    proba = np.mean(b > a)
    pvalue = min(2*proba,2-2*proba)
    uplift_ci = np.quantile((b-a)/a,q=[q_l,q_h])
    
    stats = namedtuple('BayesResult', ('pvalue', 'uplift','uplift_ci','diff_ci'))
    return stats(pvalue, lift, uplift_ci, diff_ci)
