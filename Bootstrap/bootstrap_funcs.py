def bootstrap_ab(a,b, stat=np.mean, confidence_level=0.95, boot_size=10_000, two_tailed=True, axis_0=True, random_state=None, **kwargs) -> tuple:
    ''''''
    np.random.seed(random_state)
    size = max(len(a),len(b))
    
    a_sample, b_sample = np.random.choice(a, size=(size,boot_size), replace=True), np.random.choice(b, size=(size,boot_size), replace=True)
    
    if axis_0:
        statistic_a, statistic_b = stat(a_sample, axis=0, **kwargs), stat(b_sample, axis=0, **kwargs)
    else:
        statistic_a, statistic_b = stat(a_sample, **kwargs), stat(b_sample, **kwargs)
        
    p = np.mean((statistic_b - statistic_a) > 0)
    diff_ci = np.quantile((statistic_b - statistic_a), q=[(1-confidence_level)/2,1-(1-confidence_level)/2])
    #uplift_ci = np.quantile((statistic_b - statistic_a) / statistic_a, q=[(1-confidence_level)/2,1-(1-confidence_level)/2])
    return min(p*2, 2-p*2) if two_tailed else p, tuple(diff_ci), #tuple(uplift_ci)
      
      
def monte_carlo_area(x, y, num_samples=1_000_000, random_state=None):
    np.random.seed(random_state)
    min_x,max_x = min(x), max(x)
    min_y,max_y = min(y),max(y)
    # Сгенерировать случайные точки внутри этой области
    rand_x = np.random.uniform(min_x, max_x, size=num_samples)
    rand_y = np.random.uniform(min_y, max_y, size=num_samples)
    # Проверить, попадает ли точка внутрь кривой.
    # Это можно сделать, например, с помощью ломаных.
    # Для этого нужно соединить точки (x, y) ломаной и проверить,
    # пересекает ли ломаная точку (rand_x, rand_y).
    # Если точка пересекается с ломаной нечетное число раз,
    # значит она находится внутри кривой.
    num_under_curve = 0
    for i in range(len(x) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        condition = (rand_x >= min(x1, x2)) == (rand_x <= max(x1, x2)) 
        r_x = rand_x[condition]
        r_y = rand_y[condition]
        # Проверить, пересекает ли точка ломаную
        # (x1, y1) - (x2, y2) на заданном интервале
        y_intersect = y1 + (r_x - x1) * (y2 - y1) / (x2 - x1)
        num_under_curve += (y_intersect >= r_y).sum()
    # Вычислить площадь под кривой как долю точек,
    # которые попали под кривую, умноженную на площадь исходной области.
    ratio = (num_under_curve / num_samples)
    area =  ratio * ((max_x - min_x) * (max_y - min_y))
    return ratio, area
  
  
def fwer(n_comparison:int, alpha=0.05):
    '''multiple comaprison error'''
    return 1-(1-alpha)**n_comparison
  
  
  
def bootstrap_rel(sample_diff, boot_size=10_000, conf_level=0.05, random_state=None):
    np.random.seed(random_state)
    means = []
    for i in range(boot_size):
        s = np.random.choice(sample_diff, size=len(sample_diff), replace=True)
        means.append(s.mean())
    means = np.array(means)
    p = (0 > means).mean()
    p_value =  min(2*p , 2-2*p)               
    return p_value, np.quantile(means, q=[conf_level/2,1-conf_level/2])
  
  
  
def ci_binom_wilson(n, p, conf=0.95):
    '''$$\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}$$'''
    alpha_2 = (1-conf)/2
    z = st.norm.ppf(1-alpha_2)
    a = p + (z**2 / (2 * n)) 
    b = z * (p*(1-p)/n + z**2/(4*n**2))**.5
    c = (1+z**2/n)
    lower, upper = (a-b)/c, (a+b)/c
    return lower,upper
  
    
def bootstrap_duration_estimator(target_sample, 
                                 total_daily_size, 
                                 uplift, 
                                 stat=st.ttest_ind, 
                                 boot_size=10_000, 
                                 power_level=0.8, 
                                 significance_level=0.05, 
                                 random_state=None,
                                 progress_bar=True,
                                 **kwargs):
    
    np.random.seed(random_state)
    uplift += 1
    current_day_size = 0
    power = 0
    
    while power < power_level:
        current_day_size += total_daily_size
        p_values = []
        rng = tqdm(range(boot_size)) if progress_bar else range(boot_size)
        
        for i in rng:
            sample_data = np.random.choice(target_sample, size=current_day_size, replace=True)
            sample_size = int(current_day_size/2)
            a,b = sample_data[:sample_size], sample_data[sample_size:] * uplift
            stat_result = stat(a,b, **kwargs)
            p_values.append(stat_result.pvalue if isinstance(stat_result, tuple) else stat_result)
            power = (np.array(p_values) <= significance_level).mean()
            
    result = namedtuple('EstimatorResult',('days', 'total_sample_size', 'power'))    
    return result(int(current_day_size/total_daily_size), current_day_size, power)


def bootstrap_conversion_duration_estimator(cr_baseline, 
                                            uplift, 
                                            total_daily_size, 
                                            power_level=0.8, 
                                            boot_size=10_000, 
                                            significance_level=0.05,
                                            random_state=None, 
                                            progress_bar=True,
                                            **kwargs):
    '''
    https://www.optimizely.com/sample-size-calculator
    
    required libs:
    import numpy as np
    from collections import namedtuple
    from tqdm import tqdm
    '''
    from statsmodels.stats.proportion import proportions_ztest
    
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
        rng = tqdm(range(boot_size)) if progress_bar else range(boot_size)
        for i in rng:
            a_control = np.random.binomial(n=size_per_sample, p=p_control)
            a_test = np.random.binomial(n=size_per_sample, p=p_test)
            probas.append(proportions_ztest([a_control, a_test],[size_per_sample]*2,**kwargs)[1])
            
        days += 1
        power = (np.array(probas) <= significance_level).mean()

    result = namedtuple('EstimatorResult',('days', 'total_sample_size', 'power'))    
    return result(days, sample_size, power)


class CohensD:
    
    def __init__(self, calc='from_samples'):
        if calc not in ['from_samples', 'from_stats']:
            raise ValueError(f"{calc=}. Must equal to 'from_samples' or 'from_stats'")
        else:
            self.calc = calc
    
    def __call__(self, *args):
        if self.calc == 'from_samples':
            sample1, sample2 = args
            lift = abs(np.mean(sample1) - np.mean(sample2))
            sd_pooled = ((np.var(sample1,ddof=1) + np.var(sample2,ddof=1)) / 2)**.5
            return lift / sd_pooled
        else:
            mu1, mu2, std1, std2 = args
            return abs(mu1-mu2) / ((std1**2 + std2**2) / 2)**.5

def permutation_test(sample1, sample2, num_permutations=10000, random_state=None):
    observed_diff = np.mean(sample1) - np.mean(sample2)
    combined = np.concatenate((sample1, sample2))
    count = 0
    
    np.random.seed(random_state)
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_sample1 = combined[:len(sample1)]
        perm_sample2 = combined[len(sample1):]
        perm_diff = np.mean(perm_sample1) - np.mean(perm_sample2)
        
        if perm_diff >= observed_diff:
            count += 1
    
    p = count / num_permutations
    p_value = min(2*p, 2-2*p)
    return p_value


def calculate_number_of_trials(prob_success: float, desired_accuracy: float) -> int:
    """
    Calculates the number of independent trials required to succeed at least once with a probability of desired_accuracy.

    Parameters:
    prob_success (float): Probability of success for each independent trial.
    desired_accuracy (float): Desired probability of success.

    Returns:
    int: Number of independent trials required.

    Raises:
    ValueError: If prob_success or desired_accuracy is not in the range (0, 1).
    """
    def log(base, number):
        return np.log(number) / np.log(base)
    
    if prob_success <= 0 or prob_success >= 1:
        raise ValueError("prob_success should be in the range (0, 1).")
    if desired_accuracy <= 0 or desired_accuracy >= 1:
        raise ValueError("desired_accuracy should be in the range (0, 1).")

    num_trials = log(1 - prob_success, 1 - desired_accuracy)
    return int(np.ceil(num_trials))


def g_squared(contingency_table):
    """
    Function: g_squared
    
    Description:
    This function calculates the G-squared statistic (also known as the likelihood-ratio test statistic) 
    and its corresponding p-value for a given contingency table. The G-squared statistic is used to test 
    the independence of two categorical variables, as an alternative to the chi-squared test. The function 
    takes a contingency table as input and returns the G-squared statistic and the p-value.
    
    Parameters:
    contingency_table (array-like): A two-dimensional contingency table, where the rows represent one categorical 
    variable, and the columns represent another categorical variable. The values in the cells represent the frequency 
    of observations for each combination of category levels.

    Returns:
    tuple: A tuple containing the G-squared statistic (float) and the p-value (float). The p-value is calculated 
    using the chi-squared distribution with the degrees of freedom determined by the dimensions of the input 
    contingency table.
    
    """
    ct = np.asarray(contingency_table)
    _, _, dof,exp_freq = st.chi2_contingency(ct)
    g_squared = 2 * np.sum(ct * np.log(ct/exp_freq))
    return g_squared, st.chi2.sf(g_squared,dof)


def t_confidence_interval(a,b, confidence_level=0.95) -> tuple:
    m_a, m_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    n_a, n_b = len(a), len(b)
    t = st.t.ppf(1-(1-confidence_level) / 2, df = n_a + n_b - 2)
    diff = m_b - m_a
    lower = diff - t * (std_a**2 / n_a + std_b**2 / n_b)**.5
    upper = diff + t * (std_a**2 / n_a + std_b**2 / n_b)**.5
    return (lower,upper), (lower / m_a, upper / m_a)

def proportions_uplift_ci(p1,p2,n1,n2, confidence_level=0.95):
    Z = st.norm.ppf(1-(1-confidence_level)/2)
    lower = (p2 - p1) - Z * ((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))**.5
    upper = (p2 - p1) + Z * ((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))**.5
    return lower/p1, upper/p1



def monte_carlo_optimization(func, bounds, minimize=True, num_samples=1_000_000):
    best_params = None
    best_loss = None

    lh = defaultdict(tuple)
    for i in bounds:
        lh['low'] += (i[0],)
        lh['high'] += (i[1],)
    
    params = np.random.uniform(lh['low'], lh['high'], size=(num_samples,len(bounds)))
    
    for i in tqdm(params):
        loss = func(i)
        
        if minimize:
            if best_loss is None or loss < best_loss:
                best_params = i
                best_loss = loss
        else:
            if best_loss is None or loss > best_loss:
                best_params = i
                best_loss = loss
                
    return best_params, best_loss
