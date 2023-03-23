

def bootstrap_ab(a,b, stat=np.mean, boot_size=10_000, random_state=None, alternative='two-sided', progress_bar=False, **kwargs) -> float:
    if alternative not in ['two-sided','one-sided']:
        raise ValueError(f"Received value: '{alternative}'. Сhoose 'two-sided' or 'one-sided'")
    np.random.seed(random_state)
    data = []
    size = max(len(a),len(b))
    rng = tqdm(range(boot_size)) if progress_bar else range(boot_size)
    for i in rng:
        a_sample = np.random.choice(a, size=size, replace=True)
        b_sample = np.random.choice(b, size=size, replace=True)
        data.append(stat(a_sample,**kwargs) > stat(b_sample,**kwargs))
    if alternative == 'two-sided':
        return min(np.mean(data)*2, 2-np.mean(data)*2)
    else:
        return np.mean(data)
      
      
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
  