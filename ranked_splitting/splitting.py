class RankedSplit:
    
    def __init__(self, size_per_sample, strat_size=10, random_state=None):
        self.strat_size = strat_size
        self.random_state = random_state
        
        if not isinstance(strat_size, int):
            raise TypeError('strat_size must be int')
        
        if size_per_sample * strat_size < 1:
            raise ValueError(f'With {size_per_sample=} your strat_size must be >= {int(1/size_per_sample)}')
        
        if size_per_sample > 0 and size_per_sample <= 0.5:
            self.test_size = size_per_sample
            self.control_size = size_per_sample / (1 - size_per_sample)
        else:
            raise ValueError('Size per sample have to > 0 and <= 0.5') 
        

               
    def get_rank(self, collection):
        rank = pd.Series(collection).rank(ascending=False, method='first')
        output = pd.DataFrame({'data':collection,'rank':rank})
        return output
    
    def fit(self, data):
        self.dataset = self.get_rank(data)
        self.ranked_dataset = (pd.concat([self.get_rank(self.dataset[self.dataset['rank'] % self.strat_size == i]['data']) 
                                          for i in range(0,self.strat_size)])
                               .sort_values(by=['rank','data'], ascending=(True,False)))
        
    def get_split(self):
        
        _, first = train_test_split(self.ranked_dataset, 
                                         test_size=self.test_size,
                                         shuffle=True, 
                                         stratify=self.ranked_dataset['rank'],
                                         random_state=self.random_state)
        if self.test_size == 0.5:
            return first, _
        else:
            _, second = train_test_split(_, 
                                         test_size=self.control_size, 
                                         shuffle=True, 
                                         stratify=_['rank'],
                                         random_state=self.random_state)
            return first, second
