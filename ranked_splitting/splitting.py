import pandas as pd
from sklearn.model_selection import train_test_split

class RankedSplit:
    
    def __init__(self, size_per_sample, strat_size=10, samples_cnt=2, random_state=None):
        self.strat_size = strat_size
        self.random_state = random_state

        if samples_cnt not in [1,2]:
            raise ValueError(f"Invalid property value. Recieved {samples_cnt=}. Use 1 or 2.")
        else:
            self.samples_cnt = samples_cnt
            
        if not isinstance(strat_size, int):
            raise TypeError('strat_size must be int')
        
        if size_per_sample * samples_cnt > 1 or size_per_sample <= 0:
            raise ValueError('The total ratio of samples must be greater than 0 and must not exceed 1')
        else:
            self.test_size = size_per_sample
            #self.control_size = size_per_sample / (1 - size_per_sample)
        if size_per_sample * strat_size < 1:
            raise ValueError(f'With {size_per_sample=} your strat_size must be >= {int(1/size_per_sample)}')
        

            

    def get_rank(self, collection):
        rank = pd.Series(collection).rank(ascending=False, method='first')
        output = pd.DataFrame({'data':collection,'rank':rank})
        return output
    
    def fit(self, data):
        self.dataset = self.get_rank(data)
        self.ranked_dataset = (pd.concat([self.get_rank(self.dataset[self.dataset['rank'] % self.strat_size == i]['data']) 
                                          for i in range(0,self.strat_size)])
                               .sort_values(by=['rank','data'], ascending=(True,False)))
        filter_ = self.ranked_dataset['rank'].value_counts().to_frame().query('rank < @self.strat_size').index
        self.ranked_dataset = self.ranked_dataset[~self.ranked_dataset['rank'].isin(filter_)]
    
    def get_split(self):
        if self.samples_cnt == 1:
            train_size = None
        else:
            train_size = self.test_size
        second, first = train_test_split(self.ranked_dataset, 
                                         train_size=train_size,
                                         test_size=self.test_size,
                                         shuffle=True, 
                                         stratify=self.ranked_dataset['rank'],
                                         random_state=self.random_state)
       
        return first, second
