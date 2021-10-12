import pandas as pd
from sklearn.model_selection import train_test_split

def calc_subgroup_size(extraction_size, expected_ratio):
    return int(round(extraction_size/expected_ratio))

def calc_extraction_size(subgroup_size, expected_ratio):
    return int(round(subgroup_size*expected_ratio))

class RankedSplit:
    
    def __init__(self, group_size, extraction_size, random_state=None):
        self.separation = group_size
        self.random_state = random_state
        self.extraction = extraction_size
        self.split_size = extraction_size/group_size
       
    def __repr__(self):
        return f'{type(self)}\n\
separation_group_size: {self.separation}\n\
extraction_size: {self.extraction}\n\
random_state_parameter: {self.random_state}\n\
output_split_size: {self.split_size}'
        
    @staticmethod
    def get_rank(collection):
        #input any ordered collection
        rank = pd.Series(collection).rank(ascending=False, method='first')
        output = pd.DataFrame({'data':collection,'rank':rank})
        return output
    
    def fit(self, data):
        self.dataset = self.get_rank(data)
        
    def get_split(self):
        self.ranked_dataset = pd.concat([self.get_rank(self.dataset[self.dataset['rank'] % self.separation == i]['data']) for i in range(0,self.separation)])
        filter_ = self.ranked_dataset['rank'].value_counts().to_frame().query('rank < @self.separation').index
        self.ranked_dataset = self.ranked_dataset[~self.ranked_dataset['rank'].isin(filter_)]
        biggest_part, test = train_test_split(self.ranked_dataset, random_state=self.random_state, 
                                          test_size=self.split_size, shuffle=True, stratify=self.ranked_dataset['rank'])
        expected_ratio = self.split_size / (1 - self.split_size)
        _, control = train_test_split(biggest_part, random_state=self.random_state, test_size=expected_ratio, 
                                      shuffle=True, stratify=biggest_part['rank'])
        return test, control

