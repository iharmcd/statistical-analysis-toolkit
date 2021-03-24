import pandas as pd
from sklearn.model_selection import train_test_split

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
split_size: {self.split_size}'
    
    def fit(self,dataset):
        self.input_data = dataset
        group_ratio = round(len(self.input_data)/ self.separation * self.extraction / len(self.input_data), 4) * 100
        return f'output size for test and control groups will be {group_ratio}% for each group'
    
    @staticmethod
    def get_rank(data,column):
        data['rank'] = data[column].sort_values(ascending=False).rank(ascending=False, method='first')
        return data
    
    def get_split(self, column):
        self.data = self.get_rank(self.input_data, column)
        self.column = column
        ranked_dataset = pd.concat([self.get_rank(self.data[self.data['rank'] % self.separation == i], self.column) 
                                    for i in range(0,self.separation)])
        filter_ = ranked_dataset['rank'].value_counts().to_frame().query('rank < @self.separation').index
        ranked_dataset = ranked_dataset[~ranked_dataset['rank'].isin(filter_)]
        biggest_part, test = train_test_split(ranked_dataset, random_state=self.random_state, 
                                              test_size=self.split_size, shuffle=True, stratify=ranked_dataset['rank'])
        expected_ratio = self.split_size/(biggest_part.shape[0]/(biggest_part.shape[0] + test.shape[0]))
        _, control = train_test_split(biggest_part, random_state=self.random_state, test_size=expected_ratio, 
                                      shuffle=True, stratify=biggest_part['rank'])
        return test, control