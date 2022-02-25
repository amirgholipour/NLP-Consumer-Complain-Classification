import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
class ReadData():
    '''
    Turn raw data into features for modeling
    ----------

    Returns
    -------
    self.final_set:
        Features for modeling purpose
    self.labels:
        Output labels of the features
    enc: 
        Ordinal Encoder definition file
    ohe:
        One hot  Encoder definition file
    '''
    def __init__(self, DATA_PATH="./dataset/csv/data.csv", SPLIT_RATE=.2, INPUT_FEATURE_NAME ='consumer_complaint_narrative' , OUTPUT_FEATURE_NAME='product'):
#     def __init__(self, *args, **kwargs):
        self.data_path =  DATA_PATH 
        self.client = []
        self.split_rate = SPLIT_RATE
        
        self.in_fe_name = 'consumer_complaint_narrative'
        self.out_fe_name = 'product'
        self.df = []
        self.train_data = []
        self.test_data = []

        self.train_labels = []
        self.test_labels = []
        self.enc = []
        
    
        
#         self.final_set,self.labels = self.build_data()
    def ReadInput(self):
        '''
        Read Data from csv file
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        
        self.df = pd.read_csv(self.data_path)

        
#         return self.df
    ## read the data from the source file
    def SplitData(self):
        '''
        Reading the csv file
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.df[self.in_fe_name], self.df[self.out_fe_name],stratify=self.df[self.out_fe_name], 
                                                    test_size=self.split_rate)
        
#         return self.train_data, self.test_data, self.train_labels, self.test_labels
    def LabelEncoding(self):
        '''
        GetRequired Info
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        ##label encoding target variable
        self.enc = preprocessing.LabelEncoder()
        self.train_labels = self.enc.fit_transform(self.train_labels)
        self.test_labels = self.enc.fit_transform(self.test_labels)

        print(self.enc.classes_)
        print(np.unique(self.train_labels, return_counts=True))
        print(np.unique(self.test_labels, return_counts=True))



    ## address the missing information
    def ReadDataFrameData(self):
        '''
        Replace the missing value with the zero.
        ----------
        
        Returns
        -------
        Dataframe with replaced missing value.
        '''
        self.ReadInput()
        self.SplitData()
        self.LabelEncoding()
        
        
        
        return self.train_data, self.test_data, self.train_labels, self.test_labels,self.enc