import os
import tensorflow as tf
import subprocess
import joblib


import tensorflow as tf


    

class TrainModel():
    '''
    Build Lstm model for tensorflow
    ----------

    Returns
    -------
    self.model:
        Deep learning based Model
    
    '''
    
    def __init__(self, MODEL, TOKENIZER, ENC,TRAIN_DATA, TRAIN_LABELS,TEST_DATA, TEST_LABELS, BATCH_SIZE=64,EPOCHS=10):
        self.model_checkpoint_callback = []
        self.enc = ENC
        self.tokenizer = TOKENIZER
        self.model = MODEL
        self.train_data = TRAIN_DATA
        self.train_labels = TRAIN_LABELS
        self.test_data  = TEST_DATA
        self.test_labels = TEST_LABELS
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.history = []


    def DefineCheckPoint(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        #Bidirectional LSTM
        checkpoint_filepath = './models/model.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_acc',
            mode='max',
            save_best_only=True)
        
        
    
    def SavePKL(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        joblib.dump(self.enc, 'labelencoder.pkl')  
        joblib.dump(self.tokenizer, 'tokenizer.pkl')  

        
    
        
    def ModelTraining(self):
        '''
        Define the model
        ----------
        
        Returns
        -------
        
        '''
        # self.SavePKL()
        self.DefineCheckPoint()
        
        self.history = self.model.fit(self.train_data, self.train_labels,
                 batch_size=self.batch_size,
                 epochs=self.epochs,
                 validation_data=(self.test_data, self.test_labels),callbacks=[self.model_checkpoint_callback])
        return self.model,self.history
        # return self.final_set,self.labels, self.enc, self.ohe,self.encoding_flag
    