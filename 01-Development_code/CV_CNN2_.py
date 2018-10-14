#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:03:28 2018

@author: flaviagv

In this script, the training, validation, and test accuracy of a CNN with different inputs has been made. This was in order to compare 
what input performs better with this CNN. This CNN is composed by the following layers:
2D CONV(128 filters, 3x3) + MAX_POOLING(2x2) + 2D CONV(256 filters, 3x3) + MAX_POOLING + FLATTEN + DENSE(64) + DROPOUT(0.5) + DENSE(num_classes = 50)

"""

from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import tensorflow as tf
import keras
from six.moves import cPickle as pickle

from CV_CNN1_ import plot_model_history 
import matplotlib.pyplot as plt


def createConvNN(input_shape, num_classes):
    model = Sequential()
    
    # conv: 128 filtros
    model.add(Conv2D(filters     = 128, 
			         kernel_size = 3,
			         activation  = "relu",
			         use_bias    = True,
			         strides     = 1,
	                 padding      = "same",
			         input_shape = input_shape))
			     
	
    
	# max pool 2x2
    model.add(MaxPooling2D(pool_size = 2, strides   = None, padding   = "same"))


	# conv
	# 256 filtros
    model.add(Conv2D(filters     = 256, 
			         kernel_size = 3,
			         activation  = "relu",
			         use_bias    = True,
    			     strides     = 1,
                      padding     = "same"))
			         # kernel_initializer='glorot_uniform', 
			         # bias_initializer='zeros'
                     # padding='valid'

	# max pool 2x2
    model.add(MaxPooling2D(pool_size = 2,
				           strides   = None,
				           padding   = "same"))

	# flatten -> poner todo en un vector (sera de 256x32x32)
    model.add(Flatten())

	# fully connected 64
    model.add(Dense(64, activation='relu'))


    model.add(Dropout(0.5))   # we randomnly disable 20% of the neurons


	# softmax de 50 outputs 
    model.add(Dense(num_classes, activation='softmax'))

	# Adam Optimizer, Batch size  = 64, n_epochs -> early stopping
    model.compile(optimizer = Adam(), #lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0) 
			      loss      = categorical_crossentropy, #"mse", 
			      metrics   = ["accuracy"])

    return model




def cross_validation(X, y, tituloIm):
    """
    Hacemos tanto las 4 rounds como el train y la evaluation
    El 75% del train será usado para el tren de CV y el 15% para la validacion
    """
    # X sigue siendo una lista 
    
    # iteracion es un numero que tiene que estar en range(5), sino lanzar una excepcion 
    
    
    # quedarnos con 4/5(80%) para el train y 1/5 para el test(20%) 
    # En cada iteracion coger un trozo diferente del test, y que el resto sean train
    acc_train_list = []
    acc_val_list = []
    num_classes = 20
    num_elems = len(y)//20 # Coger 3
    sumAcc = 0 
     
    # Si hay restantes, se lo queda el train
        
    # primer bloque  -> [0:num_elems]
    # segundo bloque -> [num_elems*1:num_elems*2]
    # tercer bloque  -> [num_elems*2:num_elems*3]
    # cuarto bloque  -> [num_elems*3:num_elems*4]
    # quinto bloque  -> [num_elems*4:num_elems*5]

    # el numero del bloque del test te lo va a dar la iteracion 
    np.random.seed(133)
    np.random.shuffle(X)
    np.random.shuffle(y)
    fin = 0
        #np.random.permutation(len(y)).shape
    for iteracion in range(6):
        
        X_val = X[fin: 3*num_elems*(iteracion+1)]  
        y_val = y[fin : 3*num_elems*(iteracion+1)]
            
            
        X_train = X[0:fin] + X[3*num_elems*(iteracion+1):len(X)]
        y_train = y[0:fin] + y[3*num_elems*(iteracion+1):len(y)]
        print("----- iteracion %d -----"%iteracion)
        print(" X_val --> %f"%(len(X_val)*100/len(X)))
        print(" y_val --> %f"%(len(y_val)*100/len(X)))
        print(" X_train --> %f"%(len(X_train)*100/len(X)))
        print(" y_train --> %f\n"%(len(y_train)*100/len(X)))
        fin = 3*num_elems*(iteracion+1)
        
        X_train = np.asarray(X_train)        
        y_train = np.asarray(y_train)
        X_val   = np.asarray(X_val)
        y_val   = np.asarray(y_val)
            
    
        ### HACEMOS EL ENTRENAMIENTO DE LA RED 
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_val  = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        # y_val  = keras.utils.to_categorical(y_val, num_classes)
           
        input_shape = X_train.shape[1:]
            
            
        model     = createConvNN(input_shape, num_classes)
        
        	# define early stopping callback
    #    earlystop = EarlyStopping(monitor   = 'val_acc', 
    #    		                      min_delta = 0.0001, 
    #    		                      patience  = 5, 
    #                              	  verbose   = 1, 
    #                              	  mode      = 'auto')
        	
        # callbacks_list = [earlystop]
        
        keras.backend.get_session().run(tf.global_variables_initializer())
        
        model_info = model.fit(X_train, 
        		      			   y_train, 
        		      			   verbose    = 0,
        		      			   batch_size = 16, #64,
        		      			   epochs     = 100)#100, #100
        		      			   #callbacks  = callbacks_list)  # No estoy poniendo validation split
        
        
        
        plot_model_history(model_info, iteracion, tituloIm)   # Te indica en que epoch se ha parado debido a early stopping y te enseÃ±an dos grÃ¡ficas:
        	                                 # - Model accuracy / epoch 
        	                                 # - Model loss / epoch 
        
        
            #score = model.evaluate(X_test, y_test, verbose = 0)
        y_pred_probs = model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis = 1)
            #y_pred = model.predict_classes(X_test)
            
        print("Getting validation ACC ...")
        acc_val = accuracy_score(y_val, y_pred)
        print("Getting train ACC ...")
        acc_train = model.evaluate(X_train, y_train, verbose = 0)[1]
        print("Validation ACC iteration " + str(iteracion) +": " + str(acc_val))
        print("Train ACC: iteration " + str(iteracion) + ": " +  str(acc_train))
        acc_val_list.append(acc_val)
        acc_train_list.append(acc_train)
            
        sumAcc += acc_val
        
        #print('Test loss:', score[0])
    # sumAcc += roc_auc_score(np.argmax(y_val, axis = 1), y_pred, average = "weighted")
        # roc_auc_score(y_val, y_pred)
        # sumAcc += accuracy_score(np.argmax(y_val, axis = 1), y_pred)    #score[1])
         
        # print("CONFUSION MATRIX: \n", confusion_matrix(np.argmax(y_test, axis = 1), y_pred))
    
    k_fold = range(6)
    plt.plot(k_fold, acc_train_list)
    plt.plot(k_fold, acc_val_list, "g-")
    #plt.plot(eje_x, f1_10_list_, "r-")
    #plt.plot(eje_x, f1_20_list_, "y-")
    plt.legend(["Train accuracy", "Test accuracy"])#, "Top 20 list"])
    plt.title("Accurary in each cross validation fold")
    plt.xlabel("Number of fold")
    plt.ylabel("Accuracy")
    plt.savefig(tituloIm + '.jpg')  
        
    return (sumAcc/6)# average AUC values 
        
        
def LoadDatasets(pickle_file = "dataset.pickle"):
    f        = open(pickle_file, "rb")
    Datasets = pickle.load(f)
    
    return Datasets


if __name__ == "__main__":	
    
    ### ITERAR PARA CADA MFCC QUE HE SACADO

    Datasets = LoadDatasets("../02-Data_out/Xy_7_MFCC1.pickle")
    X_train, X_test, y_train, y_test = train_test_split(Datasets["X"], Datasets["y"], test_size = 0.1, random_state=42)
    acc_7_MFCC1 = cross_validation(X_train, y_train, "CNN1_7_MFCC1")
    print("ACC 7 SECONDS MFCC1 -->" + str(acc_7_MFCC1))
    
    
    Datasets = LoadDatasets("../02-Data_out/Xy_7_MFCC2.pickle")
    X_train, X_test, y_train, y_test = train_test_split(Datasets["X"], Datasets["y"], test_size = 0.1, random_state=42)
    acc_7_MFCC2 = cross_validation(X_train, y_train, "CNN1_7_MFCC2")
    print("ACC 7 SECONDS MFCC2 -->" + str(acc_7_MFCC2))


    Datasets = LoadDatasets("../02-Data_out/Xy_14_MFCC1.pickle")
    X_train, X_test, y_train, y_test = train_test_split(Datasets["X"], Datasets["y"], test_size = 0.1, random_state=42)
    acc_14_MFCC1 = cross_validation(X_train, y_train, "CNN1_14_MFCC1")
    print("ACC 14 SECONDS MFCC1 -->" + str(acc_14_MFCC1))


    Datasets = LoadDatasets("../02-Data_out/Xy_14_MFCC2.pickle")
    X_train, X_test, y_train, y_test = train_test_split(Datasets["X"], Datasets["y"], test_size = 0.1, random_state=42)
    acc_14_MFCC2 = cross_validation(X_train, y_train, "CNN1_14_MFCC2")
    print("ACC 14 SECONDS MFCC2 -->" + str(acc_14_MFCC2))

    

    Datasets = LoadDatasets("../02-Data_out/Dataset_MFCC_h5_24-48_segs.pickle")
    X_train, X_test, y_train, y_test = train_test_split(Datasets["X"], Datasets["y"], test_size = 0.1, random_state=42)
    acc_24_48_MFCC_h5 = cross_validation(X_train, y_train, "CNN1_24-48_MFCC_h5")
    print("ACC 24-48 SECONDS MFCC H5-->" + str(acc_24_48_MFCC_h5))  


    Datasets = LoadDatasets("../02-Data_out/Dataset_MFCC_h5_12-24_segs.pickle")   
    X_train, X_test, y_train, y_test = train_test_split(Datasets["X"], Datasets["y"], test_size = 0.1, random_state=42)
    acc_12_24_MFCC_h5 = cross_validation(X_train, y_train, "CNN1_12-24_MFCC_h5")
    print("ACC 12-24 SECONDS MFCC H5-->" + str(acc_12_24_MFCC_h5))  
    
    
    Datasets = LoadDatasets("../02-Data_out/Dataset_MFCC_h5_6-12_segs.pickle")
    X_train, X_test, y_train, y_test = train_test_split(Datasets["X"], Datasets["y"], test_size = 0.1, random_state=42)
    acc_6_12_MFCC_h5 = cross_validation(X_train, y_train, "CNN1_6-12_MFCC_h5")
    print("ACC 6-12 SECONDS MFCC H5-->" + str(acc_6_12_MFCC_h5))  
    
   
    
    

	# plt.plot(range(1, 11), history.acc)
	# plt.xlabel('Epochs')
	# plt.ylabel('Accuracy')
	# plt.show()
    
    
    
    # rounded = [round(x) for x in y_pred[:][0]]
    # print(rounded)


# model.save(model_path)

