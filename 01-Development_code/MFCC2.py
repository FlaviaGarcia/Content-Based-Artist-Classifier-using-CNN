#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:21:53 2018

@author: flaviagv

From the MP3 files the MFCCs are extracted (matrices). This MFCCs have been extracted by setting the following parameters:
n_mels = 128, n_fft = 2048, hop_length = 1024 in fragments of 7s and 14s of each song. The resulting MFCC matrices have been stored 
in: 
    - 02-Data_out/Xy_14_MFCC2.pickle
    - 02-Data_out/Xy_7_MFCC2.pickle

"""

import librosa
import librosa.display
import numpy as np
import os 
import glob 
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import math


def get_numSelected(datasetFolder_name, extension, segundos): 
    ### Obtener el  numero comun de canciones de todos los artistas
    num_selected = []

    for artist_folder in [x for x in os.listdir(datasetFolder_name) if "." not in x]:
        print("Artist : " + artist_folder)
        num_empty = 0 
        artist_folder_path = os.path.join(datasetFolder_name, artist_folder)
        mp3_files = glob.glob1(artist_folder_path,"*.%s"%extension)
        # hacer esto solo para extension .mp3
        if extension == "mp3":
            for mp3_file in mp3_files:
                print("mp3 File: " + mp3_file)
                
                full_path_mp3 = os.path.join(artist_folder_path, mp3_file)
                
                #audio = MP3(full_path_mp3)
                #length = audio.info.length           
                #tag = TinyTag.get(full_path_mp3)
                #length = tag.duration
                
                audio_sampled_tmp, fs = librosa.load(full_path_mp3)
                length = len(audio_sampled_tmp)/fs
                
                #statinfo = os.stat(full_path_mp3)
                if  length < segundos: #statinfo.st_size == 0 : # or LENGTH_SONG_SECONDS < segundos:
                    os.remove(full_path_mp3)
                    num_empty += 1
                    print("Numero registro borrado: %s --> %s \n" %(num_empty, full_path_mp3))
        
        print ("nº de registros borrados: " + str(num_empty))
        num_mp3_artist = len(mp3_files)-num_empty
        
        if num_mp3_artist != 0:
            num_selected.append(num_mp3_artist)
        else: 
            #eliminar esa carpeta
            os.rmdir(artist_folder)
            print(artist_folder, "deleted\n")
        
        print("artist folder:", artist_folder, "num: ", num_mp3_artist, "\n")
        
    return (min(num_selected))
  



def create_X_y (datasetFolder_name, num_selected, seconds, num_artists = 50):
    X = []  # Creamos lista vacia 
    y = []
    #### Nos tendremos que recorrer las carpetas de artistas 
    artist_tag = 0 #Sera un contador
    #init      = 0 
    
    for artist_folder in [x for x in os.listdir(datasetFolder_name) if "." not in x]:
        
        # El vector de etiquetas (y) lo vamos rellenando aqui: 
        
        mp3_filesInFolder = glob.glob1(os.path.join(datasetFolder_name, artist_folder),"*.mp3") 
        
        for mp3_file in mp3_filesInFolder[:num_selected]:
         
            path_mp3 = os.path.join(datasetFolder_name, artist_folder, mp3_file)
            # COGER SOLO LOS 30 PRIMEROS SEGUNDOS            
            #AudioSegment.converter = which("ffmpeg")
            
            #song = AudioSegment.from_mp3(path_mp3)
             
            #crop_song = song[:30000]
            
            #mp3_file_split = mp3_file.split(".")[0]
            
            #path_mp3_crop = os.path.join(datasetFolder_name, artist_folder, "%s_crop.mp3"%mp3_file_split)
            
            
            #crop_song.export(path_mp3_crop, format = "mp3")
            
            #path_mp3 = "TRAEEUX128F425E0E6.mp3"
            
            audio_sampled_tmp, fs = librosa.load(path_mp3)

            # Nos quedamos solo con los primeros 30 segundos 
            audio_sampled = audio_sampled_tmp[:(fs * 28)]
            
            #print ("Sample rate: {} Hz, Duration: {} sec".format(fs, y.shape[0]/fs))
            
            #t = np.arange(0, y.shape[0]/fs, 1./fs )
            #plt.plot(t, y)
    

            #SHORT-TIME FOURIER TRANSFORM
            # Si 30 segs la cancion -> (129,1324) --> 500
#            stft = librosa.core.stft(audio_sampled,         # mi señal muestreada              
#                                     n_fft = 256,           # tamaño ventana
#                                     hop_length = 500,     # number audio of frames between STFT columns. If unspecified, defaults win_length / 4.
#                                     #win_length = None,    # The window will be of length win_length and then padded with zeros to match n_fft. f unspecified, defaults to win_length = n_fft
#                                     window ='hann', 
#                                     center = True,         # y is padded so that frame D[:, t] is centered at y[t * hop_length].
#                                     #dtype = <class 'numpy.complex64'>, 
#                                     pad_mode = 'reflect')  # the padding mode to use at the edges of the signal
#              
#
#            X.append(stft[:, 0:300])     
#            X.append(stft[:, 300:600])
#            X.append(stft[:, 600:900])
#            
            S = librosa.feature.melspectrogram(audio_sampled, 
                                               n_mels = 128,
                                               sr = fs, 
                                               S  = None, 
                                               n_fft = 2048, 
                                               hop_length = 1024,
                                               power = 2.0)
            
            #S.shape
            
            # Cogemos todas las frecuencias, y cierto intervalo en t 
            # De cada cancion extraigo 3 partes para tener mas inputs
            if (seconds == 14 ):
                X.append(S[:, 0:300])     
                X.append(S[:, 300:600])
                y = y + [artist_tag, artist_tag]
            elif (seconds == 7):
                X.append(S[:, 0:150])     
                X.append(S[:, 150:300])
                X.append(S[:, 300:450])
                X.append(S[:, 450:609])
                y = y + [artist_tag, artist_tag, artist_tag, artist_tag]
            # X.append(S[:, 600:900])
            
            
            #y = y + [artist_tag, artist_tag]#, artist_tag]
            
        # Actualizamos valores para la siguiente iteracion         
        artist_tag += 1
    
    return (X,y)
   
    

def get_Train_Test_sets (X, y, iteracion = 0):
    # X sigue siendo una lista 
    
    # iteracion es un numero que tiene que estar en range(5), sino lanzar una excepcion 
    
    
    # quedarnos con 4/5(80%) para el train y 1/5 para el test(20%) 
    # En cada iteracion coger un trozo diferente del test, y que el resto sean train
    
    
    num_elems = len(y)//5
    # Si hay restantes, se lo queda el train
    
    # primer bloque  -> [0:num_elems]
    # segundo bloque -> [num_elems*1:num_elems*2]
    # tercer bloque  -> [num_elems*2:num_elems*3]
    # cuarto bloque  -> [num_elems*3:num_elems*4]
    # quinto bloque  -> [num_elems*4:num_elems*5]

    # el numero del bloque del test te lo va a dar la iteracion 
    
    if not iteracion in range(5):
        raise ValueError("The iteration variable is not any of this values: [0, 1, 2 ,3, 4]")
    
    
    # HACER UN RANDOM TANTO DE X como de y, par que este bien mezclado 
    np.random.seed(133)
    np.random.shuffle(X)
    np.random.shuffle(y)

    #np.random.permutation(len(y)).shape
    
    
    X_test_tmp = X[num_elems*iteracion : num_elems*(iteracion+1)]  
    y_test_tmp = y[num_elems*iteracion : num_elems*(iteracion+1)]
    
    sep    = len(X_test_tmp)//2
    X_test = X_test_tmp[:sep]
    y_test = y_test_tmp[:sep]
    
    X_val = X_test_tmp[sep:]
    y_val = y_test_tmp[sep:]
    
    X_train = X[0:num_elems*iteracion] + X[num_elems*(iteracion+1):len(X)]
    y_train = y[0:num_elems*iteracion] + y[num_elems*(iteracion+1):len(y)]
    
    return(X_train, y_train, X_val, y_val, X_test, y_test)
    


def SaveDatasets(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, pickle_file = "dataset.pickle"):
    try:
        f = open(pickle_file, 'wb')
        save = {
        'X_train': X_train,
        'y_train': y_train,
        "X_val": X_val,
        "y_val": y_val,
        'X_test': X_test,
        'y_test': y_test,
        'num_classes': num_classes,
        }
        pickle.dump(save, f, 2)
        #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    
    return pickle_file        
        


def SaveXy(X, y, pickle_file = "Xy.pickle"):
    try:
        f = open(pickle_file, 'wb')
        save = {
        'X': X,
        'y': y,
        }
        #pickle.dump(save, f, 2)
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    
    return pickle_file 


if __name__ == '__main__':
    
    # hop_length = 221
    window_size = 551

    fs = 22050

    segundos = math.ceil(900 * window_size/fs) # redondear a la alta 


    segundos = 28
    
    datasetFolder_name  = "../totalArtistSet-2"
    num_selected = get_numSelected(datasetFolder_name, "mp3", segundos) 
    num_artists  = len([x for x in os.listdir(datasetFolder_name) if "." not in x])
   
    X_14, y_14 = create_X_y(datasetFolder_name, num_selected, 14, num_artists)


    # Comprobar que todas las MFCC contenidos en X tienen la misma shape
    mal = 0
    pos =[]
    for position in range(len(X_7)):
        if X_7[position].shape != (128,150):
            mal += 1
            print(str(mal) + " --> index: ", str(position))
            pos.append(position)
    # todo bien         
    len(X_7)

    a = [x for x in X_7 if x.shape == (128,150)]
    b = np.delete(np.array(y_7),pos, None).tolist()
    
    X_train_14, y_train_14, X_val_14, y_val_14, X_test_14, y_test_14 = get_Train_Test_sets (X_14, y_14)
   
    X_7, y_7 = create_X_y(datasetFolder_name, num_selected, 7, num_artists)
    X_train_7, y_train_7, X_val_7, y_val_7, X_test_7, y_test_7 = get_Train_Test_sets (X_7, y_7)

    SaveXy(X_14, y_14, pickle_file = "Xy_14_MFCC2.pickle")
    SaveXy(X_7, y_7, pickle_file = "../02-Data_out/Xy_7_MFCC2.pickle")
    
    SaveXy(a, b, pickle_file = "../02-Data_out/Xy_7_MFCC2.pickle")

    
    ## Save sets 
    pickle_file = SaveDatasets(X_train, y_train, X_val, y_val, X_test, y_test, num_artists, "../00-Data_in/dataset_MFCC2_Protocol2.pickle")











from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_14, y_14, test_size = 0.1, random_state=42)




