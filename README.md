# Content-based artist classifier using CNN
Using Million Song Dataset, an artist classifier has been developed. In order to this, different network architectures have been designed. 
<p>The inputs of the networks are in 02-Data_out folder, while the scripts to train and get the accuracy of each network are in 01-Development_cod folder.
<p>There are 4 different network designs: 
	- CV_CNN_1_.py
	- CV_CNN_2_.py
	- CV_CNN_tesis_.py
	- CV_CNN_articulo_.py

Each network has 5 different inputs to be compared:
	- Xy_7_MFCC1.pickle                 --> MFCCs extracted with MFCC_1.py on songs fragments of 7 seconds
	- Xy_7_MFCC2.pickle                 --> MFCCs extracted with MFCC_2.py on songs fragments of 7 seconds
	- Xy_14_MFCC1.pickle                --> MFCCs extracted with MFCC_1.py on songs fragments of 14 seconds
	- Xy_14_MFCC2.pickle                --> MFCCs extracted with MFCC_2.py on songs fragments of 14 seconds
	- Dataset_MFCC_h5_24-48_segs.pickle --> MFCCs precalculated by the dataset developers on songs fragments of 24-48 seconds
	- Dataset_MFCC_h5_12-24_segs.pickle --> MFCCs precalculated by the dataset developers on songs fragments of 12-24 seconds
	- Dataset_MFCC_h5_6-12_segs.pickle  --> MFCCs precalculated by the dataset developers on songs fragments of 6-12 seconds

To train the networks is necessary to have a GPU. 
