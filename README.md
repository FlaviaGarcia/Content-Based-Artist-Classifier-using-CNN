# Content-based artist classifier using CNN
<p>tUsing Million Song Dataset, an artist classifier has been developed. In order to this, different network architectures have been designed. 
<p>The inputs of the networks are in 02-Data_out folder, while the scripts to train and get the accuracy of each network are in 01-Development_cod folder.
<p>There are 4 different network designs: 
	<p>- CV_CNN_1_.py
	<p>- CV_CNN_2_.py
	<p>- CV_CNN_tesis_.py
	<p>- CV_CNN_articulo_.py

<p>Each network has 5 different inputs to be compared:
	<p>- Xy_7_MFCC1.pickle                 --> MFCCs extracted with MFCC_1.py on songs fragments of 7 seconds
	<p>- Xy_7_MFCC2.pickle                 --> MFCCs extracted with MFCC_2.py on songs fragments of 7 seconds
	<p>- Xy_14_MFCC1.pickle                --> MFCCs extracted with MFCC_1.py on songs fragments of 14 seconds
	<p>- Xy_14_MFCC2.pickle                --> MFCCs extracted with MFCC_2.py on songs fragments of 14 seconds
	<p>- Dataset_MFCC_h5_24-48_segs.pickle --> MFCCs precalculated by the dataset developers on songs fragments of 24-48 seconds
	<p>- Dataset_MFCC_h5_12-24_segs.pickle --> MFCCs precalculated by the dataset developers on songs fragments of 12-24 seconds
	<p>- Dataset_MFCC_h5_6-12_segs.pickle  --> MFCCs precalculated by the dataset developers on songs fragments of 6-12 seconds
<p>
<p>To train the networks is necessary to have a GPU. 
