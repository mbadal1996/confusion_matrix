# confusion_matrix
The python code confusion_matrix provides a flexible foundation for creating attractive confusion matrices for machine learning results.

=======================================================================

confusion_matrix v1.0

Comments:
The purpose of this python code is to provide a convenient foundation for 
generating a confusion matrix for train and validation data (or any other
data desired). The code provided can be used for binary or multi-class
problems. The user can input number of classes in 'num_class' below.

NOTE: Torch tensors were used here instead of numpy arrays since the data
were assumed coming from Pytorch output and it was simply more convenient. 
The torch tensors could be replaced by numpy arrays if not using Pytorch. 
In that case the use of .item() would not be necessary.

=======================================================================
