# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:01:05 2021

@author: mbadal1996
"""

# =======================================================================
# confusion_matrix v1.0
#
# Comments:
# The purpose of this python code is to provide a convenient foundation for 
# generating a confusion matrix for train and validation data (or any other
# data desired). The code provided can be used for binary or multi-class
# problems. The user can input number of classes in 'num_class' below.
#
# NOTE: Torch tensors were used here instead of numpy arrays since the data
# were assumed coming from Pytorch output and it was simply more convenient. 
# The torch tensors could be replaced by numpy arrays if not using Pytorch. 
# In that case the use of .item() would not be necessary.
#
# =======================================================================


# Python
import matplotlib.pyplot as plt

# Pytorch
import torch


# Input number of classes (chosen by user)
num_class = 3

# Create fake example data to plot in confusion matrix:
# NOTE: These would normally be the predictions ('predict') from your 
# model (for training or validation data) and the true values ('true') 
# from your labeled data (again for training or validation data).   
predict_train = torch.randint(low=0,high=num_class,size=(300,))
y_true_train = torch.randint(low=0,high=num_class,size=(300,))
predict_val = torch.randint(low=0,high=num_class,size=(100,))
y_true_val = torch.randint(low=0,high=num_class,size=(100,))


# Initialize tensors to store confusion values
ZZ_train = torch.zeros(num_class,num_class).long()  
ZZ_val = torch.zeros(num_class,num_class).long()  

# Step through "True" and "Predicted" values and update confusion matrix
for j in range(0,len(y_true_train)):
    # For each "j" add +1 to ZZ at coordinate (predict_train[j],y_true_train[j])
    ZZ_train[predict_train[j].item(),y_true_train[j].item()] += 1

for j in range(0,len(y_true_val)):
    # For each "j" add +1 to ZZ at coordinate (predict_val[j],y_true_val[j])
    ZZ_val[predict_val[j].item(),y_true_val[j].item()] += 1 


# -------------------------------------------------------
# Create plotting function for confusion matrices

def confplot_func(input_data):

    # Create list of axis tick marks based on number of classes
    axes = []  # initialize list
    for i in range(0,num_class):
        axes.append(i)
        
    # Plot results of confusion matrix ZZ for Train and Validation data
    plt.imshow(input_data,extent=(-0.5,num_class-0.5,num_class-0.5,-0.5),
               interpolation='none',cmap='coolwarm')
    plt.xlabel('True Classes', fontsize = 12)
    plt.ylabel('Predicted Classes', fontsize = 12)
    plt.colorbar(fraction=0.145, pad=0.055, aspect=5.5)
    plt.xticks(axes, fontsize = 11)
    plt.yticks(axes, fontsize = 11)

    # Insert count text for each box in confusion matrix
    for i in range(0,num_class):
        for j in range(0,num_class):
            plt.text(j, i, format(input_data[i,j], 'd'), 
            horizontalalignment="center", verticalalignment="center", 
            color="white", fontsize = 16, fontweight = 'bold')

# -------------------------------------------------------

# Plot results of confusion matrix ZZ for Train and Validation data
plt.figure(figsize=(9,9))  # plot figures as subplots with given size

plt.subplot(1,2,1)
confplot_func(ZZ_train)  # pass train data to plotting function
plt.title('Confusion Matrix Train')

plt.subplot(1,2,2)
confplot_func(ZZ_val)  # pass validation data to plotting function
plt.title('Confusion Matrix Val')

plt.tight_layout(pad=1.5, w_pad=2.5, h_pad=1.0)

plt.show()
