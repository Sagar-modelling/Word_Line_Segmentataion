# Word_Line_Segmentataion(Lines and words Segmentation of IAM Forms dataset using OpenCV)

Offline Handwritten Text Recognition (HTR) systems transcribe text contained in scanned images into digital text.

In general, developed HTR consists of several parts, which are responsible for processing pages with full text (scanned or photographed), dividing them into lines, splitting the resulting lines into words and following recognition of words from them.

For solve the problem of recognition, it was decided to use Neural Network (NN). It consists of convolutional NN (CNN) layers, recurrent NN (RNN) layers and a final Connectionist Temporal Classification (CTC) layer.

But in this notebook only the task of page segmentation is highlighted. It was decided to do this in several different ways shown below.
