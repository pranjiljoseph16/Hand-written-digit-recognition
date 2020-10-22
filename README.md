# Hand-written-digit-recognition
1. Handwritten Digits Image Processing Dataset
2. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
3. We have fetched the dataset from keras deep learning library.
4. Data Preprocessing: The dataset is structured as a 3-dimensional array of instance, image width and image height. For a multi-layer perceptron model, we must reduce the images down into a vector of pixels. We did this transform easily using the reshape() function on the NumPy array.

   We have also reduced our memory requirements by forcing the precision of the pixel values to be 32 bits, the default precision used by Keras anyway. The pixel values are gray scale between 0 and 255.

   We have normalized the pixel values to the range 0 and 1 by dividing each value by the maximum of 255. Finally, the output variable is an integer from 0 to 9. To transform the vector of class integers into a binary matrix, we applied the built-in np_utils.to_categorical() helper function in Keras.

5. Modelling –
         
        i. kNN classifier: Modelling was done with default number of neighbours i.e. k=5. Accuracy is 96.88% Model evaluation is done by                              using classification report – precision, recall, f1-score.
        
        ii. MLP classifier: Hypertuning parameters – (activation=&#39;relu&#39;, hidden_layer_sizes= (200, 200), alpha = 0.3) 
        
                             Accuracy on training score: 89.8% 
                              
                              Accuracy on testing score: 89.06%.
        
        iii. MLP keras:  We have used Sequential model for building the network. Added the Dense layer, ReLU activation function. The last                          layer is a softmax layer as it is a multiclass classification problem. we configured the optimizer to be adam. We                          specified the loss type which is categorical cross entropy and the metrics (accuracy in this case) which we want                             to track during the training process.
        
                        We have used the data in the history object to plot the loss and accuracy curves to check how the training process went.
                        
                        We have observed a clear sign of overfitting. Initially validation loss decreasing but then increasing gradually. There is a substantial difference between training and validation accuracy. It shows overfitting. To overcome that we are doing regularization by adding dropout layer.
                        
                        To overcome this problem, we have done regualrization by adding adropout layer. Then we have predictedthe data using test dataset.
                        
                        Accuracy is 97.72%.

        iv. CNN: For implementing a CNN, we have stacked up Convolutional Layers, followed by Max Pooling layers. We have used 3 convolutional layers and 1 fully connected layer. We have used convolution layers with 32 and 64 filters with a kernel size of 3x3 and a max pooling layer with kernel size 2x2. We have also included dropout layer with dropout ratio 0.25 to avoid overfitting. In the final lines, we add the layer which performs the classification among 10 classes using a softmax layer.
        
                 Since it is a 10 class classification problem, we have used a categorical cross- entropy loss and adam optimizer to train the network.
    
                 We have used the data in the history object to plot the loss and accuracy curves to check how the training process went.
        
                 We got test accuracy greater than training accuracy. We have calculated predictions and incorrect predictions.

                 Accuracy is 99.4% 

        v. fastai: Uploaded data using fastai provided academic dataset – untar Extracted data using ImageDataBunch bu using transformation as a part of data augmentation, batch size – 64 and normalization – stats of MNIST. Modelled using model architecture resnet 34 with accuracy as metrics. Fitted the data inside the learner so that it could learn the data.  There is no overfitting since validation loss is always lesser than training loss. Saved the weights and interpreted the results using ClassificationInterpretation. Plotted top_losses, confusion matrix and interpreted most confused entries. Next, we have unfreezed the model to train the whole layers of the model. We tried to find the meagre learning rate for the initial layers.
  
                   Accuracy is 99.31%.
