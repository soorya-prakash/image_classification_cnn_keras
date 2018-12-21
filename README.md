# image_classification_cnn_keras
Introduction:
  Using convolutional neural networks classifying  the indian one rupee,two rupee and five rupee coin images. 
Aim:
To classify indian coins and obtain high accuracy.

Requirements/dependency:
1)keras,cv2

image folder:
  contains two repo namely train and test.
  1)train repo is used for training model
  2)test repo is used for evaluate and test the build model.

script:
indian_coin_images_classificatio1.py:
  step 1:mport the required frameworks and libs 
  step 2:define the train and test path
  step 3:read_img_train() and read_img_test() functions will read the images from the defined path and resize the images into (150*150)pixels.
         Both funtions returns the labels and array of numbers ranging from 0 to 1 as data.
  step 4:function createModel() will define architecture of the convolution neural networks and returns the model.
  step 5:reshape the image shape into neural network learnable way (number of images,height,width,1)
  step 6:one hot encode the train data labels(here 0:5 rupee coin, 1:1 rupee coin, 2: 2 rupee coin)
  step 7:repeat step 5 and step 6 for test data
  step 8:used Imagedatagenerator() for data agumentation (because of less amount of train data we used data agumentation which gives random
         views to the images such as rotation,zoom etc.)
  step 9:initialize the model to object (here model1)
  step 10: define batch_size,epochs and compile the model(used 'rmsprop' optimizer ,loss as 'categorial_crossentropy')
  step 11: predict the model1 with test data which return the predicted values for the number of test data given as input to model.
  step 12: model1.evaluate() gives the loss and accuracy of the model 
  
