# Hand Writing Recognition Using Deep Learning Method
## Framework: Tensorflow, Keras

Keras has become so popular, that it is now a superset, included with TensorFlow releases now! If you're familiar with Keras previously, you can still use it, but now you can use tensorflow.keras to call it. By that same token, if you find example code that uses Keras, you can use with the TensorFlow version of Keras too. In fact, you can just do something like:
```
import tensorflow.keras as keras
```

## Programs
- MnistRecognition.py: a test program, using deep neural network to recognize hand writing 0-9
- TrainingDataProcessing.py: read from data set, create training and testing image array and saved
- CnnModelTraining.py: create CNN model, fit with the training and testing data

## dockerfile
```
FROM qiime2/conda-channel-resource

WORKDIR /tensorflow-deeplearning

# Copy the current directory contents into the container at /app
COPY . /tensorflow-deeplearning

# Install any needed packages
RUN conda install python=3.6
RUN conda install tensorflow
RUN conda install tqdm
RUN conda install opencv
RUN conda install matplotlib

# Make port 80 available to the world outside this container
EXPOSE 80

CMD ["python", "CnnModelTraining.py"]
```
- run docker image mount with local data folder
```
docker run -v /Users/.../DeepLearnignUsingTensorflowAndKeras/data:/tensorflow-deeplearning/data --rm keras-deeplearning
```
