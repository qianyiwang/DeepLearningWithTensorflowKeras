# Use an official Python runtime as a parent image
FROM publicisworldwide/python-conda

WORKDIR /tensorflow-deeplearning

# Copy the current directory contents into the container at /app
COPY . /tensorflow-deeplearning

# Install any needed packages specified in requirements.txt

RUN conda install tensorflow
RUN conda install tqdm
RUN conda install opencv
RUN conda install matplotlib

# Make port 80 available to the world outside this container
EXPOSE 80

CMD ["python", "CnnModelTraining.py"]
