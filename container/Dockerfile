# Copyright 2020 Juv Chan. All Rights Reserved.
FROM tensorflow/tensorflow:2.3.0-gpu

LABEL maintainer="Juv Chan <juvchan@hotmail.com>"

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl
RUN pip install --no-cache-dir --upgrade pip tensorflow-hub tensorflow-datasets sagemaker-tensorflow-training

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /code /opt/ml/code
WORKDIR /opt/ml/code

RUN chmod 755 serve