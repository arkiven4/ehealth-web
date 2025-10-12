FROM tensorflow/tensorflow

# RUN apt-get update -yq \
#     && apt-get -yq install curl gnupg ca-certificates zip \
#     && curl -L https://deb.nodesource.com/setup_16.x | bash \
#     && apt-get update -yq \
#     && apt-get install -yq \
#         dh-autoreconf=20 \
#         nodejs build-essential gcc libsndfile1

# RUN if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi
# RUN python3 -m pip install --upgrade pip

# #RUN pip install tensorflow
# RUN pip install numpy
# RUN pip install matplotlib
# RUN pip install seaborn
# RUN pip install librosa==0.9.2
# #RUN pip install numba==0.48.0

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    zip \
    build-essential \
    gcc \
    libsndfile1 \
    libgl1 && \
    curl -L https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -yq nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    librosa==0.9.2 \
    scipy \
    joblib \
    tensorflow \
    tensorflow_hub \
    protobuf==3.20.3 \
    torch \
    torchaudio \
    pillow \
    ultralytics

ENV NUMBA_CACHE_DIR=/tmp/
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN mkdir -m 777 /tmp/NUMBA_CACHE_DIR /tmp/MPLCONFIGDIR $YOLO_CONFIG_DIR
ENV NUMBA_CACHE_DIR=/tmp/NUMBA_CACHE_DIR/
ENV MPLCONFIGDIR=/tmp/MPLCONFIGDIR/

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm install --verbose

COPY . .
