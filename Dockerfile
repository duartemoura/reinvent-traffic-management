FROM python:3.7

# Install SUMO dependencies and other necessary tools
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    graphviz \
    wget

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the tf_gpu conda environment
RUN conda create --name tf_gpu python=3.7 -y

# Activate the tf_gpu environment and install tensorflow-gpu
RUN conda run -n tf_gpu conda install tensorflow-gpu -y

# Install pydot and other Python dependencies
RUN conda run -n tf_gpu pip install pydot
COPY requirements.txt .
RUN conda run -n tf_gpu pip install --no-cache-dir -r requirements.txt

# Set up working directory
WORKDIR /opt/ml/code

# Copy project files
COPY . .

# Copy and set permissions for the run.sh script
COPY run.sh /opt/ml/code/run.sh
RUN chmod +x /opt/ml/code/run.sh

# Set environment variables for SUMO
ENV SUMO_HOME=/usr/share/sumo

# Set the entrypoint to run the wrapper script
ENTRYPOINT ["./run.sh"]