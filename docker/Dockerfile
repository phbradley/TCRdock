FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL authors="Elias Ball" \
    title="TCRdock" \
    Version="2.0.0" \
    description="Docker image containing all dependencies to run TCRdock (https://github.com/phbradley/TCRdock, created by Philip Bradley)"

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    git \
    wget

WORKDIR /opt

# Clone TCRdock
RUN git clone --depth 1 --branch v2.0.0 https://github.com/phbradley/TCRdock.git

# Install conda
RUN mkdir -p /opt/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh && \
    bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 && \
    rm -rf /opt/miniconda3/miniconda.sh
ENV PATH="/opt/miniconda3/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/miniconda3/lib:$LD_LIBRARY_PATH"

# Install conda dependencies
RUN conda install -y -c conda-forge \
    openmm=7.7.0 \
    pdbfixer=1.8.1 \
    python=3.10 \
    pip=23.3 \
    matplotlib=3.4.3 && \
    conda clean --all --force-pkgs-dirs --yes

# Install python dependencies
RUN pip install --upgrade --no-cache-dir \
    absl-py==1.0.0 \
    biopython==1.79 \
    chex==0.0.7 \
    dm-haiku==0.0.10 \
    dm-tree==0.1.8 \
    docker==5.0.0 \
    immutabledict==2.0.0 \
    jax==0.3.25 \
    https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn805-cp310-cp310-manylinux2014_x86_64.whl \
    ml-collections==0.1.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scipy==1.11.1 \
    tensorflow-cpu==2.13.0

# Temporarily allow write to /opt/TCRdock for all users
RUN chmod a+rw -R /opt/TCRdock

# Create and switch to non-root user
RUN useradd -ms /bin/bash tcrdockuser
USER tcrdockuser

# Download model parameters and BLAST
WORKDIR /opt/TCRdock
RUN mkdir -p alphafold_params/params && \
    wget -O 'alphafold_params/params/params_model_2_ptm.npz' 'https://www.dropbox.com/s/e3uz9mwxkmmv35z/params_model_2_ptm.npz' && \
    wget -O 'alphafold_params/params/tcrpmhc_run4_af_mhc_params_891.pkl' 'https://www.dropbox.com/s/jph8v1mfni1q4y8/tcrpmhc_run4_af_mhc_params_891.pkl' && \
    python download_blast.py

# Set correct access permissions to all users
USER root
RUN chown -R root:root /opt/TCRdock && \
    chmod a+r -R /opt/TCRdock && \
    chmod a-w -R /opt/TCRdock

# Switch back to non-root user
USER tcrdockuser

# Create data directory
RUN mkdir /home/tcrdockuser/data
WORKDIR /home/tcrdockuser/data
