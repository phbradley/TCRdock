################## BASE IMAGE ######################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

################## METADATA ######################
LABEL base_image="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
LABEL version="1"
LABEL software="TCRdock"
LABEL software.version="1.0.0"
LABEL software.author="Philip Bradley <pbradley@fredhutch.org>"
LABEL about.summary="python tools for TCR:peptide-MHC modeling and analysis"
LABEL about.home="https://github.com/phbradley/TCRdock"
LABEL about.documentation="https://github.com/phbradley/TCRdock"
LABEL about.license_file="https://github.com/phbradley/TCRdock"
LABEL about.license="MIT (Apache-2.0 for original Alphafold)"
LABEL about.tags="Structure Prediction:Immunology"
MAINTAINER Philip Bradley <pbradley@fredhutch.org>

################## INSTALLATION ######################
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    git \
    wget

# Create non-root user
RUN useradd -ms /bin/bash tcrdockuser
USER tcrdockuser
WORKDIR /home/tcrdockuser/

# Clone TCRdock
RUN git clone https://github.com/phbradley/TCRdock.git && \
    cd TCRdock && \
    git checkout af232_update
# TODO remove branching and do release instead

# Install conda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh
ENV PATH="/home/tcrdockuser/miniconda3/bin:$PATH"
ENV LD_LIBRARY_PATH="/home/tcrdockuser/miniconda3/lib:$LD_LIBRARY_PATH"

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

# Download network parameters and BLAST
WORKDIR /home/tcrdockuser/TCRdock
RUN mkdir -p alphafold_params/params && \
    wget -O 'alphafold_params/params/params_model_2_ptm.npz' 'https://www.dropbox.com/s/e3uz9mwxkmmv35z/params_model_2_ptm.npz' && \
    wget -O 'alphafold_params/params/tcrpmhc_run4_af_mhc_params_891.pkl' 'https://www.dropbox.com/s/jph8v1mfni1q4y8/tcrpmhc_run4_af_mhc_params_891.pkl' && \
    python download_blast.py

# Create data directory
RUN mkdir /home/tcrdockuser/data
WORKDIR /home/tcrdockuser/data
