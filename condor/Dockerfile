FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git util-linux \
    && rm -rf /var/lib/apt/lists/*
COPY condor/requirements-condor.txt /tmp/requirements-condor.txt
RUN python -m pip install --no-cache-dir --upgrade pip==24.3.1 \
    && python -m pip install --no-cache-dir -r /tmp/requirements-condor.txt
WORKDIR /workspace
# The live repository is home-mounted by SIC. PYTHONPATH is set by setup_environment.sh.
CMD ["/bin/bash"]
