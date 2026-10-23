FROM neurips23

RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.0
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip3 install maturin

# Holographic Memory, pinned to the measured streaming candidate (alpha 1.0, 3-byte consolidated ids).
ARG HMS_COMMIT=082766eb2c0483d4d41cec8715c6fb793b49746e
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/writerslogic/holographic-memory.git /home/app/holographic-memory \
    && cd /home/app/holographic-memory && git checkout ${HMS_COMMIT} \
    && maturin build --release --features python -i python3 -o /home/app/wheels \
    && pip3 install /home/app/wheels/*.whl
WORKDIR /home/app
