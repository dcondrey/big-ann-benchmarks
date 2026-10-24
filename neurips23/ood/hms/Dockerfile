FROM neurips23

RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.0
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip3 install maturin

# Holographic Memory with million-query graph projection and batch array queries
# (writerslogic/holographic-memory, branch feat/bigann-ood).
ARG HMS_COMMIT=7511dd1e
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/writerslogic/holographic-memory.git /home/app/holographic-memory \
    && cd /home/app/holographic-memory && git checkout ${HMS_COMMIT} \
    && RUSTFLAGS="-C target-cpu=native" maturin build --release --features python -i python3 -o /home/app/wheels \
    && pip3 install /home/app/wheels/*.whl
WORKDIR /home/app
