FROM --platform=linux/amd64 nvcr.io/nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (curl/git for setup, nginx/ssh/jupyter for the start script)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    openssh-server \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN mkdir /workspace
RUN uv venv && uv pip install jupyterlab

# Clone and set up the project
RUN git clone https://github.com/ahsanMah/tiny-rl.git ~/tiny-rl; \
cd ~/tiny-rl/mini-dreamer && \
git checkout jax-dreams && \
uv add jax[cuda13] && \
uv sync


# Add the container start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

WORKDIR /root/tiny-rl/mini-dreamer

EXPOSE 8888 22

CMD ["/start.sh"]
