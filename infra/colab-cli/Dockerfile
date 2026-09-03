## Container for google-colab-cli (https://github.com/googlecolab/google-colab-cli), which only
## supports Linux/macOS natively — this dev machine is Windows, so the CLI runs here instead of
## under WSL2 directly, consistent with how the rest of algoforge's Linux-only tooling runs in
## containers (see ../docker-compose.dev.yml). See ./README.md for setup and usage.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir google-colab-cli \
    # google-colab-cli's own pyproject.toml pins jupyter-kernel-client to Google's own fork
    # (git+https://github.com/googlecolab/jupyter-kernel-client.git) via a `[tool.uv.sources]`-
    # style override that plain `pip install google-colab-cli` does not honor -- pip instead
    # resolves the generic PyPI package of the same name (an unrelated project by Datalayer),
    # which exposes a different API (`JupyterKernelClient`, not `KernelClient`) and breaks
    # `colab exec` at runtime with `AttributeError: module 'jupyter_kernel_client' has no
    # attribute 'KernelClient'`. Force-reinstall the correct fork over it, after
    # google-colab-cli's own install has already resolved every other dependency normally.
    && pip install --no-cache-dir --force-reinstall --no-deps \
       "git+https://github.com/googlecolab/jupyter-kernel-client.git"

WORKDIR /work

# Auth tokens/settings land in ~/.config/colab-cli by default — see docker-compose.yml's
# colab_cli_config volume, which persists this across container recreation.

# No long-running process of its own; kept alive so `docker compose exec` can run one-off
# `colab ...` commands against it.
CMD ["sleep", "infinity"]
