# Imagem com uv pré-instalado sobre python:3.12-slim-bookworm, compatível com
# requires-python>=3.12, sem precisar inicializar o uv via pip.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# UID/GID do usuário do container casam com o usuário do host por padrão, para
# que arquivos criados em bind mounts (ex.: data/) não fiquem com dono root.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" gatefall \
    && useradd --uid "${UID}" --gid "${GID}" --create-home gatefall

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN chown "${UID}:${GID}" /app

USER gatefall

COPY --chown=${UID}:${GID} pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY --chown=${UID}:${GID} . .
RUN uv sync --locked

CMD ["bash"]
