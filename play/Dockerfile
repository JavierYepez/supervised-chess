FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
USER $MAMBA_USER
ARG MAMBA_DOCKERFILE_ACTIVATE=1 
CMD micromamba run -n base fastapi run