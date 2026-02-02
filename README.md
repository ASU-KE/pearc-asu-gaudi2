# PEARC Conference Paper: Intel Gaudi2 on Kubernetes

This repository contains supplementary materials for our PEARC conference paper on deploying and benchmarking Intel Gaudi2 accelerators in Kubernetes environments.

## Contents

- **Kubernetes YAML Files**: Configuration files for deploying Gaudi2 workloads
- **Benchmark Results**: Performance measurements and analysis
- **Documentation**: Setup guides and deployment instructions

## Overview

This work demonstrates the deployment and performance characteristics of Intel Gaudi2 AI accelerators managed through Kubernetes orchestration at Arizona State University's research computing facilities.

## Prerequisites

- **HuggingFace token**: Create a Kubernetes Secret named `huggingface-token` with your HuggingFace Hub token so recipes can pull gated models:
  ```bash
  kubectl create secret generic huggingface-token --from-literal=HUGGINGFACE_HUB_TOKEN=<your-token> -n <namespace>
  ```
  Use the same secret name in the namespace where you deploy each recipe (e.g. `image-generation`, `aibrix-system-llm`).

- **Container registry**: Some recipes reference `registry.rc.asu.edu` (ASU Research Computing). Replace with your own registry URL and image if deploying elsewhere.
