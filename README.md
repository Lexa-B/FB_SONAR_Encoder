# Sonar Encoder

## Overview

This project is a simple encoder that uses the Sonar concept-level auto-encoder to encode text into a PyTorch tensor.

## Installation

### Install the dependencies
```bash
uv sync
```

### Activate the virtual environment
```bash
source .venv/bin/activate
```

## Usage

### Arguments

* `-t`: Input text to encode.
* `-o`: Name to save the encoded tensor.
* `-l`: Language to encode from (FLORES-200 code) (e.g., jpn_Jpan, eng_Latn).
* `-r`: Save as raw PyTorch file (.pt) over .safetensors.
* `-v`: Enable verbose output.

### Example
The command below will encode the text "Hello, world!" from English to a PyTorch tensor and save it to the `OutputData` directory with the name `hello_world.pt`. 

```bash
uv run ./fb_sonar_encoder/Encoder.py -t "Hello, world!" -l eng_Latn -o "hello_world" -v
```
