# 1. Hardware Information
All experiments were conducted on a single `NVIDIA A100 Tensor Core GPU` device.


# 2. Virtual Environment Configuration

We configured two virtual environments based on the execution requirements:
- One for reproducing the time-series foundation models (see: `TS-LibProejct folder`, note that Python 3.8 is required here).
- One for reproducing the time-series large models (see: `LLMProject folder`, note that Python 3.10 is required here).
We recommend using `conda` to manage virtual environments, as it simplifies the process significantly. For details on `conda` usage, please refer to official documentation or search online.


# 3. Data Preparation

Refer to the `DataProcessProject` folder for data preparation. Read its `README file` and follow the experimental steps. Proceed to the next step (reproduction) only after successfully completing the experiment.


# 4. Model Reproduction

The main model reproduction is divided into two parts:
Part 1: Reproduction of time-series foundation models (`TS-LibProejct folder`)
Part 2: Reproduction of time-series large models (`LLMProject folder`)
Detailed reproduction instructions are available in the README.md file within each corresponding folder.
