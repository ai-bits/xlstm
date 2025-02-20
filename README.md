# xLSTM
xLSTM inference first working examples<br>
[Model card](https://huggingface.co/NX-AI/xLSTM-7b) with 3 lines of `pip` for a Python 3.11 environment ON LINUX!<br>
`pip install xlstm`<br>
`pip install mlstm_kernels #all 'Requirement already satisfied' #Included in the xlstm code and redundant??`<br>
`pip install git+https://github.com/NX-AI/transformers.git@integrate_xlstm#egg=transformers`

Currently only on Linux. Tested on Ubuntu 24.04 - to be specific.

20250215 2230 This is the quickest and dirtiest example imaginable to finally see working inference from the xLSTM-7b model.
If you want to get your fingers dirty, `examples/hello-torch-cpu.py` works on the CPU, but I guess you need more than 32GB RAM. More than available on typical machines.

Be patient, the sample out (one chunk, no streaming) took 5 minutes on my Xeon CPU, but 64GB RAM are easier to find than >32GB VRAM on one card.

20250217 Got multi-GPU (dual) inference to work on my 2 x 20GB RTX 4000 Ada 256GB Xeon machine, but it needs an xLSTM code change.<br>
20250220 If you want to do inference with such a config you need a Python environment like that:<br>
**As of 1530 both reinstalls (above and below) run into a PyTorch (2.6?) problem that I saw somewhere in the issues. Need to check.**

`git clone https://github.com/ai-bits/xlstm-fork`
`cd xlstm-fork`
`conda create -n xlstm python=3.11`
`conda activate xlstm`
`pip install -e . #to get the code change for the env from the forked and updated repo`
`pip install mlstm_kernels #all 'Requirement already satisfied' #Included in the xlstm code and redundant??`<br>
`pip install git+https://github.com/NX-AI/transformers.git@integrate_xlstm#egg=transformers`
