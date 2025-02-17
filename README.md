# xLSTM
xLSTM inference first working examples<br>
[Model card](https://huggingface.co/NX-AI/xLSTM-7b) with 3 lines of `pip` for a Python 3.11 environment<br>
`pip install xlstm`<br>
`pip install mlstm_kernels`<br>
`pip install git+https://github.com/NX-AI/transformers.git@integrate_xlstm#egg=transformers`

Currently only on Linux. Tested on Ubuntu 24.04 - to be specific.

20250215 2230 This is the quickest and dirtiest example imaginable to finally see working inference from the xLSTM-7b model.
If you want to get your fingers dirty, `examples/hello-torch-cpu.py` works on the CPU, but I guess you need more than 32GB RAM. More than available on typical machines.

Be patient, the sample out (one chunk, no streaming) took 5 minutes on my Xeon CPU, but 64GB RAM are easier to find than >32GB VRAM on one card.

20250217  Got inference to work on my 2 x 20GB RTX 4000 Ada 256GB Xeon machine, but it needs an xLSTM code change first and I have no idea when this will arrive in the wild.
