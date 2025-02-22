# xLSTM
xLSTM inference first working examples<br>
[Model card](https://huggingface.co/NX-AI/xLSTM-7b) with 3 lines of `pip` for a Python 3.11 environment ON LINUX!<br>
`pip install xlstm`<br>
`pip install mlstm_kernels #all 'Requirement already satisfied' #Included in the xlstm code and redundant??`<br>
`pip install git+https://github.com/NX-AI/transformers.git@integrate_xlstm#egg=transformers`

Tested on Ubuntu 24.04 - to be specific.<br>
If you run into a Torch 2.6.0 or a multi-GPU problem, take the kludge install below!

20250222 2320 `hello-torch-gpu2-ui.py` WORKING! No CUDA crash. Leave `hello-torch-gpu2-ui1.py` for reference.

20250222 1015 With the help of OpenAI's 4o I got steaming almost working in `hello-torch-gpu2-ui1.py` - until CUDA crashes. (guess out of memory)<br>
Mind: You need >32GB VRAM to run this. I hope someone from NX-AI will react before I decease to set up an inference server. (I am not aware of one.)

20250215 2230 This is the quickest and dirtiest example imaginable to finally see working inference from the xLSTM-7b model.
If you want to get your fingers dirty, `examples/hello-torch-cpu.py` works on the CPU, but I guess you need more than 32GB RAM. More than available on typical machines.

Be patient, the sample out (one chunk, no streaming) took 5 minutes (for 1k tokens) on my Xeon CPU, but 64GB RAM are easier to find than >32GB VRAM on one card.

20250217 Got multi-GPU (dual) inference to work on my 2 x 20GB RTX 4000 Ada 256GB Xeon machine, but it needs an xLSTM code change and a Torch 2.6.0 problem cropped up.<br>
20250220 If you want to do inference with such a config you need a Python environment like that:<br>

`git clone https://github.com/ai-bits/xlstm-fork`<br>
`cd xlstm-fork`<br>
`conda create -n xlstm python=3.11 #conda remove -n xlstm --all #get rid of old env`<br>
`conda activate xlstm`<br>
`pip install -e . #to get the code change for the env from the forked and updated repo`<br>
`pip install mlstm_kernels #all 'Requirement already satisfied' #Included in the xlstm code and redundant??`<br>
`pip install git+https://github.com/NX-AI/transformers.git@integrate_xlstm#egg=transformers`<br>
`pip install accelerate>=0.26.0 #forgotten requirement`<br>
`#kludge for Torch 2.6.0 incompatibility until the code is fixed:`<br>
`pip uninstall torch ##torchvision torchaudio #only torch installed / needed`<br>
`pip install torch==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 #Torch INCL! CUDA`<br>
`export CUDA_HOME=/home/gy/anaconda3/pkgs #and add to .bashrc`<br>

If you are curious re the code changes, have a look in the [AI-bits/xlstm-fork repo](https://github.com/ai-bits/xlstm-fork) in `xlstm/xlstm_large/model.py` lines 504 and 508.<br>
Looks simple once you found it out. ;-)
