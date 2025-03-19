# xLSTM-test
xLSTM inference first working examples ON LINUX (Tested on Ubuntu 24.04 - to be specific.)<br>
[Model card and files download](https://huggingface.co/NX-AI/xLSTM-7b)

20250319 As of today morning I can confirm that xLSTM inference has no longer a problem with the latest PyTorch (2.6.0), so we are back to a simpler env install.<br>
Sadly, the fix for multi-GPU must still be done manually if you do not own a very expensive 40GB+ GPU (or sth in the cloud / Google Colab) because of another env install bug. (see below)

**If you want to do inference you need a Python environment ON LINUX OR WSL (!) like that:**<br>
`conda create -n xlstm python=3.11` #conda remove -n xlstm --all #JIC you need to get rid of the old env<br>
`conda activate xlstm`<br>
`pip install xlstm` #alternatively:<br>
#`pip install -e .` #editable env seems still a problem, so you must do the 2-line code change for multi-GPU manually!!!<br>
`pip install mlstm_kernels` #all 'Requirement already satisfied' #Included in xlstm above and redundant??<br>
`pip install git+https://github.com/NX-AI/transformers.git@integrate_xlstm#egg=transformers`<br>
`pip install accelerate>=0.26.0` #forgotten requirement<br>

20250223 2325 I'm pretty underwhelmed by o3-mini-high's optimizations in `hello-torch-gpu2-ui2.py`.<br>
Output a little faster, but still slows down with the length of the text.<br>
And after optimizing speed a bit, chatGPT messed up the output in suggested variants. (jerky 50 char batches; word-by-word, but no spaces; no paragraphs;...)

20250222 2320 `hello-torch-gpu2-ui.py` WORKING! No CUDA crash. Leave `hello-torch-gpu2-ui1.py` for reference.

20250222 1015 With the help of OpenAI's 4o I got steaming almost working in `hello-torch-gpu2-ui1.py` - until CUDA crashes. (guess out of memory)<br>
Mind: You need >32GB VRAM to run this. I hope someone from NX-AI will react before I decease to set up an inference server. (I am not aware of one.)

20250215 2230 This is the quickest and dirtiest example imaginable to finally see working inference from the xLSTM-7b model.
If you want to get your fingers dirty, `examples/hello-torch-cpu.py` works on the CPU **on (WSL) Ubuntu**, but I guess you need more than 32GB RAM. More than available on typical machines.<br>
(I thought I had it working on Windows too, but after the nth reinstall (Torch cpuonly) it still throws "No module named 'triton'". And I don't want to get into the weeds of a Docker Triton install or so.)

Be patient, the 500 token sample out (one chunk, no streaming) took 5 minutes on my Xeon CPU, but >32GB RAM are easier to find than >32GB VRAM on a single GPU.

20250217 **Got multi-GPU (dual) inference to work on my 2 x 20GB RTX 4000 Ada 256GB Xeon machine, but `pip install -e .` does not work as expected, multi-GPU needs an xLSTM code change ~~and a Torch version 2.6.0 v 2.5.1 problem cropped up~~.**<br>
Inference on GPU on Windows throws "No module named 'triton'", though  I an not aware of Triton funcs explicitly used.<br>
**WSL (Ubuntu) turns out to be an overlooked CUDA gem on Windows! Conveniently try it instead of VMs (GPU problem) or booting Linux!**

20250319 **Multi-GPU:**<br>

#`git clone https://github.com/ai-bits/xlstm-fork` #Currently not needed, because<br>
#first the `pip install -e .` problem must be worked out to get the multi-GPU fix with the repo.<br>
#`cd xlstm-fork`<br>

The code changes for multi-GPU should be in the [AI-bits/xlstm-fork repo](https://github.com/ai-bits/xlstm-fork) in `xlstm/xlstm_large/model.py` lines 504 and 508.<br>
As there is a problem with `pip install -e .`, you must change the two lines manually **after** env installation in<br>
`anaconda3/envs/xlstm/lib/python3.11/site-packages/xlstm/xlstm_large/model.py`

`  x = x + x_mlstm.to(x.device) #x = x + x_mlstm #gue22 20250217`<br>
`  x = x + x_ffn.to(x.device) #x = x + x_ffn #gue22 20250217`<br>
Looks simple once you found out. ;-)