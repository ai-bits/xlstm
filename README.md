# xLSTM
xLSTM inference first working examples<br>
[Model card](https://huggingface.co/NX-AI/xLSTM-7b) with 3 lines of `pip` for a Python 3.11 environment<br>
20250215 2230 This is the quickest and dirtiest example imaginable to finally see working inference from the xLSTM-7b model.
If you want to get your fingers dirty you'll find an example that works on the CPU, but I guess you need more than 32GB RAM.

Be patient, the sample out (one chunk, no streaming) took 5 minutes on my Xeon CPU, but 64GB RAM are easier to find than >32GB VRAM on one card.<br>
Tried on my 2 x 20GB RTX 4000 Ada, but couldn't get it to work so far.
