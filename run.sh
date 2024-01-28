#!/bin/bash
conda activate x-decoder
python ffmpeg_seem.py
cd Segment-Everything-Everywhere-All-At-Once/
python demo/seem/app.py
conda deactivate
conda activate llama-gradio
cd ../
cd llama-2-7b-chat/
python app.py
cd ../