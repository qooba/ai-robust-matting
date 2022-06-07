#!/bin/bash



cd /workspace/BackgroundMattingV2
python inference_video.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --model-checkpoint "/src/model.pth" \
        --video-src "/src/C0283_1.mp4" \
        --video-bgr "/src/bgr_white2.png" \
        --output-dir "/src/output/" \
        --output-type com fgr pha err ref

cd /src/
