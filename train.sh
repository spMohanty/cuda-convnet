python convnet.py --data-path /plantvillage-extra/cuda-convnet-batch-data-with-none \
  --layer-params ./example-layers/cuda_layer_params.cfg \
  --layer-def ./example-layers/cuda_layer.cfg \
  --train-range 0-60 \
  --test-range 61-105 \
  --save-path /plantvillage-extra/cuda-convnet-snapshots-with-none \
  --test-freq 50 \
  --epochs 200 \
  --data-provider raw-cropped \
  --image-size 256 \
  --mini 100 \
  --binary-save 1 \
  --print-entire-array 0 \
  --crop-border 0
