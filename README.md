# DatasetMake
Making dataset using DSO (Direct sparse odometry)

The project was modified on the basis of [DSO](https://github.com/JakobEngel/dso).

## How to run (for example)

bin/dso_dataset 
"files=/home/lsp/disk1/TUMmono/all_sequences/sequence_13/images.zip" "calib=/home/lsp/disk1/TUMmono/all_sequences/sequence_13/camera.txt" "gamma=/home/lsp/disk1/TUMmono/all_sequences/sequence_13/pcalib.txt" "vignette=/home/lsp/disk1/TUMmono/all_sequences/sequence_13/vignette.png" "filepath=data/TUMmono32_32/sequence_13" "nogui=1" "preset=0" "mode=0"

## NOTE

Line 609 in main_dso_pangolin.cpp is the "txt" text save path.

Line 242, 243 in HessianBlocks.cpp are the patch image save patch.

Please modify the path as needed.
