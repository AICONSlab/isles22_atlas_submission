# AICONSlab - ISLES22 ATLAS

This repository represents our lab's submission to the [ATLAS 2.0 segmentation challenge](https://atlas.grand-challenge.org/) (incorporated into ISLES at MICCAI 2022) for stroke lestion segmentation based on T1-weighted (T1w) imaging.

In our solution, we used a patch-based 3D residual U-Net to segment stroke lesions using two input channels: one being the original T1w image, and another being the same image flipped symmetrically across the left-right axis. Given the often-unilateral nature of stroke lesions, incorporation of a symmetrically flipped channel – such that the model sees a patch from roughly the same region of each hemisphere – helped improve sensitivity and overall performance significantly.

This repository was based on the [ATLAS 2.0 sample Docker](https://github.com/npnl/atlas_sample_docker). The text from the original README.md is maintained below for completeness:

This repository serves as a template for your to produce a Docker container with your model.
Your model should be trained and loadable at this stage.  
There are three important files: for you to modify:
- `requirements.txt` - Python dependencies for your model.  
  Python packages specified in `requirements.txt` will be installed in the container's Python environment when it is built.
- `process.py` - Modify the section to load and call your model.  
  Load your model and use it to make predictions on the input.
- `Dockerfile` - Add the files needed to run your model (model weights, code, etc.)

Once complete, you can run `build.sh` to build the container, and `export.sh` to package it for upload.
The original source code for the algorithm container was generated with evalutils version 0.3.1.
