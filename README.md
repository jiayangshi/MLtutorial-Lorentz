# Machine Learning Tutorial for Tomography

This repository contains tutorials that cover different topics related to Machine Learning applications in tomography. The tutorials have been designed for the [Integrating Acquisition and AI in Tomography workshop](https://www.lorentzcenter.nl/integrating-acquisition-and-ai-in-tomography.html) at the Lorentz Center.

## Tutorials Covered

1. **Post-processing for CT Images** ([MLtutorial_postproc.ipynb](MLtutorial_postproc.ipynb))
2. **Deep Image Prior for CT Reconstruction** ([MLtutorial_dip.ipynb](MLtutorial_dip.ipynb))
3. **Implicit Neural Representations for CT Reconstruction** ([MLtutorial_inr.ipynb](MLtutorial_inr.ipynb))

## Dataset

- **Training Data for Post-processing**: `recon_low.npy`, `recon_high.npy`
- **Phantom for Training Data Generation**: `train_phantom.h5`
- **Object for Reconstruction Tasks**: `lung_small.png`
  
Additional utility:
- `generate_projs.py`: Python script to generate projections for the phantom.

## Requirements

- **For local machines**: Please install the packages listed in the `environment.yml` file. You can create a new conda environment with the specified packages using:
```bash
conda env create -f environment.yml
```

- **For Google Colab**: The necessary steps to install the required packages are already included in the notebooks.


## Workshop Details

For more details about the workshop and additional resources, please visit the [official workshop website](https://www.lorentzcenter.nl/integrating-acquisition-and-ai-in-tomography.html).