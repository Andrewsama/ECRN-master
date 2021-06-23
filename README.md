# ECRN
Enhanced contrastive representaion on graph (Ma *et al.*):


## Overview
Here we provide an implementation of Enhanced contrastive representaion on graph (ECRN) in PyTorch, along with a example (on the CoauthorPhysics dataset(ms_academic_phy.npz)). The repository is organised as follows:
- `data/` contains the necessary dataset files for CoauthorPhysics;
- `models/` contains the implementation of the ECRN pipeline (`ecrn.py`) and our logistic regressor (`logreg.py`);
- `layers/` contains the implementation of a GCN layer (`gcn.py`), the averaging readout (`readout.py`), and the bilinear discriminator (`discriminator.py`);
- `utils/` contains the necessary processing subroutines (`process.py`).

Finally, `execute.py` puts all of the above together and may be used to execute a full training run on CoauthorPhysics.
