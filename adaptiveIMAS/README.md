# Best affine invariant performances with the speed of CNNs

This repository is linked to the paper [CNN-assisted coverings in the Space of Tilts](https://rdguez-mariano.github.io/pages/adimas). Viewpoint invariance is attained through Image Matching by Affine Simulation (IMAS) methods. The source code implements Adaptive IMAS methods from the aforementioned paper.

## Prerequisites

In order to quickly start using Adaptive IMAS we advise you to install [anaconda](https://www.anaconda.com/distribution/) and follow this guide.

##### Creating a conda environment for Adaptive IMAS

```bash
conda create --name adimas python=3.5.4

source activate adimas

pip install --upgrade pip
pip install -r requirements.txt

sudo apt-get install -y libconfig++-dev
sudo apt-get install -y liblapack-dev
```

##### Setup

Download the repository [fast_imas_ipol](https://github.com/rdguez-mariano/fast_imas_IPOL) and open a terminal in that folder. Then do,

```bash
cd adaptiveIMAS
bash setup.sh
```

##### Possible install errors

If AttributeError: module 'cv2.cv2' has no attribute 'xfeatures2d' reinstall opencv-contrib

```bash
pip uninstall opencv-contrib-python
pip install opencv-python==3.4.2.16
```

#### Uninstall the adimas environment

If you want to remove this project from your computer just do:

```bash
conda deactivate
conda-env remove -n adimas
rm -R /path/to/adimas
```

## Using Adaptive IMAS

The code bellow will launch:

- Affine-RootSIFT
- Adaptive ARootSIFT (as presented in [this paper](https://rdguez-mariano.github.io/pages/adimas))
- Greedy ARootSIFT (as presented in [this paper](https://rdguez-mariano.github.io/pages/adimas))
- Adaptive ARootSIFT with an added dilation procedure.
- Greedy ARootSIFT with affine simulations predicted from Affnet applied on SIFT patches (instead of Hessaff patches).


```python
import cv2
from libLocalDesc import *
from library import *

img1 = cv2.cvtColor(cv2.imread('/path/to/image1/img1.png'),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('/path/to/image2/img2.png'),cv2.COLOR_BGR2GRAY)

# The Affine-RootSIFT method (25 simulations)
matches = IMAScaller(img1,img2, desc = 11, Visual=True, covering=1.7, GFilter='IntCode-4')

# The Adaptive ARootSIFT method presented in the paper
matches = IMAScaller(img1,img2, desc = 11, Visual=True, covering=1.7, GFilter='IntCode-4',AdOPT = 'FixedTilts')

# The Greedy ARootSIFT method presented in the paper
matches = IMAScaller(img1,img2, desc = 11, Visual=True, covering=1.7, GFilter='IntCode-4',AdOPT = 'Greedy')

# Adding a simple dilation procedure to Adaptive ARootSIFT
matches = IMAScaller(img1,img2, desc = 11, Visual=True, covering=1.7, GFilter='IntCode-4',AdOPT = 'FixedTilts', dilate=True)

# Changing Affnet predictions to be applied on SIFT like patches instead of HessAff patches.
matches = IMAScaller(img1,img2, desc = 11, Visual=True, covering=1.7, GFilter='IntCode-4',AdOPT = 'Greedy', Detector='SIFT')
```

## Authors

* **Mariano Rodríguez** - [web page](https://rdguez-mariano.github.io/)
* **Gabriele Facciolo**
* **Rafael Grompone Von Gioi**
* **Pablo Musé**
* **Julie Delon** - [web page](https://delon.wp.imt.fr/)
* **Jean-Michel Morel** - [web page](https://sites.google.com/site/jeanmichelmorelcmlaenscachan/)


## Acknowledgments

##### This project

* calls [fast_imas_ipol](https://github.com/rdguez-mariano/fast_imas_IPOL). Copyright (c) 2018, Mariano Rodríguez, Julie Delon and Jean-Michel Morel, distributed under the BSD 2-Clause "Simplified" License.
* calls Affnet. Copyright (c) 2018 Dmytro Mishkin / See its [source code](https://github.com/ducha-aiki/affnet) distributed under the permissive MIT License.

## Github repository

<https://github.com/rdguez-mariano/fast_imas_IPOL/tree/master/adaptiveIMAS>
