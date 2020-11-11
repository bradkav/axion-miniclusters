# axion-miniclusters

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4006128.svg)](https://doi.org/10.5281/zenodo.4006128) [![LICENSE](https://img.shields.io/github/license/bradkav/axion-miniclusters)](https://github.com/bradkav/axion-miniclusters/blob/master/LICENSE) ![MINICLUSTER-BADGE](https://img.shields.io/badge/miniclusters-perturbed-blueviolet)

Code and results for the disruption of axion miniclusters (AMCs) in the Milky Way, as well as radio signals from encounters between AMCs and neutrons stars.  
The key parts of the computation are:
* Calculation of the perturbations to individual AMCs due to stellar encounters ([`code/Distribution_PL.ipynb`](code/Distribution_PL.ipynb) and [`code/Distribution_NFW.ipynb`](code/Distribution_NFW.ipynb)) 
* Monte Carlo simulations of the disruption of individual AMCs orbiting in the Milky Way ([`code/MC_script_ecc.py`](code/MC_script_ecc.py))  
* Processing and calculation of the distributions of AMC properties from the Monte Carlo results ([`code/prepare_distributions.py`](code/prepare_distributions.py))  
* Sampling of properties of individual encounters between AMCs and neutrons stars in the Milky Way ([`code/simulate_signal.py`](code/simulate_signal.py))

#### To-do

* Add link to signals - https://doi.org/10.6084/m9.figshare.13204856.v1  
* Check .gitignore file


#### Authors

Written and maintained by Bradley J. Kavanagh, Thomas D. P. Edwards and Luca Visinelli.
