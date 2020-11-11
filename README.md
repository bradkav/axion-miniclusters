# axion-miniclusters

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4006128.svg)](https://doi.org/10.5281/zenodo.4006128) ![GitHub](https://img.shields.io/badge/miniclusters-perturbed-green) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Code and results for the disruption of axion miniclusters (AMCs) in the Milky Way, as well as radio signals from encounters between AMCs and neutrons stars.*

The key parts of the computation are:
* Calculation of the perturbations to individual AMCs due to stellar encounters ([`code/Distribution_PL.ipynb`](code/Distribution_PL.ipynb) and [`code/Distribution_NFW.ipynb`](code/Distribution_NFW.ipynb)) 
* Monte Carlo simulations of the disruption of individual AMCs orbiting in the Milky Way ([`code/MC_script_ecc.py`](code/MC_script_ecc.py))  
* Processing and calculation of the distributions of AMC properties from the Monte Carlo results ([`code/prepare_distributions.py`](code/prepare_distributions.py))  
* Sampling of properties of individual encounters between AMCs and neutrons stars in the Milky Way ([`code/simulate_signal.py`](code/simulate_signal.py))

Scripts for generating plots from the results are in [`code/plotting/`](code/plotting). The first thing to do is to edit [`code/dirs.py`](code/dirs.py) so that the directory variables point to the right place.

The raw Monte Carlo results are archived online at https://doi.org/10.6084/m9.figshare.13224386.v1. Edit the file [`code/dirs.py`](code/dirs.py) to specify the directory where these Monte Carlo results are located (though the raw files are only needed if you want to re-calculate the AMC distributions).

Full samples of radio signal events due to AMC-NS encounters are archived online at https://doi.org/10.6084/m9.figshare.13204856.v1. These should be placed in the [`data/`](data/) folder. The full samples contain 10^7 events each; without these the plotting scripts will use instead the 'short' sample files provided, each of which is 10^5 events.

### Re-interpreting the results

If you would like to re-compute the AMC distributions assuming a different axion mass or initial mass function, the mass function can be specified in [`code/mass_function.py`](code/mass_function.py). For a power-law mass function, you can simply edit the command
```python
AMC_MF = mass_function.PowerLawMassFunction(m_a, gamma)
```
in the scripts [`code/prepare_distributions.py`](code/prepare_distributions.py) and [`code/simulate_signal.py`](code/simulate_signal.py). The new AMC distributions and signal samples can then be computed with 
```bash
./GetDistributions.sh
```
and 
```bash
./GetSignals.sh
```
Note that these commands will replace the distribution and signal sample files which are already in the repository. You can specify the `IDstr` variable hard-coded in [`code/prepare_distributions.py`](code/prepare_distributions.py) and [`code/simulate_signal.py`](code/simulate_signal.py) to add a file suffix so that the old files are not overwritten.

### Citation

If you use the code or the associated data, please cite this repository and its DOI: [10.5281/zenodo.4006128](https://doi.org/10.5281/zenodo.4006128).

Please also cite the two associated papers:
> B. J. Kavanagh, T. D. P. Edwards, L. Visinelli & C. Weniger (2020), "Stellar Disruption of Axion Miniclusters in the Milky Way".

>T. D. P. Edwards, B. J. Kavanagh, L. Visinelli & C. Weniger (2020), "Transient Radio Signatures from Neutron Star Encounters with QCD Axion Miniclusters".

### Authors & Contact

Written and maintained by Bradley J. Kavanagh, Thomas D. P. Edwards and Luca Visinelli.

This repository contains code which is being actively used for research, so in places it may not be 100% clear. If you have any questions whatsoever, or if the code behaves in an unexpected way, please do not hesitate to contact the authors (e.g. at bradkav@gmail.com).

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
