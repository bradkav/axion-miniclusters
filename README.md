# axion-miniclusters

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4006128.svg)](https://doi.org/10.5281/zenodo.4006128) [![arXiv](https://img.shields.io/badge/arXiv-2011.05377-B31B1B)](http://arxiv.org/abs/2011.05377) [![arXiv](https://img.shields.io/badge/arXiv-2011.05378-B31B1B)](http://arxiv.org/abs/2011.05378) ![GitHub](https://img.shields.io/badge/miniclusters-perturbed-green) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Code and results for the disruption of axion miniclusters (AMCs) in the Milky Way, as well as radio signals from encounters between AMCs and neutrons stars.*

The key parts of the computation are:
* Calculation of the perturbations to individual AMCs due to stellar encounters ([`code/Distribution_PL.ipynb`](code/Distribution_PL.ipynb) and [`code/Distribution_NFW.ipynb`](code/Distribution_NFW.ipynb)) 
* Monte Carlo simulations of the disruption of individual AMCs orbiting in the Milky Way ([`code/MonteCarlo.py`](code/MonteCarlo.py))  
* Processing and calculation of the distributions of AMC properties from the Monte Carlo results ([`code/prepare_distributions.py`](code/prepare_distributions.py))  
* Sampling of properties of individual encounters between AMCs and neutrons stars in the Milky Way ([`code/simulate_signal.py`](code/simulate_signal.py))

Scripts for generating plots from the results are in [`code/plotting/`](code/plotting). The first thing to do is to edit [`code/dirs.py`](code/dirs.py) so that the directory variables point to the right place.

**An example script showing how to run the pipeline 'end-to-end' is given in [`code/RunPipeline.py`](code/RunPipeline.py).**

The raw Monte Carlo results are archived online at https://doi.org/10.6084/m9.figshare.13224386.v1. Edit the file [`code/dirs.py`](code/dirs.py) to specify the directory where these Monte Carlo results are located (though the raw files are only needed if you want to re-calculate the AMC distributions).

Full samples of radio signal events due to AMC-NS encounters are archived online at https://doi.org/10.6084/m9.figshare.13204856.v1. These should be placed in the [`data/`](data/) folder. The full samples contain 10^7 events each; without these the plotting scripts will use instead the 'short' sample files provided, each of which is 10^5 events.

### The pipeline

After a substantial update, you now specify most of the parameters as function arguments. These include the axion mass, density profile, mass function and galaxy. 

In particular:
	- `profile`: Internal AMC density profile. Options: `PL`, `NFW`
	- `mass_function_ID`: String identifying the mass function you want to use:
		* `powerlaw` - standard power-law mass function, with log-slope gamma = -0.7
		* `delta_a` - delta function mass function, centred on the average mass of a `powerlaw` function
		* `delta_c` - delta function, centred on the characteristic AMC mass
		* `delta_p` - delta function, centred on the peak of the AMC mass function at MRE. 
	- `galaxyID`: String identifying the galaxy to be used. Options: `MW`, `M31`.

An example of the pipeline can be found in [`code/RunPipeline.py`](code/RunPipeline.py). As always, you should edit [`code/dirs.py`](code/dirs.py) so that the directory variables point to the right place.


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
