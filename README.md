## About
**AutoGLIM** is an automated system for the segmentation of Golgi ministacks in cells imaged by fluorescent microscopy. It also leverages previously published techniques to find the localisation of a specific protein in the ministack. This method of determining where a protein exists in the Golgi was previously developed by Tie *et al.*, 2017. This program fully automates the pipeline and, given a three-channel flourescent microscopy image, performs automated background subtraction, segmentation, chromatic abberation correction and localisation quotient (LQ) calculation.

Segmentation of cells is performed using a trained Segformer (credit to @lucidrains for open-source implementation), and automated background subtraction done using dual annealing of three channels with the objective of maximising the number of ROIs.

## Paper
Publication: https://elifesciences.org/reviewed-preprints/98582

## Installation
```sh
git clone https://github.com/Chokyotager/AutoGLIM.git
cd AutoGLIM
```

```sh
conda env create -f environment.yml
conda activate autoglim
```

There is also an explicit link file in requirements.txt for all Conda packages.

AutoGLIM has been tested on an Ubuntu 22.04.1 LTS (GNU/Linux 5.15.0-58-generic x86_64) system, Windows 11 and macOS Sequoia. It theoretically should work for any environment so long as all package requirements are fulfilled.

Installation should take under ten minutes in most cases.

## Usage
Simply run `process.py` after installation. It requires the input of beads_images and a folder. `wrapper.py` is a convenience script that can be used to run batches of images.

To run on one's own images, they have to be pre-processed into the three channels as shown in `images/AVG_Composite.tif` and beads images of the same format found in `beads_images/`.

Final results are output in `results/` with `filtered_metrics.tsv` showing individual LQs of segmentations. These are typically averaged to calculate the final LQ of the protein. `.ijroi.zip` files are also produced which can be opened in ImageJ.

## Limitations
The training of the Segformer model was done on fluorescent microscope and Airyscan microscope images from the School of Biological Sciences at Nanyang Technological University with images from as many people as reasonably possible. As such, the model may not be fully generalisable to other instruments/conditions. If you would like to contribute training data, please contact A/Prof. Lu Lei and CC Hilbert Lam.

## Maintenance
This project is maintained by Hilbert Lam.

## License
License details can be found in the LICENSE file.

## Citation
```
@article{tie_quantitative_2024,
	title = {Quantitative intra-Golgi transport and organization data suggest the stable compartment nature of the Golgi},
	issn = {2522-5839},
	url = {https://elifesciences.org/reviewed-preprints/98582},
	doi = {10.7554/eLife.98582.1},
	language = {en},
	urldate = {2024-06-03},
	journal = {eLife},
	author = {Tie, Hieng Chiong and Wang, Haiyun and Mahajan, Divyanshu and Lam, Hilbert Yuen In and Sun, Xiuping and Chen, Bing and Mu, Yuguang and Lu, Lei},
	month = jun,
	year = {2024},
}
```
