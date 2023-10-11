# A Data-driven Study of Auditory Iconicity

![alt text](https://github.com/Andrea-de-Varda/iconicity-datadriven/blob/main/old/ico.png)

## Motivation

Auditory iconic words such as onomatopoetic words (_crack_, _bubble_, _whisper_) imitate in their phonological form the sound of their referents. This study aims to test the pervasiveness of iconicity in English through a data-driven procedure. The analyses are based on the representation of word and natural sounds into a shared vector space through:

1. a short-time Fourier transform
2. a convolutional neural network trained to classify sounds
3. a network trained on speech recognition

We employ the obtained vector representations to measure their objective sound resemblance. Their similarity metric is employed to (a) assess the pervasiveness of iconicity in the auditory vocabulary, and (b) predict human iconicity ratings.

## Materials and scripts

### Python scripts
The python scripts are available in the ``python`` folder.
- The scripts for generating the sound embeddings from the VGGish network are in the ``VGGish_iconicity.py`` file
- The ones based on the SpeechVGG architecture are in ``SpeechVGG_iconicity.py``. This script assumes that you have cloned the [SpeechVGG](https://github.com/bepierre/SpeechVGG) repository in your working directory.
- The ones based on the sound spectra are in the ``spectrogram_iconicity.py`` file
- The scripts for correlating them with human ratings and extracting the various covariates are in ``correlational_analyses.py``. This script assumes that you have downloaded the necessary resources, and you provide the correct path to locate them in your computer (see below for the links and details).

### R scripts
The R scripts are available in the ``python`` folder.
- The R file ``lmer_models.R`` analyzes the similarity of word sounds and natural sounds, testing whether matching pairs have higher similarity.
- The R file ``predict_icoratings.R`` filts a series of linear regression models to predict human iconicity judgements on the basis of the similarity between word and natural sound vectors.

To successfully run the scripts it is important that you have a folder ``MALD1_rw`` (as it can be downloaded [here](http://mald.artsrn.ualberta.ca/)). 

### Data and pre-computed metrics
Unfortunately, due to copyright restrictions we are unable to release the raw sound data we employed in our analyses, as the different sound files are protected by different copyright restrictions. We release however aggregated data from the two networks (VGGish and SpeechVGG), and the final inconicity measurements that we obtained with our procedure. 

A dataframe reporting the iconicity scores obtained with our experiments, as well as the human ratings employed for the validation (Winter et al., [2017](https://www.researchgate.net/publication/318364562_Which_words_are_most_iconic_Iconicity_in_English_sensory_words), [2022](https://osf.io/qvw6u/)), are reported in the folder **additional-data** under ``icodf_covariates.csv`` 


## Pre-existing resources
Our analyses are based on two pre-trained convolutional neural networks, that are publicly available :

- [SpeechVGG](https://github.com/bepierre/SpeechVGG)
- [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) (although we employed the tf-hub release, see ``VGGish_iconicity.py``)

> [!NOTE]
> In our prediction of human ratings, we make use of several pre-existing datasets. Remember to provide the correct path to the norms in the Python scripts, and to cite the relevant articles if you use these norms.

- Iconicity ratings, dataset 1 released by [Winter et al., 2017](http://pure-oai.bham.ac.uk/ws/files/38406823/sensory_iconicity_revisions_final.pdf)
- Iconicity ratings, dataset 2 released by [Winter et al., 2022](https://link.springer.com/article/10.3758/s13428-023-02112-6)
- Sensory norms released by [Lynott et al. 2019](https://link.springer.com/article/10.3758/s13428-019-01316-z)
- Concreteness estimates released by [Brysbaert et al., 2014](https://link.springer.com/article/10.3758/s13428-013-0403-5)
- Frequency norms released by [Brysbaert et al., 2012](https://link.springer.com/article/10.3758/s13428-012-0190-4)
- Age-of-Acquisition norms released by [Kuperman et al., 2012](https://link.springer.com/article/10.3758/s13428-012-0210-4)

