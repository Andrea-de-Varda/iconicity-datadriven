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
Unfortunately, due to copyright restrictions we are unable to release the raw sound data we employed in our analyses, as the different sound files are protected by different copyright restrictions. We release however pre-processed data from the two networks (VGGish and SpeechVGG), and the final inconicity measurements that we obtained with our procedure. 

The pre-processed data from the two networks is available in the folder ``additional_data``. The files of interest are ``word_sounds_VGGish``, ``natural_sounds_VGGish``, ``word_sounds_SpeechVGG``, ``natural_sounds_SpeechVGG``. They are Python dictionaries that can be easily loaded as:

```python
import pickle

with open("natural_sounds_VGGish", 'rb') as handle:
    natural_sounds_VGGish = pickle.load(handle)
```

They are organized as a dictionary where the key indicates the label for which the sounds where extracted, and the value is a list of numpy arrays. Each arrays corresponds to the sound representation of one specific instance of a natural sound, as scaped from Freesound. Below is an example of how the data are structured:

```python
{'man': [array([ 0.24334893, -0.21349812,  0.13270998, -0.5002561 , -0.18519397,
         -0.43334043, -0.24831858, -0.3443597 , -0.26181513,  0.10784349,
         -0.47075054, -0.5784661 , -0.47967815, -0.1911524 , -0.06631487,
          [...]
         -0.2263436 , -0.5942359 ,  0.15638947, -0.839978  ,  0.39491096,
          0.40798616, -0.15921661, -0.06044232, -0.04705894, -0.10617545,
          0.15600096, -0.00818225, -0.38729894], dtype=float32),
  array([ 1.92212760e-02, -2.53211856e-02,  8.34971666e-04, -1.49130642e-01,
          1.22893751e-02, -1.84382632e-01, -3.79019916e-01,  1.32700175e-01,
         [...]
         -4.32686716e-01,  1.83223188e-01, -6.68203831e-01,  2.66942441e-01,
          6.02958798e-02, -8.18358138e-02, -7.08016753e-02,  8.71086121e-03,
          5.13320446e-01,  1.70548424e-01, -2.70681679e-02, -4.12309706e-01],
        dtype=float32), [...]}
```

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

