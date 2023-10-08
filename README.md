# iconicity-datadriven

![alt text](https://github.com/Andrea-de-Varda/iconicity-datadriven/blob/main/old/ico.png)


Materials for deriving data-driven iconicity measurements in the auditory modality. 

- The scripts for generating the sound embeddings from the VGGish network are in the ``VGGish_iconicity_sound.py`` file
- The ones based on the SpeechVGG architecture are in ``SpeechVGG_iconicity_sound.py``
- The scripts for correlating them with human ratings are in ``Correlations_iconicity_sounds.py``

A dataframe reporting the iconicity scores obtained with our experiments, as well as the human ratings employed for the validation (Winter et al., [2017](https://www.researchgate.net/publication/318364562_Which_words_are_most_iconic_Iconicity_in_English_sensory_words), [2022](https://osf.io/qvw6u/)), are reported in the folder **additional-data**.

Our analyses are based on two pre-trained convolutional neural networks, that are publicly available :

- [SpeechVGG](https://github.com/bepierre/SpeechVGG)
- [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) (although we employed the tf-hub release, see ``VGGish_iconicity_sound.py``)
