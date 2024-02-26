from os import chdir, listdir
chdir("path/to/your/wd")
import numpy as np
from numpy import dot    
from numpy.linalg import norm
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sps
from numpy.lib import stride_tricks
from numpy import inf
from pymer4.models import Lmer
import pandas as pd
import re
import glob
from researchpy import ttest, corr_pair
import pickle
from tqdm import tqdm
import librosa
from scipy.signal import stft
from libs.data_generator import log_standardize, pad_spec

def cosine(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def audio_to_input(fn, resample):
    x, sr = librosa.load(fn,sr=None) # automatically converts to mono
    x_16k = librosa.resample(x,sr,resample) # resample to 16KHz
    if len(x_16k) < resample:
        x_16k = librosa.util.fix_length(x_16k, size=resample)
    audio = x_16k
    if resample and len(audio) > resample:
        interest_range = [int(len(audio)/2)-(resample/2), int(len(audio)/2)+(resample/2)]
        audio = audio[int(interest_range[0]):int(interest_range[1])]
    f, t, seg_stft = stft(audio,
                        window='hamming',
                        nperseg=256,
                        noverlap=128)
    mag_spec = np.abs(seg_stft)
    spec_tmp = np.swapaxes(mag_spec, 0, 1)
    data_tmp = spec_tmp[..., np.newaxis]
    data_tmp[:,:,0] = log_standardize(data_tmp[:,:,0])
    data_tmp= np.delete(data_tmp, (128), axis=1)
    #print(data_tmp.shape)
    return data_tmp.flatten()

files = [re.sub("\.wav", "", f.lower()) for f in listdir("path/to/your/MALD1_rw")]
print(files)

filepath = "path/to/your/MALD1_rw"

sensory = pd.read_csv("path/to/your/sensory_norms.csv")
print(len(sensory))
sensory.columns
sensory["Word"] = sensory.Word.str.lower()
sensory_d = {}
for index, row in sensory.iterrows():
    sensory_d[row.Word] = row["Auditory.mean"]
    
candidates = []
for file in files:
    try:
        candidates.append([file, sensory_d[file]])
    except KeyError:
        pass
    
candidates = pd.DataFrame(candidates, columns=["word", "auditory"])
m  = np.mean(candidates.auditory)
sd = np.std(candidates.auditory)
c = candidates[candidates.auditory > m+1.5*sd]

spect = []
for name in c.word:
    file = name.upper()+".wav"
    ims = audio_to_input(filepath+'/'+file, resample=16200)
    spect.append(ims)
    print(ims.shape)

# importing natural sounds
noise_clips = glob.glob('/path/to/your/freesound-data/*.wav')
print(len(noise_clips))

spect_noise = []
for noise_clip in tqdm(noise_clips):
    ims = audio_to_input(noise_clip, resample = 16200)
    spect_noise.append(ims)

word_sounds = {name : f for name, f in zip(c.word,spect)}
print(len(word_sounds))

noises1 = [re.sub('/path/to/your/freesound-data/', "", s) for s in noise_clips]
noises = [re.search(r"([a-zA-Z]+)_\d+_\d+\.wav", filename).group(1) for filename in noises1]
print("Check =", len(noises) == len(spect_noise))

natural_sounds = []
for name, f in zip(noises, spect_noise):
    f[f == -inf] = np.mean([val for val in f if val != -inf])
    natural_sounds.append(f)

############################################################
# visualize with t-sne and compute within-class similarity #
############################################################

features_all = []
clabels = []
for wordvec in word_sounds.values():
    if wordvec.shape == (16384,):
        if np.isnan(wordvec).any():
            pass
        else:
            features_all.append(wordvec)
            clabels.append(0)
for soundvec in natural_sounds:
    if soundvec.shape == (16384,):
        if np.isnan(soundvec).any():
            pass
        else:
            features_all.append(soundvec)
            clabels.append(1)

from sklearn.manifold import TSNE
import matplotlib as pylab
pylab.rc('font', family='sans serif', size=12)
tsne = TSNE(n_components=2).fit_transform(np.array(features_all))

plt.figure(dpi=300)
plt.scatter(tsne[len(word_sounds.values()):,0], tsne[len(word_sounds.values()):,1], label='$\overrightarrow{s_n}$', color=[0,0.6,0.8,1], s=7)
plt.scatter(tsne[:len(word_sounds.values()),0], tsne[:len(word_sounds.values()),1], label='$\overrightarrow{s_w}$', color=[0,0.3,0.7,1], s=7)
plt.legend()

plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False,
    labelleft=False)
fig = plt.gcf()
fig.set_size_inches(7, 6)

#############################################################

my_check = ["yes_2_373.wav", "yes_6_141.wav", "yes_6_148.wav", "no_4_109.wav", "no_5_38.wav", "no_5_55.wav", "no_5_109.wav", "no_5_113.wav", "no_5_163.wav", "no_5_174.wav", "no_5_200.wav", "no_6_357.wav"]

filter_array = [noise not in my_check for noise in noises1]
noises = [noise for noise, keep in zip(noises, filter_array) if keep]
natural_sounds = [noise for noise, keep in zip(natural_sounds, filter_array) if keep]

data_dict = {}
for label, vec in zip(noises, natural_sounds):
    if label in data_dict.keys():
        data_dict[label].append(vec)
    else:
        data_dict[label] = [vec]
                
data_dict1 = {}
for k, v in data_dict.items():
    v = [vec for vec in v if vec.shape == (16384,)]
    data_dict1[k] = np.mean(v, axis=0)
    
wordsounds = set(word_sounds.keys())
natsounds  = set(data_dict.keys())
words_cleaned = set.intersection(natsounds, wordsounds)
len(words_cleaned) # 367

natural_sounds = {key:value for key, value in data_dict1.items() if key in words_cleaned}
word_sounds = {key:value for key, value in word_sounds.items() if key in words_cleaned}
    
with open("natural_sounds_spect", 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("natural_sounds_spect", 'rb') as handle:
    data_dict = pickle.load(handle)

with open("word_sounds_spect", 'wb') as handle:
    pickle.dump(word_sounds, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("word_sounds_spect", 'rb') as handle:
    word_sounds = pickle.load(handle)

icodict = {}
for word in word_sounds.keys():
    icon = cosine(data_dict1[word], word_sounds[word])
    icodict[word] = float(icon)
    
icodf = pd.DataFrame.from_dict(icodict.items())
icodf.columns=["w", "i"]
icodf = icodf.sort_values(by="i", ascending=False).dropna()
icodf.head(10)
icodf.tail(10)

icodf.to_csv("icodf_spectrum_cleaned.csv")

markedness_dict = {}
for word, vec in word_sounds.items():
    mark = np.mean([cosine(vec, vec1) for vec1 in word_sounds.values()])
    markedness_dict[word] = mark
    
with open("markedness_spectrum", 'wb') as handle:
    pickle.dump(markedness_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

df = []
for key, value in tqdm(data_dict.items()):
    value = [v for v in value if v.shape == (16384,)]
    for key1, value1 in word_sounds.items():
        for vec in value:
            cos = cosine(vec, value1)
            if key == key1:
                df.append([1, key, key1, "-".join([key1, key]), cos])
            else:
                df.append([0, key, key1, "-".join([key1, key]), cos])
df = pd.DataFrame(df, columns=["condition", "naturalsound", "wordsound", "item", "cos"])
df = df.dropna()

df.to_csv("condition_df_spectrum.csv")
