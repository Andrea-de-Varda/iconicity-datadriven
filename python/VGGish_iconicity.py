from os import listdir, chdir
import pandas as pd
import re
import numpy as np
import tensorflow_hub as hub
import librosa
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from sklearn.manifold import TSNE
from numpy import dot    
from numpy.linalg import norm
import glob
import random
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from itertools import combinations
from tqdm import tqdm
import glob
from researchpy import ttest, corr_pair
import seaborn as sns
import statsmodels.api as sm
from pymer4.models import Lmer

chdir("path/to/your/wd")

def cosine(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

files = [re.sub("\.wav", "", f.lower()) for f in listdir("MALD1_rw")]

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

# c["query"] = 10 # for FreeSound API

c = c.drop('auditory', 1)
#c.to_csv("query.csv", header=False, index=False)

vggmodel = hub.load('https://tfhub.dev/google/vggish/1')

def embedding_from_fn(fn, resample):
    x, sr = librosa.load(fn,sr=None)
    x_16k = librosa.resample(x,sr,resample)
    
    if len(x_16k) < resample:
        x_16k = librosa.util.fix_length(x_16k, size=resample)

    embedding = vggmodel(x_16k).numpy()
    return embedding

word_sounds = {}
for word in c.word:
    filename = "MALD1_rw/"+word.upper()+".wav"
    sound = embedding_from_fn(filename, 16000) # 40000
    word_sounds[word] = sound

word_sounds.keys()
word_sounds = {key:value[0] for key, value in word_sounds.items()}

soundfiles = glob.glob('path/to/your/freesound_data/*.wav')
noises1 = [re.sub('path/to/your/freesound_data/', "", s) for s in soundfiles]
noises = [re.search(r"([a-zA-Z]+)_\d+_\d+\.wav", filename).group(1) for filename in noises1]
print("Check =", len(noises) == len(soundfiles))

# natural sounds and noises
all_sounds = []
soundfiles_cleaned = []
for filename, name in tqdm(zip(soundfiles, noises), total = len(noises)):
    sound = embedding_from_fn(filename, 16000)
    meansound = np.mean(sound, axis=0)
    if np.isnan(meansound[1]):
        print(f"ERROR: could not process {filename}")
    else:
        all_sounds.append(sound)
        soundfiles_cleaned.append(name)
        
midsampled_sounds = []
for value in all_sounds:
    n = value.shape[0]
    k = int(n/2)
    midsamp = value[k]
    print(midsamp.shape)
    midsampled_sounds.append(midsamp)
    
# with open("all_sounds_VGGish", 'wb') as handle:
#     pickle.dump(all_sounds, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("all_sounds_VGGish", 'rb') as handle:
    all_sounds = pickle.load(handle)

# with open("word_sounds_VGGish", 'wb') as handle:
#     pickle.dump(word_sounds, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("word_sounds_VGGish", 'rb') as handle:
    word_sounds = pickle.load(handle)

features_all = []
for wordvec in word_sounds.values():
    features_all.append(wordvec)
for soundvec in midsampled_sounds:
    features_all.append(soundvec)

############################################################
# visualize with t-sne and compute within-class similarity #
############################################################

import matplotlib as pylab
pylab.rc('font', family='sans serif', size=12)
tsne = TSNE(n_components=2).fit_transform(np.array(features_all))

clabels = np.concatenate((np.zeros(len(word_sounds)), np.ones(len(midsampled_sounds))))
plt.figure(dpi=300)
plt.scatter(tsne[clabels==0,0], tsne[clabels==0,1], label='$\overrightarrow{s_w}$', color=[0,0.3,0.7,1], s=7)
plt.scatter(tsne[clabels==1,0], tsne[clabels==1,1], label='$\overrightarrow{s_n}$', color=[0,0.6,0.8,1], s=7)

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

############################################################

my_check = ["yes_2_373.wav", "yes_6_141.wav", "yes_6_148.wav", "no_4_109.wav", "no_5_38.wav", "no_5_55.wav", "no_5_109.wav", "no_5_113.wav", "no_5_163.wav", "no_5_174.wav", "no_5_200.wav", "no_6_357.wav"] # manually excluded by me after annotators checked

filter_array = [noise not in my_check for noise in noises1]
noises = [noise for noise, keep in zip(noises, filter_array) if keep]
midsampled_sounds = [noise for noise, keep in zip(midsampled_sounds, filter_array) if keep]

data_dict = {}
for label, vec in zip(noises, midsampled_sounds):
    if label in data_dict.keys():
        data_dict[label].append(vec)
    else:
        data_dict[label] = [vec]
        
wordsounds = set(word_sounds.keys())
natsounds  = set(data_dict.keys())
words_cleaned = set.intersection(natsounds, wordsounds)
len(words_cleaned) # 367

data_dict = {key:value for key, value in data_dict.items() if key in words_cleaned}
word_sounds = {key:value for key, value in word_sounds.items() if key in words_cleaned}

markedness_dict = {}
for word, vec in word_sounds.items():
    mark = np.mean([cosine(vec, vec1) for vec1 in word_sounds.values()])
    markedness_dict[word] = mark
    
with open("markedness_VGGish", 'wb') as handle:
    pickle.dump(markedness_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

df = []
for key, value in tqdm(data_dict.items()):
    for key1, value1 in word_sounds.items():
        for vec in value:
            cos = cosine(vec, value1)
            if key == key1:
                df.append([1, key, key, key, cos])
            else:
                df.append([0, key, key1, "-".join([key1, key]), cos])
df = pd.DataFrame(df, columns=["condition", "naturalsound", "wordsound", "item", "cos"])
df = df.dropna()
df.to_csv("condition_df_VGGish.csv")

#####################
# iconicity ratings #
#####################

# I had multiple vectors, now I need one single value for each word --> averaging
data_dict1 = {}
for k, v in data_dict.items():
    data_dict1[k] = np.mean(v, axis=0)

icodict = {}
for word in word_sounds.keys():
    icon = cosine(data_dict1[word], word_sounds[word])
    icodict[word] = float(icon)

# in paper, list the most iconic words according to the model
icodf = pd.DataFrame.from_dict(icodict.items())
icodf.columns=["w", "i"]
icodf = icodf.sort_values(by="i", ascending=False).dropna()

icodf.head(5)
icodf.tail(5)

icodf.to_csv("icodfVGGish.csv", index=False)
