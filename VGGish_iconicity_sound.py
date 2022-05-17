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
import glob
from researchpy import ttest, corr_pair
import seaborn as sns
import statsmodels.api as sm
from pymer4.models import Lmer

def cosine(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

chdir("path/to/your/WD")

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
c["query"] = 10 # for FreeSound API
c = c.drop('auditory', 1)
c.to_csv("query.csv", header=False, index=False) # use it with the Freesound Scraper

vggmodel = hub.load('https://tfhub.dev/google/vggish/1')

def embedding_from_fn(fn, resample):
    x, sr = librosa.load(fn,sr=None)
    x_16k = librosa.resample(x,sr,resample) #resample to 16KHz
    
    if len(x_16k) < resample:
        x_16k = librosa.util.fix_length(x_16k, size=resample)
    
    #print(x_16k.shape)
    embedding = vggmodel(x_16k).numpy()
    #mean_embedding = np.mean(embedding, axis=0)
    return embedding

word_sounds = {}
for word in c.word:
    filename = "MALD1_rw/"+word.upper()+".wav"
    sound = embedding_from_fn(filename, 16000)
    word_sounds[word] = sound

word_sounds.keys()
word_sounds = {key:value[0] for key, value in word_sounds.items()}

soundfiles = glob.glob('freesound-data/**/*.wav', recursive=True)
len(soundfiles)

# natural sounds and noises
all_sounds = {}
done = set()
for filename in soundfiles:
    fileid = re.sub("freesound\-data\/samples\_[0-7]\/", "", filename)
    print(fileid)
    if fileid in done:
        print("already done")
        pass
    else:
        sound = embedding_from_fn(filename, 16000) #  40000
        print(sound.shape)
        meansound = np.mean(sound, axis=0)
        if np.isnan(meansound[1]):
            print("ERROR")
        else:
            fileid = re.sub("freesound\-data\/samples\_[0-7]\/", "", filename)
            print(fileid)
            all_sounds[fileid] = sound
        done.add(fileid)
        
examples_ns = {key:value[0] for key, value in examples_ns.items()}

midsampled_sounds = {}
for key, value in all_sounds.items():
    n = value.shape[0]
    k = int(n/2)
    midsamp = value[k]
    print(midsamp.shape)
    midsampled_sounds[key] = midsamp
    
with open("all_sounds", 'wb') as handle:
    pickle.dump(all_sounds, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("all_sounds", 'rb') as handle:
    all_sounds = pickle.load(handle)

with open("word_sounds", 'wb') as handle:
    pickle.dump(word_sounds, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("word_sounds", 'rb') as handle:
    word_sounds = pickle.load(handle)

features_all = []
for wordvec in word_sounds.values():
    features_all.append(wordvec)
for soundvec in midsampled_sounds.values():
    features_all.append(soundvec)

tsne = TSNE(n_components=2).fit_transform(features_all)

clabels = np.concatenate((np.zeros(len(word_sounds)), np.ones(len(midsampled_sounds))))
plt.figure(dpi=1200)
plt.scatter(tsne[clabels==0,0], tsne[clabels==0,1], label='speech', color=[0,0.3,0.7,1], s=7)
plt.scatter(tsne[clabels==1,0], tsne[clabels==1,1], label='noise', color=[0,0.6,0.8,1], s=7)

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

# labels
files = ["freesound-data/data"+str(i)+".csv" for i in range(1,8)]
data = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)

data_dict = {}
for idx, row in data.iterrows():
    labels = [w for w in row[2:] if w in word_sounds.keys()]
    file = row[1]
    if file in midsampled_sounds.keys():
        vec = midsampled_sounds[file]
        for label in labels:
            if label in data_dict.keys():
                data_dict[label].append(vec)
            else:
                data_dict[label] = [vec]

right = []
other = []
for key, value in data_dict.items():
    for key1, value1 in word_sounds.items():
        for vec in value:
            cos = cosine(vec, value1)
            if key == key1:
                right.append(cos)
            else:
                other.append(cos)

print("Right =", np.nanmean(right))
print("Other =", np.nanmean(other))

df = []
for key, value in data_dict.items():
    for key1, value1 in word_sounds.items():
        for vec in value:
            cos = cosine(vec, value1)
            if key == key1:
                df.append([1, key, key, key, cos])
            else:
                df.append([0, key, key1, "-".join([key1, key]), cos])
df = pd.DataFrame(df, columns=["condition", "naturalsound", "wordsound", "item", "cos"])
df.dropna

model = Lmer('cos ~ condition + (1|naturalsound) + (1|wordsound)', data=df)
fit = model.fit()
display(fit)
# ANOVA results from fitted model
display(model.anova())

with open("fit", 'wb') as handle:
    pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_dict1 = {}
for k, v in data_dict.items():
    data_dict1[k] = np.mean(v, axis=0)

icodict = {}
for word in word_sounds.keys():
    icon = cosine(data_dict1[word], word_sounds[word])
    icodict[word] = float(icon)

icodf = pd.DataFrame.from_dict(icodict.items())
icodf.columns=["w", "i"]
icodf = icodf.sort_values(by="i", ascending=False).dropna()
icodf.head(10)
icodf.tail(10)
icodf.to_csv("icodf_VGGish.csv", index=False)

################################################
# check correlations with levenshtein distance #
################################################

from Levenshtein import distance as levenshtein_distance
out = []
for c in combinations(word_sounds.keys(), 2):
    if c[0] != c[1]:
        lev = levenshtein_distance(c[0], c[1])
        aud = cosine(word_sounds[c[0]], word_sounds[c[1]])
        out.append([lev, aud])

out = pd.DataFrame(out, columns=["lev", "aud"])
corr_pair(out)

from pyphonetics import RefinedSoundex, Metaphone
rs = RefinedSoundex()
metaphone = Metaphone()

out = []
for c in combinations(word_sounds.keys(), 2):
    if c[0] != c[1]:
        rsd = rs.distance(c[0], c[1])
        mpd = metaphone.distance(c[0], c[1])
        aud = cosine(word_sounds[c[0]], word_sounds[c[1]])
        phon0 = pronounce_dict[c[0]].split()
        phon1 = pronounce_dict[c[1]].split()
        lev_phono = levenshtein.distance(phon0, phon1)
        out.append([rsd, mpd, aud, lev_phono])

out = pd.DataFrame(out, columns=["rsd", "mpd", "aud", "phon_lev"])
corr_pair(out)

# multivariate analogue of standard deviation
global_centroid = np.mean(features_all, axis=0)
out = []
for vec in features_all:
    out.append(cosine(vec, global_centroid))
sum(out)/len(features_all)
# 0.67635934