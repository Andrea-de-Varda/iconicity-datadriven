from os import chdir, listdir
import librosa
chdir("path/to/your/WD/") # path to cloned directory of SpeechVGG GitHub repo
import sys
sys.path.append("../..")
from libs.speech_vgg import speechVGG
import numpy as np
from keras.models import Model
from libs.data_generator import log_standardize, pad_spec
import soundfile as sf
from scipy.signal import stft
from glob import glob
from sklearn.manifold import TSNE
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import re
from researchpy import ttest
from scipy.stats import ttest_1samp
from Levenshtein import distance as levenshtein_distance
from pyphonetics import RefinedSoundex, Metaphone
from textdistance import levenshtein
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations
from numpy import dot    
from numpy.linalg import norm
from researchpy import corr_pair
from pymer4.models import Lmer
np.random.seed(0)

def cosine(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

files = [re.sub("\.wav", "", f.lower()) for f in listdir("/your/path/to/MALD1_rw")]
sensory = pd.read_csv("/your/path/to/sensory_norms.csv")
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
c = candidates[candidates.auditory > m+1.5*sd] # 374 candidates

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
    print(data_tmp.shape)
    return data_tmp

def clips_to_features_rand(clips, model, num_random_patches=1):
    specs = []
    for clip in clips:
        for n in range(num_random_patches):
            specs.append(audio_to_input(clip, 16200))
    specs = np.array([pad_spec(spec) for spec in specs])
    specs = np.array([spec for spec in specs])
    feats = model.predict(specs)
    a, b, c, d = np.shape(feats)
    feats_small = np.zeros((len(clips), b, c, d))
    for i in range(0, len(feats), num_random_patches):
        feat_tmp = np.mean(feats[i:i+num_random_patches], axis=0)
        feats_small[int(i/num_random_patches), ...] = feat_tmp
    return feats_small.reshape(len(clips), -1)

model = speechVGG(
            include_top=False,
            input_shape=(128, 128, 1),
            classes=3000, # not important
            pooling=None,
            weights= '460hours_3000words_weights.19-0.28.h5',
            transfer_learning=False
        )

sVGG_extractor = Model(inputs=model.input,
                       outputs=model.get_layer('block5_pool').output)

speech_clips  = ["your/path/to/MALD1_rw/"+w.upper()+".wav" for w in c.word]

print(len(speech_clips))

noise_clips = glob('/your/path/to/freesound-data/*/*.wav')
print(len(noise_clips))

###################
# SPEECH FEATURES #
###################

features_speech = clips_to_features_rand(speech_clips, sVGG_extractor, 1)
print(features_speech.shape)

with open("features_speech", 'wb') as handle:
    pickle.dump(features_speech, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("features_speech", 'rb') as handle:
    features_speech = pickle.load(handle)

##################
# NOISE FEATURES #
##################

features_noise = clips_to_features_rand(noise_clips, sVGG_extractor, 1)
print(features_noise.shape)

with open("features_noise", 'wb') as handle:
    pickle.dump(features_noise, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("features_noise", 'rb') as handle:
    features_noise = pickle.load(handle)
    
features_all = np.concatenate([features_speech, features_noise])
print(features_all.shape)
scaled_features = StandardScaler().fit_transform(features_all)
pca = PCA(n_components=128)
all_pca = pca.fit_transform(scaled_features)

with open("all_pca", 'wb') as handle:
    pickle.dump(all_pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("all_pca", 'rb') as handle:
    all_pca = pickle.load(handle)

speech_pca = all_pca[:len(speech_clips)]
noise_pca = all_pca[len(speech_clips):]
print(speech_pca.shape, noise_pca.shape)

word_sounds = {}
for name, feat in zip(c.word,speech_pca):
    word_sounds[name] = feat
print(len(word_sounds))

# check corr with Levehnstein and Soundex
ldt = pd.read_csv("../MALD1_AllData.txt", sep="\t") # to get phonological levehnstein
POS_dict = {}
pronounce_dict = {}
for index, row in ldt.iterrows():
    POS_dict[row.Item] = row.POS
    pronounce_dict[row.Item] = row.Pronunciation
    
rs = RefinedSoundex()
metaphone = Metaphone()
out = []
for comb in combinations(word_sounds.keys(), 2):
    if comb[0] != comb[1]:
        lev = levenshtein_distance(comb[0], comb[1])
        # lev_norm = lev-(abs(len(comb[0]) - len(comb[1]))) # think about this
        aud = cosine(word_sounds[comb[0]], word_sounds[comb[1]])
        rsd = rs.distance(comb[0], comb[1])
        mpd = metaphone.distance(comb[0], comb[1])
        # phono levehnstein
        phon0 = pronounce_dict[comb[0]].split()
        phon1 = pronounce_dict[comb[1]].split()
        lev_phono = levenshtein.distance(phon0, phon1)
        out.append([comb[0], comb[1], lev, lev_phono, aud, rsd, mpd])

out = pd.DataFrame(out, columns=["item1", "item2", "lev", "phon_lev", "aud", "rsd", "mpd"])
print(corr_pair(out[["lev", "phon_lev", "aud", "rsd", "mpd"]]))

# visualize with t-sne
features_all_1 = [v for v in speech_pca]+[v for v in noise_pca]
tsne = TSNE(n_components=2).fit_transform(features_all_1)

clabels = np.concatenate((np.zeros(len(speech_pca)), np.ones(len(noise_pca))))                                                                                   
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

noises = [re.sub(r'(your/path/to/freesound-data/)(.*\/)(.*)(\.wav)', r"\3", file) for file in noise_clips]
print("Check =", len(noises) == len(noise_pca))
natural_sounds = {name:vec for name, vec in zip(noises, noise_pca)}

files = ["../freesound-data/data"+str(i)+".csv" for i in range(1,8)]
data = pd.concat([pd.read_csv(f, header=None) for f in files], ignore_index=True)

data_dict = {}
for idx, row in data.iterrows():
    labels = [w for w in row[2:] if w in word_sounds.keys()]
    file = re.sub("\.wav", "", row[1])
    if file in noises:
        vec = natural_sounds[file]
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

print("Mean right =", np.nanmean(right))
print("Mean other =", np.nanmean(other))

# >>> MIXED MODEL
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

model = Lmer('cos ~ condition + (1|naturalsound) + (1|wordsound)', data=df) # removing random slope (fails to converge)
fit = model.fit()
display(fit)
display(model.anova())

# I had multiple vectors, now I need one single value for each word --> averaging
data_dict1 = {}
for k, v in data_dict.items():
    data_dict1[k] = np.mean(v, axis=0)

icodict = {}
for word in word_sounds.keys():
    icon = cosine(data_dict1[word], word_sounds[word])
    icodict[word] = icon

icodf = pd.DataFrame.from_dict(icodict.items())
icodf.columns=["w", "i"]
icodf = icodf.sort_values(by="i", ascending=False).dropna()
icodf.head(10)
icodf.tail(10)
icodf.to_csv("icodf_speechVGG.csv", index=False)

# multivariate analogue of standard deviation
global_centroid = np.mean(all_pca, axis=0)
out = []
for vec in all_pca:
    out.append(cosine(vec, global_centroid))
print(sum(out)/len(all_pca))
