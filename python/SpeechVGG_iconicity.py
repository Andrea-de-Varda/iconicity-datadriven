from os import chdir, listdir
import librosa
chdir("path/to/your/SpeechVGG")
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
import pickle
import pandas as pd
import re
from researchpy import ttest
from scipy.stats import ttest_1samp
from Levenshtein import distance as levenshtein_distance
from scipy.spatial.distance import euclidean
from pyphonetics import RefinedSoundex, Metaphone
from textdistance import levenshtein
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from itertools import combinations
from numpy import dot    
from numpy.linalg import norm
from researchpy import corr_pair
np.random.seed(0)

def cosine(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def audio_to_input(fn, resample):
    x, sr = librosa.load(fn,sr=None)
    x_16k = librosa.resample(x,sr,resample)
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
    feats = model.predict(specs)
    a, b, c, d = np.shape(feats)
    feats_small = np.zeros((len(clips), b, c, d))
    for i in range(0, len(feats), num_random_patches):
        feat_tmp = np.mean(feats[i:i+num_random_patches], axis=0)
        feats_small[int(i/num_random_patches), ...] = feat_tmp
    return feats_small.reshape(len(clips), -1)

def remove_short_clips(clips, length=16200):
    good_idxs = []
    for i, clip in tqdm(enumerate(clips)):
        audio = sf.read(clip)
        audio = audio[0]
        if len(audio)>=length:
            good_idxs.append(i)
    clips=np.array(clips)[good_idxs]
    return clips

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

files = [re.sub("\.wav", "", f.lower()) for f in listdir("path/to/your/MALD1_rw")]
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
c = candidates[candidates.auditory > m+1.5*sd] # 374 candidates
len(c)/len(candidates)

speech_clips  = ["path/to/your/MALD1_rw/"+w.upper()+".wav" for w in c.word]
print(len(speech_clips))


noise_clips = glob('path/to/your/freesound-data/*.wav')
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

############################################################
# visualize with t-sne and compute within-class similarity #
############################################################

import matplotlib as pylab
pylab.rc('font', family='sans serif', size=12)
features_all_1 = np.array([v for v in speech_pca]+[v for v in noise_pca])
tsne = TSNE(n_components=2).fit_transform(features_all_1)

clabels = np.concatenate((np.zeros(len(speech_pca)), np.ones(len(noise_pca))))                                                                                   
plt.figure(dpi=300)
plt.scatter(tsne[clabels==0,0], tsne[clabels==0,1], label= '$\overrightarrow{s_w}$', color=[0,0.3,0.7,1], s=7)
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

##########################################################

noises = [re.sub('path/to/your/freesound-data/', "", s) for s in noise_clips]
print("Check =", len(noises) == len(noise_pca))

my_check = ["yes_2_373.wav", "yes_6_141.wav", "yes_6_148.wav", "no_4_109.wav", "no_5_38.wav", "no_5_55.wav", "no_5_109.wav", "no_5_113.wav", "no_5_163.wav", "no_5_174.wav", "no_5_200.wav", "no_6_357.wav"]

filter_array = [noise not in my_check for noise in noises]
noises = [noise for noise, keep in zip(noises, filter_array) if keep]
noise_pca = [noise for noise, keep in zip(noise_pca, filter_array) if keep]

noises = [re.search(r"([a-zA-Z]+)_\d+_\d+\.wav", filename).group(1) for filename in noises]
print("Check =", len(noises) == len(noise_pca))

data_dict = {}
for label, vec in zip(noises, noise_pca):
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

# >>> for MIXED MODEL (in R)
df = []
for key, value in tqdm(data_dict.items()):
    for key1, value1 in word_sounds.items():
        for vec in value:
            cos = cosine(vec, value1)
            if key == key1:
                df.append([1, key, key1, "-".join([key1, key]), cos])
            else:
                df.append([0, key, key1, "-".join([key1, key]), cos])
df = pd.DataFrame(df, columns=["condition", "naturalsound", "wordsound", "item", "cos"])
df = df.dropna()

df.to_csv("condition_df_SpeechVGG.csv")

#######################
# ICONICITY ESTIMATES #
#######################

# I had multiple vectors, now I need one single value for each word --> averaging
data_dict1 = {}
for k, v in data_dict.items():
    data_dict1[k] = np.mean(v, axis=0)

icodict = {}
for word in word_sounds.keys():
    try:
        icon = cosine(data_dict1[word], word_sounds[word])
        icodict[word] = icon
    except KeyError:
        pass
    
markedness_dict = {}
for word, vec in word_sounds.items():
    mark = np.mean([cosine(vec, vec1) for vec1 in word_sounds.values()])
    markedness_dict[word] = mark
    
with open("markedness_speechvgg", 'wb') as handle:
    pickle.dump(markedness_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

icodf = pd.DataFrame.from_dict(icodict.items())
icodf.columns=["w", "i"]
icodf = icodf.sort_values(by="i", ascending=False).dropna()

icodf.head(5)
icodf.tail(5)

icodf.to_csv("icodf_speechVGG_cleaned.csv")
