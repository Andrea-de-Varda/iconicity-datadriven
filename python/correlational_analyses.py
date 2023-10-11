import pandas as pd
import numpy as np
from os import chdir, listdir
from researchpy import corr_pair
import sys
import statsmodels.api as sm
import pickle
chdir("path/to/your/wd")
sys.path.append("../..") # 

vggish = pd.read_csv("icodfVGGish_cleaned.csv")
spectrum = pd.read_csv("icodf_spectrum_cleaned.csv")[["w", "i"]]
speechvgg = pd.read_csv("icodf_speechVGG_cleaned.csv")[["w", "i"]]

merged_df = pd.merge(vggish, spectrum, on=['w'], suffixes=('_vggish', '_spectrum'), how='outer')
datadriven = pd.merge(merged_df, speechvgg, on=['w'], suffixes=('', '_speechvgg'), how='outer')
datadriven.rename(columns={'i': 'i_speechvgg'}, inplace=True)

print(datadriven)

with open("markedness_speechvgg", 'rb') as handle:
    mrk_speechvgg = pickle.load(handle)
    
with open("markedness_VGGish", 'rb') as handle:
    mrk_vggish = pickle.load(handle)
    
with open("markedness_spectrum", 'rb') as handle:
    mrk_spectrum = pickle.load(handle)
    
datadriven["mark_speechvgg"] = datadriven.w.map(mrk_speechvgg)
datadriven["mark_vggish"] = datadriven.w.map(mrk_vggish)
datadriven["mark_spectrum"] = datadriven.w.map(mrk_spectrum)

sensory_sound = pd.read_csv("/path/to/your/sensory_norms.csv")
sensory_d_sound = {row.Word.lower():row["Auditory.mean"] for index, row in sensory_sound.iterrows()}

ratings_2017 = pd.read_csv("english_iconicity_ratings.csv")
r_2017 = {row.Word:row.Iconicity for index, row in ratings_2017.iterrows()}

ratings_2022 = pd.read_csv("iconicity_ratings.csv")
r_2022 = {row.word:row.rating for index, row in ratings_2022.iterrows()}

datadriven["ico_2017"] = datadriven.w.map(r_2017)
datadriven["ico_2022"] = datadriven.w.map(r_2022)

datadriven["length"] = datadriven.w.map(lambda x: len(x))
datadriven["iconicity_mean"] = datadriven[['i_spectrum', 'i_speechvgg', 'i_vggish']].mean(axis=1)
corr_pair(datadriven[["ico_2017", "ico_2022", "iconicity_mean", "i_spectrum", "i_speechvgg", "i_vggish", "length"]])

datadriven.sort_values(by="i_spectrum", ascending = False)[["w", "i_spectrum"]][:5]
datadriven.sort_values(by="i_spectrum", ascending = False)[["w", "i_spectrum"]][-5:]

datadriven.sort_values(by="i_vggish", ascending = False)[["w", "i_vggish"]][:5]
datadriven.sort_values(by="i_vggish", ascending = False)[["w", "i_vggish"]][-5:]

datadriven.sort_values(by="i_speechvgg", ascending = False)[["w", "i_speechvgg"]][:5]
datadriven.sort_values(by="i_speechvgg", ascending = False)[["w", "i_speechvgg"]][-5:]


# OTHER VARIABLES

sensory = pd.read_csv("path/to/your/amsel_2012_SER.csv")
sensory_d = {row.Concept:row.PC1_scores for index, row in sensory.iterrows()}

sensory2 = pd.read_csv("path/to/your/juhasz_yap_2013_SER.csv")
sensory_d2 = {row.Word:row.SER for index, row in sensory2.iterrows()}

conc = pd.read_csv("path/to/your/brysbaert_2014_concreteness.csv")
conc_d = {row.Word:row["Conc.M"] for index, row in conc.iterrows()}

freq = pd.read_csv("path/to/your/brysbaert_2012_SUBTLEX_POS.csv")
freq_d = {row.Word:row.Lg10WF for index, row in freq.iterrows()}
cd_d = {row.Word:row.CDcount for index, row in freq.iterrows()}

aoa = pd.read_csv("path/to/your/kuperman_2012_AOA.csv")
aoa_d = {row.Word:row["Rating.Mean"] for index, row in aoa.iterrows()}

datadriven["sensory"] = datadriven.w.map(sensory_d)
datadriven["sensory_sound"] = datadriven.w.map(sensory_d_sound)
datadriven["sensory_d2"] = datadriven.w.map(sensory_d2)
datadriven["conc"] = datadriven.w.map(conc_d)
datadriven["freq"] = datadriven.w.map(freq_d)
datadriven["cd"] = datadriven.w.map(cd_d)
datadriven["aoa"] = datadriven.w.map(aoa_d)

datadriven.to_csv("icodf_covariates.csv", index=False)
