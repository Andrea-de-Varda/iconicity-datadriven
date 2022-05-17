import pandas as pd
import numpy as np
from os import chdir, listdir
from researchpy import corr_pair
import sys
chdir("path/to/your/WD")
sys.path.append("../..")

datadriven = pd.read_csv("icodf_all.csv") # our measures, merged in a df
    
ratings_2017 = pd.read_csv("english_iconicity_ratings.csv")
r_2017 = {row.Word:row.Iconicity for index, row in ratings_2017.iterrows()}

ratings_2022 = pd.read_csv("iconicity_ratings.csv")
r_2022 = {row.word:row.rating for index, row in ratings_2022.iterrows()}

datadriven["ico_2017"] = datadriven.w.map(r_2017)
datadriven["ico_2022"] = datadriven.w.map(r_2022)

corr_pair(datadriven[["iconicity_speechVGG", "iconicity_VGGish", "ico_2017", "ico_2022"]])


