library(relaimpo)
library(dplyr)
library(tidyr)
library(lavaan)
library(irr)

setwd("path/to/your/wd")
df = read.csv("icodf_covariates.csv")
df$cd = log(df$cd)
df$mark_spectrum <- 1-df$mark_spectrum
df$mark_vggish <- 1-df$mark_vggish
df$mark_speechvgg<- 1-df$mark_speechvgg

(colMeans(is.na(df)))*100

# SpeechVGG
fit_speechvgg_2017 <- lm(ico_2017 ~ mark_speechvgg + cd + sensory_sound + conc + freq + aoa + length + i_speechvgg, data=df)
summary(fit_speechvgg_2017)
coef_summary_speechvgg_2017 <- summary(fit_speechvgg_2017)$coefficients
coef_summary_speechvgg_2017 <- round(coef_summary_speechvgg_2017, 3)
relimp_speechvgg_2017 <- calc.relimp(fit_speechvgg_2017)

coef_summary_df_speechvgg_2017 <- as.data.frame(coef_summary_speechvgg_2017)
coef_summary_df_speechvgg_2017$Variable <- rownames(coef_summary_df_speechvgg_2017)
stacked_rel_imp_df_speechvgg_2017 <- as.data.frame(stack(round(relimp_speechvgg_2017$lmg, 3)))
names(stacked_rel_imp_df_speechvgg_2017) <- c("Relative_Importance", "Variable")
coef_summary_df_speechvgg_2017$Original_Order <- seq_len(nrow(coef_summary_df_speechvgg_2017))
final_data_speechvgg_2017 <- merge(coef_summary_df_speechvgg_2017, stacked_rel_imp_df_speechvgg_2017, by = "Variable")
final_data_speechvgg_2017 <- final_data_speechvgg_2017[order(final_data_speechvgg_2017$Original_Order), ]
#final_data_speechvgg_2017$Original_Order <- NULL
final_data_speechvgg_2017[nrow(final_data_speechvgg_2017), "Variable"] <- "iconicity"
final_data_speechvgg_2017[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_speechvgg_2017) != "Variable"
colnames(final_data_speechvgg_2017)[cols_to_change] <- paste(colnames(final_data_speechvgg_2017)[cols_to_change], "_speechvgg", sep = "")

########################

fit_speechvgg_2022 <- lm(ico_2022 ~ mark_speechvgg + cd + sensory_sound + conc + freq + aoa + length + i_speechvgg, data=df)
summary(fit_speechvgg_2022)
coef_summary_speechvgg_2022 <- summary(fit_speechvgg_2022)$coefficients
coef_summary_speechvgg_2022 <- round(coef_summary_speechvgg_2022, 3)
relimp_speechvgg_2022 <- calc.relimp(fit_speechvgg_2022)

coef_summary_df_speechvgg_2022 <- as.data.frame(coef_summary_speechvgg_2022)
coef_summary_df_speechvgg_2022$Variable <- rownames(coef_summary_df_speechvgg_2022)
stacked_rel_imp_df_speechvgg_2022 <- as.data.frame(stack(round(relimp_speechvgg_2022$lmg, 3)))
names(stacked_rel_imp_df_speechvgg_2022) <- c("Relative_Importance", "Variable")
coef_summary_df_speechvgg_2022$Original_Order <- seq_len(nrow(coef_summary_df_speechvgg_2022))
final_data_speechvgg_2022 <- merge(coef_summary_df_speechvgg_2022, stacked_rel_imp_df_speechvgg_2022, by = "Variable")
final_data_speechvgg_2022 <- final_data_speechvgg_2022[order(final_data_speechvgg_2022$Original_Order), ]
#final_data_speechvgg_2022$Original_Order <- NULL
final_data_speechvgg_2022[nrow(final_data_speechvgg_2022), "Variable"] <- "iconicity"
final_data_speechvgg_2022[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_speechvgg_2022) != "Variable"
colnames(final_data_speechvgg_2022)[cols_to_change] <- paste(colnames(final_data_speechvgg_2022)[cols_to_change], "_speechvgg", sep = "")

# spectrum 

fit_spectrum_2017 <- lm(ico_2017 ~ mark_spectrum + cd + sensory_sound + conc + freq + aoa + length + i_spectrum, data=df)
summary(fit_spectrum_2017)
coef_summary_spectrum_2017 <- summary(fit_spectrum_2017)$coefficients
coef_summary_spectrum_2017 <- round(coef_summary_spectrum_2017, 3)
relimp_spectrum_2017 <- calc.relimp(fit_spectrum_2017)

coef_summary_df_spectrum_2017 <- as.data.frame(coef_summary_spectrum_2017)
coef_summary_df_spectrum_2017$Variable <- rownames(coef_summary_df_spectrum_2017)
stacked_rel_imp_df_spectrum_2017 <- as.data.frame(stack(round(relimp_spectrum_2017$lmg, 3)))
names(stacked_rel_imp_df_spectrum_2017) <- c("Relative_Importance", "Variable")
coef_summary_df_spectrum_2017$Original_Order <- seq_len(nrow(coef_summary_df_spectrum_2017))
final_data_spectrum_2017 <- merge(coef_summary_df_spectrum_2017, stacked_rel_imp_df_spectrum_2017, by = "Variable")
final_data_spectrum_2017 <- final_data_spectrum_2017[order(final_data_spectrum_2017$Original_Order), ]
final_data_spectrum_2017$Original_Order <- NULL
final_data_spectrum_2017[nrow(final_data_spectrum_2017), "Variable"] <- "iconicity"
final_data_spectrum_2017[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_spectrum_2017) != "Variable"
colnames(final_data_spectrum_2017)[cols_to_change] <- paste(colnames(final_data_spectrum_2017)[cols_to_change], "_spectrum", sep = "")

########################

fit_spectrum_2022 <- lm(ico_2022 ~ mark_spectrum + cd + sensory_sound + conc + freq + aoa + length + i_spectrum, data=df)
summary(fit_spectrum_2022)
coef_summary_spectrum_2022 <- summary(fit_spectrum_2022)$coefficients
coef_summary_spectrum_2022 <- round(coef_summary_spectrum_2022, 3)
relimp_spectrum_2022 <- calc.relimp(fit_spectrum_2022)

coef_summary_df_spectrum_2022 <- as.data.frame(coef_summary_spectrum_2022)
coef_summary_df_spectrum_2022$Variable <- rownames(coef_summary_df_spectrum_2022)
stacked_rel_imp_df_spectrum_2022 <- as.data.frame(stack(round(relimp_spectrum_2022$lmg, 3)))
names(stacked_rel_imp_df_spectrum_2022) <- c("Relative_Importance", "Variable")
coef_summary_df_spectrum_2022$Original_Order <- seq_len(nrow(coef_summary_df_spectrum_2022))
final_data_spectrum_2022 <- merge(coef_summary_df_spectrum_2022, stacked_rel_imp_df_spectrum_2022, by = "Variable")
final_data_spectrum_2022 <- final_data_spectrum_2022[order(final_data_spectrum_2022$Original_Order), ]
final_data_spectrum_2022$Original_Order <- NULL
final_data_spectrum_2022[nrow(final_data_spectrum_2022), "Variable"] <- "iconicity"
final_data_spectrum_2022[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_spectrum_2022) != "Variable"
colnames(final_data_spectrum_2022)[cols_to_change] <- paste(colnames(final_data_spectrum_2022)[cols_to_change], "_spectrum", sep = "")

# VGGish

fit_vggish_2017 <- lm(ico_2017 ~ mark_vggish + cd + sensory_sound + conc + freq + aoa + length + i_vggish, data=df)
summary(fit_vggish_2017)
coef_summary_vggish_2017 <- summary(fit_vggish_2017)$coefficients
coef_summary_vggish_2017 <- round(coef_summary_vggish_2017, 3)
relimp_vggish_2017 <- calc.relimp(fit_vggish_2017)

coef_summary_df_vggish_2017 <- as.data.frame(coef_summary_vggish_2017)
coef_summary_df_vggish_2017$Variable <- rownames(coef_summary_df_vggish_2017)
stacked_rel_imp_df_vggish_2017 <- as.data.frame(stack(round(relimp_vggish_2017$lmg, 3)))
names(stacked_rel_imp_df_vggish_2017) <- c("Relative_Importance", "Variable")
coef_summary_df_vggish_2017$Original_Order <- seq_len(nrow(coef_summary_df_vggish_2017))
final_data_vggish_2017 <- merge(coef_summary_df_vggish_2017, stacked_rel_imp_df_vggish_2017, by = "Variable")
final_data_vggish_2017 <- final_data_vggish_2017[order(final_data_vggish_2017$Original_Order), ]
final_data_vggish_2017$Original_Order <- NULL
final_data_vggish_2017[nrow(final_data_vggish_2017), "Variable"] <- "iconicity"
final_data_vggish_2017[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_vggish_2017) != "Variable"
colnames(final_data_vggish_2017)[cols_to_change] <- paste(colnames(final_data_vggish_2017)[cols_to_change], "_vggish", sep = "")

########################

fit_vggish_2022 <- lm(ico_2022 ~ mark_vggish + cd + sensory_sound + conc + freq + aoa + length + i_vggish, data=df)
summary(fit_vggish_2022)
coef_summary_vggish_2022 <- summary(fit_vggish_2022)$coefficients
coef_summary_vggish_2022 <- round(coef_summary_vggish_2022, 3)
relimp_vggish_2022 <- calc.relimp(fit_vggish_2022)

coef_summary_df_vggish_2022 <- as.data.frame(coef_summary_vggish_2022)
coef_summary_df_vggish_2022$Variable <- rownames(coef_summary_df_vggish_2022)
stacked_rel_imp_df_vggish_2022 <- as.data.frame(stack(round(relimp_vggish_2022$lmg, 3)))
names(stacked_rel_imp_df_vggish_2022) <- c("Relative_Importance", "Variable")
coef_summary_df_vggish_2022$Original_Order <- seq_len(nrow(coef_summary_df_vggish_2022))
final_data_vggish_2022 <- merge(coef_summary_df_vggish_2022, stacked_rel_imp_df_vggish_2022, by = "Variable")
final_data_vggish_2022 <- final_data_vggish_2022[order(final_data_vggish_2022$Original_Order), ]
final_data_vggish_2022$Original_Order <- NULL
final_data_vggish_2022[nrow(final_data_vggish_2022), "Variable"] <- "iconicity"
final_data_vggish_2022[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_vggish_2022) != "Variable"
colnames(final_data_vggish_2022)[cols_to_change] <- paste(colnames(final_data_vggish_2022)[cols_to_change], "_vggish", sep = "")

df_2017 <- merge(final_data_spectrum_2017, merge(final_data_vggish_2017, final_data_speechvgg_2017, by="Variable"), by="Variable")
df_2022 <- merge(final_data_spectrum_2022, merge(final_data_vggish_2022, final_data_speechvgg_2022, by="Variable"), by="Variable")

df_2017 <- df_2017[order(df_2017$Original_Order_speechvgg), ]
df_2017$Original_Order_speechvgg <- NULL

df_2022 <- df_2022[order(df_2022$Original_Order_speechvgg), ]
df_2022$Original_Order_speechvgg <- NULL

write.csv(df_2017,file='results_2017.csv', row.names=FALSE)
write.csv(df_2022,file='results_2022.csv', row.names=FALSE)

#########################
# negative cosines to 0 #
#########################

df$i_speechvgg <- ifelse(df$i_speechvgg < 0, 0, df$i_speechvgg)
df$i_spectrum <- ifelse(df$i_spectrum < 0, 0, df$i_spectrum)
df$i_vggish <- ifelse(df$i_vggish < 0, 0, df$i_vggish)

cor.test(df$i_spectrum, df$ico_2017, use = "complete.obs")
cor.test(df$i_spectrum, df$ico_2022, use = "complete.obs")

cor.test(df$i_vggish, df$ico_2017, use = "complete.obs")
cor.test(df$i_vggish, df$ico_2022, use = "complete.obs")

cor.test(df$i_speechvgg, df$ico_2017, use = "complete.obs")
cor.test(df$i_speechvgg, df$ico_2022, use = "complete.obs")

cor.test(df$i_vggish, df$i_speechvgg, use = "complete.obs")

# SpeechVGG
fit_speechvgg_2017 <- lm(ico_2017 ~ mark_speechvgg + cd + sensory_sound + conc + freq + aoa + length + i_speechvgg, data=df)
summary(fit_speechvgg_2017)
coef_summary_speechvgg_2017 <- summary(fit_speechvgg_2017)$coefficients
coef_summary_speechvgg_2017 <- round(coef_summary_speechvgg_2017, 3)
relimp_speechvgg_2017 <- calc.relimp(fit_speechvgg_2017)

coef_summary_df_speechvgg_2017 <- as.data.frame(coef_summary_speechvgg_2017)
coef_summary_df_speechvgg_2017$Variable <- rownames(coef_summary_df_speechvgg_2017)
stacked_rel_imp_df_speechvgg_2017 <- as.data.frame(stack(round(relimp_speechvgg_2017$lmg, 3)))
names(stacked_rel_imp_df_speechvgg_2017) <- c("Relative_Importance", "Variable")
coef_summary_df_speechvgg_2017$Original_Order <- seq_len(nrow(coef_summary_df_speechvgg_2017))
final_data_speechvgg_2017 <- merge(coef_summary_df_speechvgg_2017, stacked_rel_imp_df_speechvgg_2017, by = "Variable")
final_data_speechvgg_2017 <- final_data_speechvgg_2017[order(final_data_speechvgg_2017$Original_Order), ]
#final_data_speechvgg_2017$Original_Order <- NULL
final_data_speechvgg_2017[nrow(final_data_speechvgg_2017), "Variable"] <- "iconicity"
final_data_speechvgg_2017[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_speechvgg_2017) != "Variable"
colnames(final_data_speechvgg_2017)[cols_to_change] <- paste(colnames(final_data_speechvgg_2017)[cols_to_change], "_speechvgg", sep = "")

########################

fit_speechvgg_2022 <- lm(ico_2022 ~ mark_speechvgg + cd + sensory_sound + conc + freq + aoa + length + i_speechvgg, data=df)
summary(fit_speechvgg_2022)
coef_summary_speechvgg_2022 <- summary(fit_speechvgg_2022)$coefficients
coef_summary_speechvgg_2022 <- round(coef_summary_speechvgg_2022, 3)
relimp_speechvgg_2022 <- calc.relimp(fit_speechvgg_2022)

coef_summary_df_speechvgg_2022 <- as.data.frame(coef_summary_speechvgg_2022)
coef_summary_df_speechvgg_2022$Variable <- rownames(coef_summary_df_speechvgg_2022)
stacked_rel_imp_df_speechvgg_2022 <- as.data.frame(stack(round(relimp_speechvgg_2022$lmg, 3)))
names(stacked_rel_imp_df_speechvgg_2022) <- c("Relative_Importance", "Variable")
coef_summary_df_speechvgg_2022$Original_Order <- seq_len(nrow(coef_summary_df_speechvgg_2022))
final_data_speechvgg_2022 <- merge(coef_summary_df_speechvgg_2022, stacked_rel_imp_df_speechvgg_2022, by = "Variable")
final_data_speechvgg_2022 <- final_data_speechvgg_2022[order(final_data_speechvgg_2022$Original_Order), ]
#final_data_speechvgg_2022$Original_Order <- NULL
final_data_speechvgg_2022[nrow(final_data_speechvgg_2022), "Variable"] <- "iconicity"
final_data_speechvgg_2022[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_speechvgg_2022) != "Variable"
colnames(final_data_speechvgg_2022)[cols_to_change] <- paste(colnames(final_data_speechvgg_2022)[cols_to_change], "_speechvgg", sep = "")

# spectrum 

fit_spectrum_2017 <- lm(ico_2017 ~ mark_spectrum + cd + sensory_sound + conc + freq + aoa + length + i_spectrum, data=df)
summary(fit_spectrum_2017)
coef_summary_spectrum_2017 <- summary(fit_spectrum_2017)$coefficients
coef_summary_spectrum_2017 <- round(coef_summary_spectrum_2017, 3)
relimp_spectrum_2017 <- calc.relimp(fit_spectrum_2017)

coef_summary_df_spectrum_2017 <- as.data.frame(coef_summary_spectrum_2017)
coef_summary_df_spectrum_2017$Variable <- rownames(coef_summary_df_spectrum_2017)
stacked_rel_imp_df_spectrum_2017 <- as.data.frame(stack(round(relimp_spectrum_2017$lmg, 3)))
names(stacked_rel_imp_df_spectrum_2017) <- c("Relative_Importance", "Variable")
coef_summary_df_spectrum_2017$Original_Order <- seq_len(nrow(coef_summary_df_spectrum_2017))
final_data_spectrum_2017 <- merge(coef_summary_df_spectrum_2017, stacked_rel_imp_df_spectrum_2017, by = "Variable")
final_data_spectrum_2017 <- final_data_spectrum_2017[order(final_data_spectrum_2017$Original_Order), ]
final_data_spectrum_2017$Original_Order <- NULL
final_data_spectrum_2017[nrow(final_data_spectrum_2017), "Variable"] <- "iconicity"
final_data_spectrum_2017[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_spectrum_2017) != "Variable"
colnames(final_data_spectrum_2017)[cols_to_change] <- paste(colnames(final_data_spectrum_2017)[cols_to_change], "_spectrum", sep = "")

########################

fit_spectrum_2022 <- lm(ico_2022 ~ mark_spectrum + cd + sensory_sound + conc + freq + aoa + length + i_spectrum, data=df)
summary(fit_spectrum_2022)
coef_summary_spectrum_2022 <- summary(fit_spectrum_2022)$coefficients
coef_summary_spectrum_2022 <- round(coef_summary_spectrum_2022, 3)
relimp_spectrum_2022 <- calc.relimp(fit_spectrum_2022)

coef_summary_df_spectrum_2022 <- as.data.frame(coef_summary_spectrum_2022)
coef_summary_df_spectrum_2022$Variable <- rownames(coef_summary_df_spectrum_2022)
stacked_rel_imp_df_spectrum_2022 <- as.data.frame(stack(round(relimp_spectrum_2022$lmg, 3)))
names(stacked_rel_imp_df_spectrum_2022) <- c("Relative_Importance", "Variable")
coef_summary_df_spectrum_2022$Original_Order <- seq_len(nrow(coef_summary_df_spectrum_2022))
final_data_spectrum_2022 <- merge(coef_summary_df_spectrum_2022, stacked_rel_imp_df_spectrum_2022, by = "Variable")
final_data_spectrum_2022 <- final_data_spectrum_2022[order(final_data_spectrum_2022$Original_Order), ]
final_data_spectrum_2022$Original_Order <- NULL
final_data_spectrum_2022[nrow(final_data_spectrum_2022), "Variable"] <- "iconicity"
final_data_spectrum_2022[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_spectrum_2022) != "Variable"
colnames(final_data_spectrum_2022)[cols_to_change] <- paste(colnames(final_data_spectrum_2022)[cols_to_change], "_spectrum", sep = "")

# VGGish

fit_vggish_2017 <- lm(ico_2017 ~ mark_vggish + cd + sensory_sound + conc + freq + aoa + length + i_vggish, data=df)
summary(fit_vggish_2017)
coef_summary_vggish_2017 <- summary(fit_vggish_2017)$coefficients
coef_summary_vggish_2017 <- round(coef_summary_vggish_2017, 3)
relimp_vggish_2017 <- calc.relimp(fit_vggish_2017)

coef_summary_df_vggish_2017 <- as.data.frame(coef_summary_vggish_2017)
coef_summary_df_vggish_2017$Variable <- rownames(coef_summary_df_vggish_2017)
stacked_rel_imp_df_vggish_2017 <- as.data.frame(stack(round(relimp_vggish_2017$lmg, 3)))
names(stacked_rel_imp_df_vggish_2017) <- c("Relative_Importance", "Variable")
coef_summary_df_vggish_2017$Original_Order <- seq_len(nrow(coef_summary_df_vggish_2017))
final_data_vggish_2017 <- merge(coef_summary_df_vggish_2017, stacked_rel_imp_df_vggish_2017, by = "Variable")
final_data_vggish_2017 <- final_data_vggish_2017[order(final_data_vggish_2017$Original_Order), ]
final_data_vggish_2017$Original_Order <- NULL
final_data_vggish_2017[nrow(final_data_vggish_2017), "Variable"] <- "iconicity"
final_data_vggish_2017[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_vggish_2017) != "Variable"
colnames(final_data_vggish_2017)[cols_to_change] <- paste(colnames(final_data_vggish_2017)[cols_to_change], "_vggish", sep = "")

########################

fit_vggish_2022 <- lm(ico_2022 ~ mark_vggish + cd + sensory_sound + conc + freq + aoa + length + i_vggish, data=df)
summary(fit_vggish_2022)
coef_summary_vggish_2022 <- summary(fit_vggish_2022)$coefficients
coef_summary_vggish_2022 <- round(coef_summary_vggish_2022, 3)
relimp_vggish_2022 <- calc.relimp(fit_vggish_2022)

coef_summary_df_vggish_2022 <- as.data.frame(coef_summary_vggish_2022)
coef_summary_df_vggish_2022$Variable <- rownames(coef_summary_df_vggish_2022)
stacked_rel_imp_df_vggish_2022 <- as.data.frame(stack(round(relimp_vggish_2022$lmg, 3)))
names(stacked_rel_imp_df_vggish_2022) <- c("Relative_Importance", "Variable")
coef_summary_df_vggish_2022$Original_Order <- seq_len(nrow(coef_summary_df_vggish_2022))
final_data_vggish_2022 <- merge(coef_summary_df_vggish_2022, stacked_rel_imp_df_vggish_2022, by = "Variable")
final_data_vggish_2022 <- final_data_vggish_2022[order(final_data_vggish_2022$Original_Order), ]
final_data_vggish_2022$Original_Order <- NULL
final_data_vggish_2022[nrow(final_data_vggish_2022), "Variable"] <- "iconicity"
final_data_vggish_2022[1, "Variable"] <- "markedness"
cols_to_change <- colnames(final_data_vggish_2022) != "Variable"
colnames(final_data_vggish_2022)[cols_to_change] <- paste(colnames(final_data_vggish_2022)[cols_to_change], "_vggish", sep = "")

df_2017 <- merge(final_data_spectrum_2017, merge(final_data_vggish_2017, final_data_speechvgg_2017, by="Variable"), by="Variable")
df_2022 <- merge(final_data_spectrum_2022, merge(final_data_vggish_2022, final_data_speechvgg_2022, by="Variable"), by="Variable")

df_2017 <- df_2017[order(df_2017$Original_Order_speechvgg), ]
df_2017$Original_Order_speechvgg <- NULL

df_2022 <- df_2022[order(df_2022$Original_Order_speechvgg), ]
df_2022$Original_Order_speechvgg <- NULL

write.csv(df_2017,file='results_2017_costo0.csv', row.names=FALSE)
write.csv(df_2022,file='results_2022_costo0.csv', row.names=FALSE)

###############################################################################

onom_a <- read.csv("df_onomat/AE_annotate_onomatopoeia_1.csv", sep=",")
onom_r <- read.csv("df_onomat/RS_annotate_onomatopoeia.csv")

onom <- merge(onom_a, onom_r, by = "word")
onom$onom <- rowSums(onom[,c("onomatopoeia.x", "onomatopoeia.y")], na.rm=TRUE)

# agreement
onom$onom1 <- is.na(onom[, "onomatopoeia.x"])
onom$onom2 <- is.na(onom[, "onomatopoeia.y"])

result <- kappa2(as.matrix(onom[,c("onom1", "onom2")]))
print(result)

nrow(onom[onom$onom == 0,])
nrow(onom[onom$onom != 0,]) 

tokeep <- onom[onom$onom == 0,]$word

df_filtered <- df[df$w %in% tokeep,]
nrow(df_filtered)

sum(complete.cases(df_filtered[, "ico_2017"]))
sum(complete.cases(df_filtered[, "ico_2022"]))

cor.test(df_filtered$i_spectrum, df_filtered$ico_2017, use = "complete.obs")
cor.test(df_filtered$i_spectrum, df_filtered$ico_2022, use = "complete.obs")

cor.test(df_filtered$i_vggish, df_filtered$ico_2017, use = "complete.obs")
cor.test(df_filtered$i_vggish, df_filtered$ico_2022, use = "complete.obs")

cor.test(df_filtered$i_speechvgg, df_filtered$ico_2017, use = "complete.obs")
cor.test(df_filtered$i_speechvgg, df_filtered$ico_2022, use = "complete.obs")
