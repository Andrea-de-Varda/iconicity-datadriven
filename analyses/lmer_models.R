library(lmerTest)

setwd("path/to/your/wd")

vggish <- read.csv("condition_df_VGGish.csv")
head(vggish)

# comment / uncomment according to how you want to deal with cosine < 0
#total_values <- length(vggish$cos)
#values_less_than_zero <- length(which(vggish$cos < 0))
#percentage_less_than_zero <- (values_less_than_zero / total_values) * 100 
#vggish$cos <- ifelse(vggish$cos < 0, 0, vggish$cos)

model_VGGish <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = vggish, REML = FALSE)
summary(model_VGGish)


speechvgg <- read.csv("condition_df_SpeechVGG.csv")
head(speechvgg)

# comment / uncomment according to how you want to deal with cosine < 0
#total_values <- length(speechvgg$cos)
#values_less_than_zero <- length(which(speechvgg$cos < 0))
#percentage_less_than_zero <- (values_less_than_zero / total_values) * 100 
#speechvgg$cos <- ifelse(speechvgg$cos < 0, 0, speechvgg$cos)

model_speechvgg <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = speechvgg, REML = FALSE)
summary(model_speechvgg)


spectrum <- read.csv("condition_df_spectrum_new.csv")
head(spectrum)

# comment / uncomment according to how you want to deal with cosine < 0
#total_values <- length(spectrum$cos)
#values_less_than_zero <- length(which(spectrum$cos < 0))
#percentage_less_than_zero <- (values_less_than_zero / total_values) * 100 
#spectrum$cos <- ifelse(spectrum$cos < 0, 0, spectrum$cos)

model_spectrum <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = spectrum, REML = FALSE)
summary(model_spectrum)

# mean values 
mean(spectrum[spectrum$condition == 1, ]$cos)
mean(spectrum[spectrum$condition == 0, ]$cos)

mean(vggish[vggish$condition == 1, ]$cos)
mean(vggish[vggish$condition == 0, ]$cos)

mean(speechvgg[speechvgg$condition == 1, ]$cos)
mean(speechvgg[speechvgg$condition == 0, ]$cos)


sd(spectrum[spectrum$condition == 1, ]$cos)
sd(spectrum[spectrum$condition == 0, ]$cos)

sd(vggish[vggish$condition == 1, ]$cos) 
sd(vggish[vggish$condition == 0, ]$cos) 

sd(speechvgg[speechvgg$condition == 1, ]$cos)
sd(speechvgg[speechvgg$condition == 0, ]$cos)

##################################################################################

onom_a <- read.csv("df_onomat/AE_annotate_onomatopoeia_1.csv", sep=",")
onom_r <- read.csv("df_onomat/RS_annotate_onomatopoeia.csv")

onom <- merge(onom_a, onom_r, by = "word")
onom$onom <- rowSums(onom[,c("onomatopoeia.x", "onomatopoeia.y")], na.rm=TRUE)
nrow(onom[onom$onom == 0,]) # 235
nrow(onom[onom$onom != 0,]) # 139

tokeep <- onom[onom$onom == 0,]$word


# spectrum

spectrum <- read.csv("condition_df_spectrum_new.csv")
spectrum_filtered <- spectrum[spectrum$wordsound %in% tokeep & spectrum$naturalsound %in% tokeep,]
nrow(spectrum_filtered)

model_spectrum_filtered <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = spectrum_filtered, REML = FALSE)
summary(model_spectrum_filtered)


# vggish
vggish <- read.csv("condition_df_VGGish.csv")
vggish_filtered <- vggish[vggish$wordsound %in% tokeep & vggish$naturalsound %in% tokeep,]

model_vggish_filtered <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = vggish_filtered, REML = FALSE)
summary(model_vggish_filtered)


# speechvgg
speechvgg <- read.csv("condition_df_SpeechVGG.csv")
speechvgg_filtered <- speechvgg[speechvgg$wordsound %in% tokeep & speechvgg$naturalsound %in% tokeep,]

model_speechvgg_filtered <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = speechvgg_filtered, REML = FALSE)
summary(model_speechvgg_filtered)



###################
# random shuffles #
###################

n_iter = 10
model_summaries_vggish <- vector("list", n_iter)

for(i in 1:n_iter){
  shuffled_data <- vggish
  shuffled_data$condition <- sample(vggish$condition, size = nrow(vggish), replace = FALSE)
  shuffled_model <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = shuffled_data, REML = FALSE)
  model_summaries_vggish[[i]] <- summary(shuffled_model)
  print(model_summaries_vggish[[i]])
}

model_summaries_spectrum <- vector("list", n_iter)

for(i in 1:n_iter){
  shuffled_data <- spectrum
  shuffled_data$condition <- sample(spectrum$condition, size = nrow(spectrum), replace = FALSE)
  shuffled_model <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = shuffled_data, REML = FALSE)
  model_summaries_spectrum[[i]] <- summary(shuffled_model)
  print(model_summaries_spectrum[[i]])
}

model_summaries_speechvgg <- vector("list", n_iter)

for(i in 1:n_iter){
  shuffled_data <- speechvgg
  shuffled_data$condition <- sample(speechvgg$condition, size = nrow(speechvgg), replace = FALSE)
  shuffled_model <- lmer(cos ~ condition + (1|wordsound) + (1|naturalsound), data = shuffled_data, REML = FALSE)
  model_summaries_speechvgg[[i]] <- summary(shuffled_model)
  print(model_summaries_speechvgg[[i]])
}

for(i in 1:length(model_summaries_vggish)) {
  coef_summary <- model_summaries_vggish[[i]]$coefficients
  t_value_condition <- coef_summary["condition", "t value"]
  cat(t_value_condition, "  ")
}
# 1.937196   -0.6254769   -0.7808356   0.2752559   -0.4411041   0.6832468   -0.9617497   0.7603665   0.9417488   1.677705 

for(i in 1:length(model_summaries_spectrum)) {
  coef_summary <- model_summaries_spectrum[[i]]$coefficients
  t_value_condition <- coef_summary["condition", "t value"]
  cat(t_value_condition, "  ")
}
# 0.9610185   0.3518152   1.838829   0.4654003   -0.629974   -0.03413147   -0.1341185   0.2604201   1.136507   0.558322

for(i in 1:length(model_summaries_speechvgg)) {
  coef_summary <- model_summaries_speechvgg[[i]]$coefficients
  t_value_condition <- coef_summary["condition", "t value"]
  cat(t_value_condition, "  ")
}
# -1.110445   -0.3411548   1.136589   1.033223   -0.5121215   -0.2925585   -0.5833271   0.7208493   -1.081227   -1.077626 


t_values_vggish <- numeric(length(model_summaries_vggish))
for(i in 1:length(model_summaries_vggish)) {
  coef_summary <- model_summaries_vggish[[i]]$coefficients
  t_values_vggish[i] <- coef_summary["condition", "t value"]
}
mean_t_value_vggish <- mean(t_values_vggish)

t_values_spectrum <- numeric(length(model_summaries_spectrum))
for(i in 1:length(model_summaries_spectrum)) {
  coef_summary <- model_summaries_spectrum[[i]]$coefficients
  t_values_spectrum[i] <- coef_summary["condition", "t value"]
}
mean_t_value_spectrum <- mean(t_values_spectrum)

t_values_speechvgg <- numeric(length(model_summaries_speechvgg))
for(i in 1:length(model_summaries_speechvgg)) {
  coef_summary <- model_summaries_speechvgg[[i]]$coefficients
  t_values_speechvgg[i] <- coef_summary["condition", "t value"]
}
mean_t_value_speechvgg <- mean(t_values_speechvgg)

