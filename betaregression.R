library(betareg)
library(jtools)
library(dplyr)
library(Hmisc)
library(ggplot2)

setwd("Q:/FORLand/Clemens/chapter02")
#### Farms
df = read.csv("data/tables/df_data_farms.csv", sep=";")
str(df)

df$Oeko = as.factor(df$Oeko)
df$num_comp_c = as.factor(df$num_comp_c)
df$energy = as.factor(df$energy)
df$farm_size_class = as.factor(df$farm_size_class)
df$big_netw = as.factor(df$big_netw)
df$medium_netw = as.factor(df$medium_netw)
df$single_farm = as.factor(df$single_farm)
str(df)

summary(df)

df_sub = df[df$num_comp_c != "unkown" , ]
df_sub = df

df_sub_std =
  df_sub %>% 
  mutate_at(c("farm_area", "Rinder", "Schweine", "ackerzahl", "dem", "slope", "Schafe", "share_1", "share_2", "share_3", "share_4", 
    "share_5", "share_6", "share_7", "share_9", "share_10","share_12", 
    "share_13", "share_14", "share_60", "share_80", "share_99","share_1100", 
    "share_1200", "share_2100", "share_2200", "share_3100", "share_3200","share_4100", 
    "share_4200", "share_5000", "share_6110", "share_6130", "share_6140","share_6210", 
    "share_6230", "share_6240", "share_7120", "share_7130", "share_7140","share_7210", 
    "share_7220", "share_7230", "share_7310", "share_7330", "share_7400","share_7540", 
    "share_7610", "share_7630", "share_7720", "share_7800", "share_8110","share_8120", 
    "share_8130", "share_8140", "share_8210", "share_8220", "share_8230","share_8240"), scale)

m_std = betareg(ndvi_c_factor ~ farm_area + Oeko + single_farm  + big_netw +
              Rinder + Schweine + Schafe + ackerzahl + dem + slope +share_1 + share_2 +
              share_4 + share_6 + share_7 + share_9 + share_10 + share_12 + 
              share_13 + share_14 + share_8210 + share_5000 + share_4200 +
              share_3200 + share_3100 + share_2200+ share_2100 +share_1200 + 
              share_1100, data=df_sub_std)

df_sub_std$prediction1 = predict(m_std, newdata=df_sub_std, type="response")
df_sub_std$prediction_quant = predict(m_std, newdata=df_sub_std, type = "quantile", at = c(0.025, 0.975))

t = df_sub_std[, 69:72]

plot(df_sub_std$ndvi_c_factor,df_sub_std$prediction1)

s = summary(m_std, na.rm=TRUE, type = "deviance")

ggplit(df_sub_std, aes(x=))

df_plt = data.frame(
  variable=rownames(s$coefficients$mean),
  coeff_mean=s$coefficients$mean[,1],
  coeff_std_err=s$coefficients$mean[,2])
df_plt = df_plt[2:30,]
ggplot(df_plt, aes(x=variable, y=coeff_mean)) +
  geom_bar(position=position_dodge(), stat="identity",
           colour='black') +
  geom_errorbar(aes(ymin=coeff_mean-coeff_std_err, ymax=coeff_mean+coeff_std_err), width=.2) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

t = m_std$coefficients$mean
t= t[which(names(t) != "(Intercept)")]
barplot(t, las=2)


m_std2 = betareg(ndvi_c_factor ~ farm_area + Oeko +   single_farm + medium_netw + big_netw + 
                  Rinder + dem + slope +share_1 + share_2 + share_4 
                   + share_6 + share_7 + share_9 + share_10 + share_12 + 
                  share_13 + share_4200 + share_14 +
                  share_3200 + share_3100 + share_2200+ share_2100 +share_1200 + 
                  share_1100, data=df_sub)

df_sub_std$prediction2 = predict(m_std2, newdata=df_sub_std, type="response")
plot(df_sub_std$ndvi_c_factor,df_sub_std$prediction2)
summary(m_std2, na.rm=TRUE)
t = m_std2$coefficients$mean
t= t[which(names(t) != "(Intercept)")]
barplot(t, las=2)


m_std3 = betareg(ndvi_c_factor ~ farm_area + Oeko + 
                   Rinder + dem + slope + share_4200 + 
                   share_3200 + share_3100 + share_2200+ share_2100 +share_1200 + 
                   share_1100, data=df_sub)

df_sub_std$prediction3 = predict(m_std3, newdata=df_sub_std, type="response")
plot(df_sub_std$ndvi_c_factor,df_sub_std$prediction3)
summary(m_std3, na.rm=TRUE)
t = m_std3$coefficients$mean
t = t[which(names(t) != "(Intercept)")]
barplot(t, las=2)


df_sub$prediction1 = predict(m, newdata=df_sub, type="response")
plot(df_sub$ndvi_c_factor, df_sub$prediction1)
summary(m, na.rm=TRUE)
t = m$coefficients$mean
barplot(t, las=2)
t = t[which(names(t) != "(Intercept)")]
barplot(t, las=2)

#### Fields
df = read.csv("data/tables/df_data.csv", sep=";")

df_sub = df[df$treatment == "small" | df$treatment == "big", ]
str(df_sub)

df_sub$ID_KTYP = as.factor(df_sub$ID_KTYP)
df_sub$Oeko = as.factor(df_sub$Oeko)
df_sub$treatment = as.factor(df_sub$treatment)
df_sub$energy = as.factor(df_sub$energy)
df_sub$soil_type = as.factor(df_sub$soil_type)

df_sub_std =
  df_sub %>% 
  mutate_at(c("GROESSE", "farm_area", "Rinder", "Schweine", "ackerzahl", "dem", "slope", "Schafe"), scale)

spec = c(train = .5, test = .3, validate = .2)

g = sample(cut(
  seq(nrow(df_sub_std)), 
  nrow(df_sub_std)*cumsum(c(0,spec)),
  labels = names(spec)
))

res = split(df_sub_std, g)


m = betareg(ndvi_c_factor ~ ID_KTYP + GROESSE  + treatment + farm_area +
               Rinder + Schweine + ackerzahl + dem + slope,
             data=res$train)

res$test$prediction1 = predict(m, newdata=res$test, type="response")
plot(res$test$ndvi_c_factor, res$test$prediction1)

summary(m, na.rm=TRUE)