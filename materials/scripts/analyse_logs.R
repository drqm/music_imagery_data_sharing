setwd('C:/Users/au571303/Documents/projects/memory_music_iEEG')
library(MASS)
library(lme4)
library(effects)
library(ggplot2)
library(viridis)
library(RColorBrewer)
library(ez)
library(ordinal)



files <- list.files('logs', pattern = 'pilot')

# load all pilot files:
d <- data.frame()
for (f in files){
  cdata <- read.table(paste0('logs/',f),sep = ',', header = TRUE)
  if (nrow(d) < 1){
    d <- cdata
  }else{
    d <- rbind(d,cdata)
  }
}
d$subject <- as.character(d$subject)
## compute accuracy:
d$subject[d$subject == "Pilot_6"] <- "pilot6"

d$acc <- as.numeric(d$type == d$response)
d$acc_char <- factor(ifelse(d$acc == 1,"correct","incorrect"),
                     levels = c("incorrect","correct"))
d$type <- as.character(d$type)
#accuracy summary:
prop.table(table(d$subject,d$acc),1)
subs = c("pilot6","pilot7","pilot8","pilot9")
ggplot(d[d$subject== subs,],aes(block,rt, color = subject)) +
  geom_jitter(width = 0.05) +
  geom_boxplot(alpha = 0.3,
             color = 'black', width = 0.2) +
  geom_violin(color = 'black',
              alpha = 0.2,trim = FALSE)+
  facet_wrap(~type)

ggplot(d[d$subject== subs,],aes(block, fill = acc_char)) +
  geom_bar(position = "fill") +
  labs(fill="Accuracy",y="Proportion")+
  scale_fill_viridis(discrete = T)+
  facet_wrap(~subject) 

ggplot(d[d$subject!= "FS",],aes(subject, fill = acc_char)) +
  geom_bar(position = "fill") +
  labs(fill="Accuracy",y="Proportion",x ="Participant")+
  scale_fill_viridis(discrete = T) +
  facet_wrap(~block) + 
  theme_bw()+
  theme(axis.text.x= element_text(hjust = 1.1,angle = 60))


ggplot(d,aes(block,rt, color = subject)) +
  geom_jitter(width = 0.05) +
  geom_boxplot(alpha = 0.3,
               color = 'black', width = 0.2) +
  geom_violin(color = 'black',
              alpha = 0.2,trim = FALSE)+
  facet_wrap(~type)

ggplot(d[d$subject== c("FS","Pilot_6","pilot7"),],aes(block,rt, color = subject)) +
  geom_jitter(width = 0.05) +
  geom_boxplot(alpha = 0.3,
               color = 'black', width = 0.2) +
  geom_violin(color = 'black',
              alpha = 0.2,trim = FALSE)+
  facet_wrap(~type)
