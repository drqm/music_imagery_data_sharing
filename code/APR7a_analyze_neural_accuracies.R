library(dplyr)
library(readr)
library(lme4)
library(ggplot2)
library(effects)
library(patchwork)

# all_data_new <- list.files(path = "~/Desktop/accuracies", full.names = TRUE) %>% 
#   lapply(read_csv) %>% 
#   bind_rows
#scripts_dir = 'Z:/Desktop/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/data_sharing/code/'
scripts_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/working_memory/data_sharing/code/'

setwd(scripts_dir)

data_dir = "../data/neural_accuracy"
fig_dir = "../../results/figures/"
stats_dir = "../../results/stats/"

all_data <- list.files(path = data_dir, full.names = TRUE) %>% 
  lapply(read_csv) %>% 
  bind_rows

recall_within_listen <- all_data[all_data$block == "recall" & all_data$test_type == "within" & all_data$period == "listening", ]
manip_within_listen <- all_data[all_data$block == "manipulation" & all_data$test_type == "within" & all_data$period == "listening", ]
recall_within_imagine <- all_data[all_data$block == "recall" & all_data$test_type == "within" & all_data$period == "imagination", ]                                                                                            
manip_within_imagine <- all_data[all_data$block == "manipulation" & all_data$test_type == "within" & all_data$period == "imagination", ]                                                                                            

all_lists <- list(recall_within_listen, 
                  manip_within_listen, 
                  recall_within_imagine, 
                  manip_within_imagine
)

list_names <- list("rwl", 
                   "mwl", 
                   "rwi", 
                   "mwi"
)

model_list <- vector("list", length = length(all_lists))

#create the models
for (i in seq_along(all_lists)) {
  item <- all_lists[[i]]
  name <- list_names[[i]]
  print(i)
  
  #m0 <- glmer(neur_acc ~ 1 + (1 | sub), family = binomial(link = 'logit'), data = item)
  #m1 <- glmer(neur_acc ~ beh_acc * trial_type + (1 + beh_acc | sub), family = binomial(link = 'logit'), data = item)
  #r1 <- glmer(neur_acc ~ rt * trial_type + (1 + rt | sub), family = binomial(link = 'logit'), data = item)
  
  m0 <- glmer(neur_acc ~ 1 + (1 | sub), family = binomial(link = 'logit'), data = item)
  m1 <- glmer(neur_acc ~ 1 + beh_acc + (1 + beh_acc | sub), family = binomial(link = 'logit'), data = item)
  r1 <- glmer(neur_acc ~ 1 + rt + (1 + rt | sub), family = binomial(link = 'logit'), data = item)
  
  model_list[[name]] <- list(
    mod0 <- m0,
    beh1 <- m1,
    rt1 <- r1,
    beh_anova <- anova(mod0,beh1),
    rt_anova <- anova(mod0,rt1)
  )
}

df_beh_list <- list()
plot_beh_list <- list()

name_list <- c("Recall - Listening", "Manipulation - Listening", "Recall - Imagination", "Manipulation - Imagination")
export_data <- TRUE
#create behavioral accuracy plots
i = 1 #counter for name_list
for (key in list_names) {
  model <- model_list[[key]][[2]]
  
  eff <- effect("beh_acc", model, xlevels=list(beh_acc=c(0, 1)))
  df <- as.data.frame(eff)
  df_beh_list[[key]] <- df
  name <- name_list[i]
  #summary(eff)
  
  df$beh_acc <- as.character(df$beh_acc)
  
  ccoef <- coef(model)
  acc0 <- exp(ccoef$sub$`(Intercept)`)/(1 + exp(ccoef$sub$`(Intercept)`))
  acc1 <- exp(ccoef$sub$`(Intercept)` + ccoef$sub$beh_acc)/(1 + exp(ccoef$sub$`(Intercept)` + ccoef$sub$beh_acc))
  dfcoef <- data.frame('neur_accuracy' = round(c(acc0, acc1),3),
                       'subject' = c(row.names(ccoef$sub), row.names(ccoef$sub)),
                       'beh_acc' = as.character(c(rep(0, length(acc0)), rep(1, length(acc0)))))
  
  p <- ggplot(df, aes(beh_acc, fit)) + geom_point() + 
    geom_point(aes(beh_acc, neur_accuracy),data=dfcoef, alpha=0.4) +
    geom_line(aes(beh_acc, neur_accuracy, group=subject),data=dfcoef, alpha=0.2)+
    geom_violin(aes(beh_acc, neur_accuracy, fill = beh_acc), data=dfcoef, alpha=0.4, show.legend = FALSE)+
    geom_boxplot(aes(beh_acc, neur_accuracy), data=dfcoef,alpha=0.5, width=0.05, fill='white') +
    geom_hline(yintercept = 0.5)+
    coord_cartesian(ylim = c(0.1,0.9))+
    labs(title= paste(name), x="Behavioral Accuracy", y = "Neural Accuracy")+
    scale_fill_manual(values=c("blue", "red"))+
    theme_classic()
  
  plot_beh_list[[key]] <- p
  i = i+1
  if (export_data==TRUE){
    write.csv(dfcoef,paste0(stats_dir,"plot_data_neural_accuracy_beh_accuracy_",name,".csv"),row.names=F)
  }
}

# View plots together using patchwork
plot_beh_combined <- wrap_plots(plotlist = plot_beh_list)
#print(plot_beh_combined)
ggsave(paste0(fig_dir,"combined_beh_plot.pdf"), plot_beh_combined, width= 10, height=9, dpi = 300)

df_rt_list <- list()
plot_rt_list <- list()

export_data <- TRUE
#create reaction time plots
j = 1 #counter for name_list
for (key in list_names) {
  name <- name_list[j]
  model <- model_list[[key]][[3]]
  
  #eff <- effect("rt", model)
  eff <- effect('rt', model, xlevels = data.frame(rt=c(500, 1500, 2500, 3500)))
  df <- as.data.frame(eff)
  #df<- df[df$rt>100,] 
  df <- df[order(df$rt)] # Sort the data frame by rt in ascending order
  df_rt_list[[key]] <- df
  #summary(eff)
  rtcoef <- coef(model)
  
  df$rt <- as.character(df$rt)
  
  rt1 <- exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*500)/(1+exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*500))
  rt2 <- exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*1500)/(1+exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*1500))
  rt3 <- exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*2500)/(1+exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*2500))
  rt4 <- exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*3500)/(1+exp(rtcoef$sub$`(Intercept)`+rtcoef$sub$rt*3500))
  dfrt <- data.frame('neur_accuracy' = round(c(rt1, rt2, rt3, rt4),3),
                       'subject' = c(row.names(rtcoef$sub), row.names(rtcoef$sub), row.names(rtcoef$sub), row.names(rtcoef$sub)),
                       'rt' = as.character(c(rep(df_rt_list[[1]][[1]][[1]], length(rt1)), rep(df_rt_list[[1]][[1]][[2]], length(rt2)), rep(df_rt_list[[1]][[1]][[3]], length(rt3)), rep(df_rt_list[[1]][[1]][[4]], length(rt4)))))

  p <- ggplot(df, aes(rt, fit)) +  
    geom_point(aes(x=rt, y=neur_accuracy),data=dfrt, alpha=0.4) +
    geom_line(aes(x=rt, y=neur_accuracy, group=subject),data=dfrt, alpha=0.2)+
    geom_violin(aes(x= rt, y= neur_accuracy, fill = rt), data=dfrt, alpha=0.5, show.legend = FALSE)+
    geom_boxplot(aes(x= rt, y= neur_accuracy, group=rt), data=dfrt,alpha=0.5, width=0.05, fill='white') +
    geom_hline(yintercept = 0.5)+
    coord_cartesian(ylim = c(0.1, 0.9))+
    labs(title= name, x="Reaction Time (ms)", y = "Neural Accuracy")+
    scale_fill_brewer(palette="Blues")+
    guides(fill=guide_legend(title="Reaction Time (ms)"))+
    scale_x_discrete(limits = c("500", "1500", "2500", "3500"))+
    theme_classic() 
  
  plot_rt_list[[key]] <- p 
  j = j+1
  if (export_data==TRUE){
    write.csv(dfcoef,paste0(stats_dir,"plot_data_neural_accuracy_RT_",name,".csv"),row.names=F)
  }
}

plot_rt_combined <- wrap_plots(plotlist = plot_rt_list)
#print(plot_rt_combined)
ggsave(paste0(fig_dir,"combined_rt_plot.pdf"), plot_rt_combined, width= 10, height=9, dpi = 300)

#print the summaries
for (list in model_list){
  print(summary(list[[2]]))
  print(summary(list[[3]]))
}
