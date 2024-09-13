library(ggplot2)
library(lme4)
library(lmerTest)
library(effects)
library(broom.mixed)
library(patchwork)

#scripts_dir = 'Z:/Desktop/MINDLAB2020_MEG-AuditoryPatternRecognition/scripts/working_memory/'
scripts_dir = '/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scripts/working_memory/'
setwd(scripts_dir)

log_dir <- "../../scratch/working_memory/data_sharing/data/logs"
fig_dir <- "../../scratch/working_memory/results/figures"
stats_dir <- "../../scratch/working_memory/results/stats"
dem_dir <- "../../scratch/working_memory/data_sharing/data/demographics.csv"
bnames <- c('maintenance','manipulation')
lnames <- c('recognize','invert')

## Load demographics
dem <- read.table(dem_dir,sep=',',header=T)

## Load data:
subjects <- 11:90
scount <- 0
d <- c()
for (sub in subjects){
  scount <- scount + 1
  for (b in 1:length(lnames)){
    ln <- lnames[b]
    dfile <- sprintf('%s/%04.f_%s_MEG.csv',log_dir,sub,ln)
    print(dfile)
    d0 <- read.table(dfile,sep=',',header=T)
    
    d0$subject = as.factor(as.character(d0$subject))
    d0$block <- bnames[b]
    if (scount == 1 & b == 1){
      d <- d0
    }else{
      d <- rbind(d,d0)
    }
  }
}

# Fix subject code:
d$subject <- gsub("bis", "", d$subject)
d$subject <- as.numeric(as.character(d$subject))

#Change maintenance condition name
d$block <- factor(gsub("maintenance",'recall',d$block),levels=c('recall','manipulation'))

##Calculate accuracy
d$accuracy <- as.numeric(d$type == d$response)

## Aggregate data
d2 <- aggregate(d$accuracy, by=list(d$subject,d$block), mean)
colnames(d2) <- c('subject','block','accuracy')
write.csv(d2,'../../misc/WM_accuracies.csv',row.names=FALSE)

## Plot accuracy without exclusions
ap <- ggplot(data=d2, aes(block,accuracy, fill = block)) +
  geom_point(alpha=.1) +
  geom_line(alpha = .1, aes(group=subject)) +
  geom_violin(alpha = .1) +
  scale_fill_manual(values=c('blue','red'))+
  theme_bw()+
  theme(legend.position="none")
print(ap)
ggsave(paste0(fig_dir,'/behavioral_accuracy.pdf'), ap, dpi= 300, width=10, height=10,units='cm')

# exclude subjects
exc <- unique(d2[d2$accuracy <= 0.5,'subject'])
exc <- c(exc,15,32,33)
print(exc)
dexc <- d[!(d$subject %in% exc),]
dexc2 <- d2[!(d2$subject %in% exc),]
demexc <- dem[!(dem$Subject %in% exc),]
dexc2$accuracy <- round(dexc2$accuracy,3)

# Export data for figure
write.csv(dexc2,paste0(stats_dir,'/plot_data_behavioral_accuracy_exc.csv'),row.names=F)

# Plot accuracy after exclusion
ap <- ggplot(data=dexc2, aes(block,accuracy, fill = block)) +
  geom_point(alpha=.1) +
  geom_line(alpha = .1, aes(group=subject)) +
  geom_violin(alpha = .1) +
  scale_fill_manual(values=c('blue','red'))+
  ylim(c(0.4,1)) +
  geom_hline(yintercept=0.5)+
  theme_bw()+
  theme(legend.position="none")
print(ap)
ggsave(paste0(fig_dir,'/behavioral_accuracy_exc.pdf'), ap, dpi= 300, width=10, height=10,units='cm')

# Test effect of block
dexc$block <- factor(dexc$block,levels=c('manipulation','recall'))
mnull <- glmer(accuracy~1 + (1|subject), data = dexc, family = 'binomial')
m0 <- glmer(accuracy~block + (1+block|subject), data = dexc, family = 'binomial')
summary(m0)

# Likelihood ratio
anova(mnull,m0)

# Effects
block_eff <- as.data.frame(effect("block",m0))
print(block_eff)
tidy(m0,conf.int=TRUE,exponentiate=TRUE,effects="fixed")

# Summarize some demographics
table(demexc$Sex)
mean(demexc$Age,na.rm=T)
sd(demexc$Age,na.rm=T)
demexc$yomt[is.nan(demexc$yomt)] <- 0
mean(demexc$yomt)
mean(demexc$TrainingGMSI,na.rm=T)
sd(demexc$TrainingGMSI,na.rm=T)

median(demexc$TrainingGMSI,na.rm=T)
table(demexc$yomt!=0)
median(demexc$yomt[demexc$yomt!=0],na.rm=T)
summary(demexc$yomt[demexc$yomt!=0])#,na.rm=T))

median(demexc$vividness,na.rm=T)
summary(demexc$vividness)

# Summarize vividness
viv_count <- table(demexc$vividness)
viv_prop <- round(viv_count/nrow(demexc),2)
perc <- sum(viv_prop[names(viv_prop)>=0])

print(viv_count)
print(viv_prop)
print(perc)

# Add accuracies to demographics dataset after exclussions
demexc$recall_accuracy <- dexc2$accuracy[dexc2$block=='recall']
demexc$manip_accuracy <- dexc2$accuracy[dexc2$block=='manipulation']
demexc_round <- demexc
demexc_round[,c('recall_accuracy','manip_accuracy')] <- round(demexc_round[,c('recall_accuracy','manip_accuracy')],3) 

# Export data for figure
write.csv(demexc_round,paste0(stats_dir,'/plot_data_dem_behavioral_accuracy_exc.csv'),row.names=F)

# Plot correlations with music training and working memory
predictor_list <- c("WM", "TrainingGMSI")
category_list <- c("recall_accuracy", "manip_accuracy")
pdf_names <- c("WM_plot.pdf", "TrainingGMSI_plot.pdf")
xlabels <- c("Working Memory", "Training GMSI")
ylabels <- c("Recall Accuracy", "Manipulation Accuracy")

# Loop over variables and plot
for (i in seq_along(predictor_list)){
  new_models <- list()
  new_correlations <- numeric(length(category_list))
  new_p_values <- numeric(length(category_list))
  new_degrees_of_freedom <- numeric(length(category_list))
  new_plot_list <- list()
  for (j in seq_along(category_list)){
    new_plot_list[[j]] <- local({
      j <- j 
      i <- i 
      predictor_var <- predictor_list[[i]] 
      response_var <- category_list[[j]]
      xlabel <- xlabels[[i]]
      ylabel <- ylabels[[j]]

      # Fit linear model
      model <- lm(formula(paste(response_var, "~", predictor_var)), data = demexc)
      new_models[[j]] <- model
      
      # Calculate correlation coefficient
      new_correlations[j] <- cor(demexc[[predictor_var]], demexc[[response_var]], use = "complete.obs")
      
      # Extract p-value and degrees of freedom
      model_summary <- summary(model)
      new_p_values[j] <- model_summary$coefficients[predictor_var, "Pr(>|t|)"]
      new_degrees_of_freedom[j] <- model_summary$df[[2]]
      
      # Plot
      ylim <- if (j > 2) c(0, 0.2) else c(0.5, 1.05)
      plot <- ggplot(demexc, aes(x = demexc[[predictor_var]], y = demexc[[response_var]])) +
      geom_point() +
      geom_smooth(method = "lm", se = FALSE) +  # Add linear model line without standard error band
      labs(title= paste("r(", new_degrees_of_freedom[j], ") = ", round(new_correlations[j], 2), ", p = " ,format(new_p_values[j], scientific = FALSE, digits = 2)), x = xlabel, y = ylabel)+
      coord_cartesian(ylim = ylim)+
      theme_bw()
    })
  }
  
  title_text <- paste("Combined Plot -", xlabels[[i]])
  new_plot_combined <- wrap_plots(plotlist = new_plot_list) +
    plot_annotation(title = title_text, theme = theme(plot.title = element_text(size = 16)))
  print(new_plot_combined)
  ggsave(paste0(fig_dir, '/', pdf_names[[i]]), new_plot_combined, width= 14, height=9, dpi = 300)
}


