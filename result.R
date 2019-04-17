library(dplyr)
library(data.table)
library(reshape2)
library(reshape)
library(ggplot2)
library(gridExtra)
library(grid)


# Comate with image paramter ----------------------------------------------
namelist <- c("10","20","30","41","62","83","104")
named <- c("0.1","0.2","0.3","0.4","0.6","0.8","1.0")
datalist = list()
count=1

for (i in namelist) {
  assign(paste("result1_",i,sep = ""),read.csv(paste("method1_",i,".txt",sep=""),header=FALSE))
  assign(paste("result1_",i,sep=""), 
         'names<-'(get(paste("result1_",i,sep="")),
                   c("method","whole","experiments","r1","r2",
                     "r3","r4","r5","average","time")), globalenv())
  assign(paste0("df1_",i),get(paste("result1_",i,sep=""))%>%
           dplyr::group_by(method,whole)%>%
           dplyr::summarise(mean=mean(average),
                                       sd=sd(average)))
  assign(paste0("result1_",i), '[[<-'(get(paste0("result1_",i)), 'image', value = named[count]))
  datalist[[i]] <- get(paste0("result1_",i))
  count=count+1

  } 
com <-  do.call(rbind, datalist)
com$image = factor(com$image,levels = named)

df_com <- com %>% 
  dplyr::group_by(method,image) %>% 
  dplyr::summarise(mean=mean(average),
            sd=sd(average))

datalist = list()
count=1
for (i in namelist) {
  assign(paste("result2_",i,sep = ""),read.csv(paste("method2_",i,".txt",sep=""),header=FALSE))
  assign(paste("result2_",i,sep=""), 
         'names<-'(get(paste("result2_",i,sep="")),
                   c("method","whole","experiments","r1","r2",
                     "r3","r4","r5","average","time")), globalenv())
  assign(paste0("df2_",i),get(paste("result2_",i,sep=""))%>%
           dplyr::group_by(method,whole)%>%
           dplyr::summarise(mean=mean(average),
                     sd=sd(average)))
  assign(paste0("result2_",i), '[[<-'(get(paste0("result2_",i)), 'image', value = named[count]))
  datalist[[i]] <- get(paste0("result2_",i))
  count=count+1
  
} 

com_2 <-  do.call(rbind, datalist)
com_2$image = factor(com_2$image,levels = named)
df_com_2 <- com_2 %>% 
  dplyr::group_by(method,image) %>% 
  dplyr::summarise(mean=mean(average),
            sd=sd(average))



# change the variable from ?Äú‚Äù‚Äù‚Äù‚Äùhere?Äù‚Äù‚Äù‚Äù‚Ä?" --------------------------------
df=com_2
df2=df_com_2
# here here here up up up -------------------------------------------------
ggplot(df,aes(x=image,y=average))+
  geom_boxplot()+
  geom_point(data = df2,aes(y=mean,x=image),shape=2,size=3,color="blue")+
  geom_text(data = df2, aes( label=round(mean,2),y = mean+0.63),color="blue")+
  theme_bw()+
  theme(axis.title.x = element_text(size = rel(1.7)),
        axis.title.y = element_text(size = rel(1.7)),
        axis.text.x = element_text(hjust = 1, size=13),
        axis.text.y = element_text(hjust = 1, size=15),
        plot.title = element_text(size=rel(1.7),hjust = 0.5),
        strip.text = element_text(size=rel(1.7)))+
  ggtitle("The experiment result of all method")+
  ylab("Error rate")+
  xlab("Method")+
  facet_wrap(~method, nrow = 1)

ggplot(df2,aes(x=image,y=mean,group = method))+
  geom_point(aes(color=method))+
  geom_line(aes(color=method),size=0.7)+
  theme_bw()+
  theme(axis.title.x = element_text(size = rel(1.7)),
        axis.title.y = element_text(size = rel(1.7)),
        axis.text.x = element_text(hjust = 1, size=13),
        axis.text.y = element_text(hjust = 1, size=15),
        plot.title = element_text(size=rel(1.7),hjust = 0.5),
        legend.position="none",
        strip.text = element_text(size=rel(1.7)))+
  ylab("Error rate")+
  xlab(expression(theta))+
  facet_wrap(~method,nrow = 1)



# check time --------------------------------------------------------------

df3 <- com %>%
  group_by(method,image) %>%
  summarise(t_mean=mean(time),
            a_mean=mean(average),
            t_sd=sd(time),
            a_sd=sd(average))

df4 <- com_2 %>%
  group_by(method,image) %>%
  summarise(t_mean=mean(time),
            a_mean=mean(average),
            t_sd=sd(time),
            a_sd=sd(average))

df3$Aggregation_Method  = "Overlaying Method"
df4$Aggregation_Method = "Appending Method"

df5 <- rbind(df3,df4)


ggplot(df5,aes(x=image,y=t_mean,group=Aggregation_Method))+
  geom_point(aes(shape=Aggregation_Method),size=3)+
  geom_line()+
  facet_wrap(~method,nrow = 1)

ggplot(df5,aes(x=image,y=a_mean,group=Aggregation_Method))+
  geom_point(aes(shape=Aggregation_Method),size=3)+
  geom_line()+
  facet_wrap(~method,nrow = 1)


# Compare with different Method -------------------------------------------


result <- read.csv("method1_104.txt",header=FALSE)
names(result) <- c("method","whole","experiments",
                   "r1","r2","r3","r4","r5","average","time")

result <- result[result$whole=="True",]
result$method <- as.character(result$method)


df <- result %>% 
  dplyr::group_by(method) %>% 
  dplyr::summarise(mean=mean(average),
            sd=sd(average))


result2 <- read.csv("method2_104.txt",header=FALSE)
names(result2) <- c("method","whole","experiments",
                   "r1","r2","r3","r4","r5","average","time")

result2 <- result2[result2$whole=="True",]
result2$method <- as.character(result2$method)


df2 <- result2 %>% 
  dplyr::group_by(method,whole) %>% 
  dplyr::summarise(mean=mean(average),
            sd=sd(average))


ggplot(result2,aes(x=method,y=average))+
  geom_boxplot()

result$Aggregation_Method  = "Overlaying Method"
result2$Aggregation_Method = "Appending Method"

com_resul <- rbind(result,result2)
com_resul$Aggregation_Method = factor(com_resul$Aggregation_Method , 
                                      levels = c("Overlaying Method", "Appending Method"))


df3 <- com_resul %>% 
  dplyr::group_by(method,Aggregation_Method) %>% 
  dplyr::summarise(a_mean=mean(average),
            t_mean=mean(time),
            sd=sd(average))
  


ggplot(com_resul,aes(x=method,y=average))+
  geom_boxplot(aes(color=method),size=1.2)+
  geom_point(data = df3,aes(y=a_mean,x=method),shape=2,size=3,color="blue")+
  geom_text(data = df3, aes( label=round(a_mean,2),y = a_mean+0.8),color="blue",size=7)+
  theme_bw()+
  theme(axis.title.x = element_text(size = rel(1.7)),
        axis.title.y = element_text(size = rel(1.7)),
        axis.text.x = element_text(hjust = 1, size = rel(2.2)),
        axis.text.y = element_text(hjust = 1, size = rel(1.8)),
        plot.title = element_text(size=rel(1.7),hjust = 0.5),
        legend.position="none",
        strip.text = element_text(size=rel(1.7)))+
  ylab("Error rate")+
  xlab("Method")+
  facet_wrap(~Aggregation_Method, nrow = 1)

ggplot(com_resul,aes(x=method,y=time))+
  geom_boxplot(aes(color=method),size=1.2)+
  geom_point(data = df3,aes(y=t_mean,x=method),shape=2,size=3,color="blue")+
  geom_text(data = df3, aes( label=round(t_mean,2),y = t_mean+100),color="blue",size=7)+
  theme_bw()+
  theme(axis.title.x = element_text(size = rel(1.7)),
        axis.title.y = element_text(size = rel(1.7)),
        axis.text.x = element_text(hjust = 1, size = rel(2.2)),
        axis.text.y = element_text(hjust = 1, size = rel(1.8)),
        plot.title = element_text(size=rel(1.7),hjust = 0.5),
        legend.position="none",
        strip.text = element_text(size=rel(1.7)))+
  ylab("Execution time ")+
  xlab("Method")+
  facet_wrap(~Aggregation_Method, nrow = 1)




# Compare with the rank ---------------------------------------------------

Rank <- read.csv("rank.txt",header=FALSE)
names(Rank) <- c("method","whole","experiments",
                   "r1","r2","r3","r4","r5","average")

Rank <- Rank[Rank$whole=="True",]

df_rank <- Rank %>% 
  group_by(method,whole) %>% 
  summarise(mean=mean(average),
            sd=sd(average))

SoilSciGuylabs <- c("Ascending", "Descending", "Random")


ggplot(Rank,aes(x=method,y=average))+
  geom_boxplot(color="#00BCD8")+
  geom_point(data = df_rank,aes(y=mean,x=method),shape=2,size=3,color="blue")+
  geom_text(data = df_rank, aes( label=round(mean,2),y = mean+0.1),color="blue",size=7)+
  theme_bw()+
  theme(axis.title.x = element_text(size = rel(1.7)),
        axis.title.y = element_text(size = rel(1.7)),
        axis.text.x = element_text(hjust = 1, size = rel(2.2)),
        axis.text.y = element_text(hjust = 1, size = rel(1.8)),
        plot.title = element_text(size=rel(1.7),hjust = 0.5),
        strip.text = element_text(size=rel(1.7)))+
  ggtitle("The experiment result of different Rank")+
  ylab("Error rate")+
  xlab("Rank")+
  scale_x_discrete(labels= SoilSciGuylabs)

qqnorm(Rank$average[Rank$method=="random"]);qqline(Rank$average[Rank$method=="random"], col = 2)

qqplot.data <- function (vec,value) # argument: vector of numbers
{
  # following four lines from base R's qqline()
  y <- quantile(vec[!is.na(vec)], c(0.25, 0.75))
  x <- qnorm(c(0.25, 0.75))
  slope <- diff(y)/diff(x)
  int <- y[1L] - slope * x[1L]
  
  d <- data.frame(resids = vec)
  
  ggplot(d, aes(sample = resids)) + 
    stat_qq() + 
    geom_abline(slope = slope, intercept = int)+
    xlab(value)+
    theme_bw()+
    theme(axis.title.x = element_text(size = rel(1.7)))
}
p1 <- qqplot.data(Rank$average[Rank$method=="a"],"Ascending")
p2 <- qqplot.data(Rank$average[Rank$method=="d"],"Descending")
p3 <- qqplot.data(Rank$average[Rank$method=="random"],"Random")

grid.arrange(p1, p2,p3, nrow = 1,
             top=textGrob("The qqplot of the order method",gp=gpar(fontsize=20,font=3)))


ggplot(Rank[Rank$method=="random",], aes(sample = average))+
  stat_qq()+
  stat_qq_line()

wilcox.test(Rank$average[Rank$method=="a"],Rank$average[Rank$method=="d"], paired=TRUE,exact = TRUE)
test<-wilcox.test(Rank$average[Rank$method=="a"],Rank$average[Rank$method=="d"], paired=TRUE,exact = TRUE)

wilcox.test(Rank$average[Rank$method=="a"],Rank$average[Rank$method=="random"], paired=TRUE,exact = TRUE)
test<-wilcox.test(Rank$average[Rank$method=="a"],Rank$average[Rank$method=="random"], paired=TRUE,exact = TRUE)

wilcox.test(Rank$average[Rank$method=="d"],Rank$average[Rank$method=="random"], paired=TRUE,exact = TRUE)
test<-wilcox.test(Rank$average[Rank$method=="d"],Rank$average[Rank$method=="random"], paired=TRUE,exact = TRUE)

