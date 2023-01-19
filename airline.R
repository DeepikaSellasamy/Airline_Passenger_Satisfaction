#Loading the required packages
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caretEnsemble)
library(doParallel)
library(corrplot)
library(dplyr)
library(e1071)
library(mlr)
library(caTools)
library(tidyr)
library(MLmetrics)
library(ggmosaic)
library(kableExtra)
library(lmtest)
library(car)
library(xgboost)

#reading the data
setwd("C:/Users/DEEPIKA/Downloads")
airline=read.csv("airline.csv")

summary(airline)
str(airline_train)

#checking missing values
table(is.na(airline))
is.na(airline_train)
table(is.na(airline$Arrival.Delay.in.Minutes))

#filling missing values with mean
airline$Arrival.Delay.in.Minutes[is.na(airline$Arrival.Delay.in.Minutes)] <- mean(airline$Arrival.Delay.in.Minutes,na.rm=T)

airline %>% mutate(total_delay =Departure.Delay.in.Minutes+Arrival.Delay.in.Minutes )

#checking outliers
bx=boxplot(airline$Flight.Distance)

outliers = which(airline$Flight.Distance <3800)
airline[outliers,"Flight.Distance"]

levels(airline$satisfaction) <- c("satisfied", "dissatisfied")

airline <- airline %>% 
  mutate(Customer.Type = as.factor(case_when(Customer.Type == "disloyal Customer" ~ "0",
                                             TRUE ~ "1")))

#Data visualization
#ratio of satisfaction

ggplot(airline,aes(factor(satisfaction)))+geom_histogram(stat="count")

#Customer type ratio
#No.of loyal customer is higher than disloyal 

ggplot(airline,aes(factor(Customer.Type)))+geom_histogram(stat="count")

#satisfaction with customer type
#But for the loyal customer also dissatisfaction level is high

ggplot(airline,aes(Customer.Type,fill=factor(satisfaction)))+geom_histogram(stat="count")

#satisfaction with travel type
#Business travel peoples are more satisfied than personal travel

ggplot(airline,aes(Type.of.Travel,fill=factor(satisfaction)))+geom_histogram(stat="count")

#satisfaction based on  gender
#Gender doesn't play an important role in satisfaction men and women seems equally 

ggplot(airline,aes(Gender,fill=factor(satisfaction)))+geom_histogram(stat="count")

#satisfaction with class
#Business class peoples are more satisfied and the least is eco plus
ggplot(airline,aes(Class,fill=factor(satisfaction)))+geom_histogram(stat="count")

#Satisfaction with Gate location
ggplot(airline,aes(factor(satisfaction),Gate.location))+geom_bar(stat="identity")

#satisfaction with delay
#1 customer is satisfied even after delay of 1300 minutes

ggplot(airline,aes(Arrival.Delay.in.Minutes,Departure.Delay.in.Minutes,colour=factor(satisfaction)))+
  geom_point(stat="identity",position="dodge")+scale_colour_manual(
    values = c("1" = "blue","0"="green"))

#satisfaction with flight distance
#most passengers are okay with delay in departure when the distance is longer

ggplot(airline,aes(Flight.Distance,Departure.Delay.in.Minutes,colour=factor(satisfaction)))+
  geom_point(stat="identity",position="dodge")+scale_colour_manual(
    values = c("1" = "blue","0"="green"))

#In business class higher no of dissatisfied passenger when baggage handling is <4
#And for other class even baggage handling is in good range but still passenger dissatisfied

ggplot(airline,aes(Gate.location,Baggage.handling,fill=factor(satisfaction)))+geom_boxplot()+facet_grid(~Class)

#satisfaction comparing with online boarding and arrival time

ggplot(airline,aes(Departure.Arrival.time.convenient,Online.boarding,fill=factor(satisfaction)))+geom_boxplot()+facet_grid(~Class)

#In the below all cases satisfied passengers are belong to rating 4 and 5

ggplot(airline,aes(Seat.comfort,fill=factor(satisfaction)))+geom_histogram(stat="count")

ggplot(airline,aes(Cleanliness,fill=factor(satisfaction)))+geom_histogram(stat="count")

ggplot(airline,aes(Ease.of.Online.booking,fill=factor(satisfaction)))+geom_histogram(stat="count")

#data visualization for each numerical variables

ggplot(gather(airline %>% select_if(is.numeric)), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

#data visualization for each categorical variables

ggplot(gather(airline %>% select_if(is.factor)), aes(value)) + 
  geom_bar(bins = 10,fill="firebrick") + 
  facet_wrap(~key, scales = 'free_x') + labs(x="Categorical",
                                             y="Value")

#Data exploration

airline %>% 
  group_by(Type.of.Travel) %>% filter(Gender=='Female') %>% summarise(s=mean(satisfaction==1))

airline %>% 
  group_by(Type.of.Travel) %>% filter(Gender=='Male') %>% summarise(s=mean(satisfaction==1))

airline %>% 
  filter(Arrival.Delay.in.Minutes<=30) %>% summarise(s=mean(satisfaction==0))

airline %>% 
  filter(Departure.Delay.in.Minutes>=30) %>% summarise(s=mean(satisfaction==0))


#Insights from Gender data
airline %>% 
   filter(Gender=='Male') %>% summarise(s=mean(satisfaction==0))

airline %>% 
  filter(Gender=='Female') %>% summarise(s=mean(satisfaction==0))

#Insights from customer type:0-disloyal and 1-loyal
airline %>% 
  filter(Customer.Type=='0') %>% summarise(s=mean(satisfaction==0))

airline %>% 
  filter(Customer.Type=='1') %>% summarise(s=mean(satisfaction==0))

#Insights from travel type
airline %>% 
  filter(Type.of.Travel=='Business travel') %>% summarise(s=mean(satisfaction==0))

airline %>% 
  filter(Type.of.Travel=='Personal Travel') %>% summarise(s=mean(satisfaction==0))

#Insights from class
airline %>% 
  filter(Class=='Business') %>% summarise(s=mean(satisfaction==0))

airline %>% 
  filter(Class=='Eco') %>% summarise(s=mean(satisfaction==0))

airline %>% 
  filter(Class=='Eco Plus') %>% summarise(s=mean(satisfaction==0))


#checking correlation between variables

cor1 <- airline %>%
  select(-Gender,-Customer.Type,-Type.of.Travel,-Class)
correlationMatrix =cor(cor1)
cor(correlationMatrix)
corrplot(correlationMatrix)
#In the above Ease of Online booking is highly correlated with Inflight wifi service.
#Inflight service is highly correlated with Baggage_handling

highlyCorrelated = findCorrelation(correlationMatrix, cutoff=0.6)
highlyCorrelated

#below features have a low correlation with target variable
cor(airline$Age,airline$satisfaction)
cor(airline$Gate.location,airline$satisfaction)
cor(airline$Flight.Distance,airline$satisfaction)
cor(airline$Departure.Arrival.time.convenient,airline$satisfaction)
cor(airline$Departure.Delay.in.Minutes,airline$satisfaction)
cor(airline$Arrival.Delay.in.Minutes,airline$satisfaction)

#removing less important features

airline= airline %>%
  select(-c("Unnamed","Gender","id","Age","Flight.Distance","Departure.Arrival.time.convenient","Departure.Delay.in.Minutes",
            "Arrival.Delay.in.Minutes"))
summary(airline)
str(airline)

airline$Gender=as.factor(airline$Gender)
airline$Customer.Type=as.factor(airline$Customer.Type)
airline$Class=as.factor(airline$Class)
airline$Type.of.Travel=as.factor(airline$Type.of.Travel)
airline$satisfaction=as.factor(airline$satisfaction)

#Splitting data
ran = createDataPartition(airline$satisfaction, 
                          p = 0.7,                         
                          list = FALSE)

ran
airline_train=airline[ran,]
airline_test=airline[-ran,]

X = airline[, -17]
y = airline[, 17]

Xtrain = X[ran, ]
Xtest = X[-ran, ]
ytrain = y[ran]
ytest = y[-ran]

#F1
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred,
                                y_true = data$obs,
                                positive = lev[1])
  c(F1 = f1_val)
}

# Create trainControl 
Control = trainControl(
  method = "repeatedcv", 
  number = 3, 
  summaryFunction = f1,
  classProbs = FALSE, 
  verboseIter = TRUE,
  savePredictions = "final",
)

#model building for navies bayes and random forest
set.seed(456)
model=caretList(Xtrain,ytrain,trControl = Control,
                methodList=c("nb","rf"),
                tuneList=NULL,
                continue_on_fail=FALSE,
                metric="F1",
                preProcess=c("center","scale"))

model$nb
model$rf

#summarize F1 of the models
output = resamples(model)
summary(output)
dotplot(output)

#predictions on test data
predNB = predict.train(model$nb, newdata = Xtest)
predRF = predict.train(model$rf, newdata = Xtest)

#checking the predictions on target variable
model_NB = table(ytest,predNB)
model_RF = table(ytest,predRF)

#accuracy
sum(diag(model_NB)/sum(model_NB))
sum(diag(model_RF)/sum(model_RF))

#error
1-sum(diag(model_NB)/sum(model_NB))
1-sum(diag(model_RF)/sum(model_RF))

#Logistic regression model
model_logreg <-  glm(formula = satisfaction ~ ., 
                     data = airline_train,
                     family = binomial("logit"))
summary(model_logreg)

vif(model_logreg)

#Prediction
log_prob <-  predict(model_logreg,
                     newdata = airline_test,
                     type = "response")

log_label <-  as.factor(ifelse(log_prob > 0.5,
                               yes = "1",
                               no = "0"))

#confusion matrix
cm_log <- confusionMatrix(data = log_label,
                          reference = airline_test$satisfaction,
                          positive = "1")
cm_log

#XG Boost model
#convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(as.matrix(sapply(Xtrain, as.numeric)), label=ytrain)
xgboost_test = xgb.DMatrix(as.matrix(sapply(Xtest, as.numeric)), label=ytest)

#XG boost model building
model_xg <- xgboost(data = xgboost_train,                    
                    max.depth=3,                          
                    nrounds=50)  

#Prediction
predxg = predict(model, xgboost_test)

#Convert prediction to factor type
pred_y = as.factor((levels(ytest))[round(predxg)])

#create confusion matrix
conf_mat = confusionMatrix(ytest, pred_y)
conf_mat







