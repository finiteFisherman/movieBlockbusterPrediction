# Project - Movies!

#1. Logistic Regression: 

library(caret)
library(neuralnet)
library(e1071)



df = read.csv("C:/Users/dsl89/Documents/spring24/machineLearning/project/Movies.csv")

df$Blockbuster=ifelse(df$Trailer_Views > 462460,1,0)    # if trailer views greater than median of sorted values
t(t(df[1,]))    #view row names

# Partition data first
train_index<-sample(rownames(df), dim(df)[1]*0.6)
valid_index<-setdiff(rownames(df),train_index)
train_data<-df[train_index, ]
valid_data<-df[valid_index, ]

#Normalize
norm_values <- preProcess(df, method="range")
train_norm_df <- predict(norm_values, train_data)
valid_norm_df <- predict(norm_values, valid_data)


# Run logit regression model on the training data
mymodel<-glm(Blockbuster~Marketing_Expense+Production_Expense+Budget+Movie_Length+Lead_.Actor_Rating+
               Lead_Actress_Rating+Director_Rating+Producer_Rating+Twitter_Hashtags+Avg_Age_Actors, data=train_norm_df, family="binomial")   # all non categorical data

summary(mymodel)  ## regression results

# Predict using the validation data
predicted_values<- predict(mymodel, type="response", newdata=valid_norm_df)

### Confusion matrix
confusionMatrix(relevel(as.factor(ifelse(predicted_values>0.5,1,0)),"1"),
                relevel(as.factor(valid_data$Blockbuster),"1"))

#2.  ANN
# normalize
norm.values <- preProcess(df, method="range")
train.norm.df <- predict(norm.values, train_data)
valid.norm.df <- predict(norm.values, valid_data)


# build a neural net 2/3 size of input + output  or 8 or 9
# use hidden= with a vector of integers specifying number of hidden nodes in each layer
nn <- neuralnet(Blockbuster~Marketing_Expense+Production_Expense+Budget+Movie_Length+Lead_.Actor_Rating+
                   Lead_Actress_Rating+Director_Rating+Producer_Rating+Twitter_Hashtags
                +Avg_Age_Actors, data = train.norm.df, linear.output=FALSE, hidden = 8)
plot(nn)

# confusion matrix for validation data
pred <- compute(nn, valid.norm.df)
confusionMatrix(relevel(as.factor(ifelse(pred$net.result>0.5, "1", "0")),ref="1"), 
                relevel(as.factor(valid_data$Blockbuster),ref="1"))


# build a neural net 2/3 size of input + output  or 8 or 9
# use hidden= with a vector of integers specifying number of hidden nodes in each layer
nn2 <- neuralnet(Blockbuster~Marketing_Expense+Production_Expense+Budget+Movie_Length+Lead_.Actor_Rating+
                   Lead_Actress_Rating+Director_Rating+Producer_Rating+Twitter_Hashtags
                 +Avg_Age_Actors+Trailer_Views, data = train.norm.df, linear.output=FALSE, hidden = 9)
plot(nn2)

# confusion matrix for validation data
pred <- compute(nn2, valid.norm.df)
confusionMatrix(relevel(as.factor(ifelse(pred$net.result>0.5, "1", "0")),ref="1"), 
                relevel(as.factor(valid_data$Blockbuster),ref="1"))


library(e1071)
library(caret)
library(forecast)

# Support Vector Machine (SVM)

df2<-df
# Since we are doing classification SVM now, we first convert the outcomes to factor
df2$Blockbuster<-as.factor(df2$Blockbuster)
df2$X3D_Available<-as.factor(df2$X3D_Available)
df2$Genre<-as.factor(df2$Genre)

# partition the data  
train_index<-sample(rownames(df2), dim(df2)[1]*0.6)
valid_index<-setdiff(rownames(df2),train_index)
train_data<-df2[train_index, ]
valid_data<-df2[valid_index, ]


# support vector regression
#Include all categorical data as well
svm1 <- svm(Blockbuster~Marketing_Expense+Production_Expense+Budget+Movie_Length+Lead_.Actor_Rating+
              Lead_Actress_Rating+Director_Rating+Producer_Rating+Twitter_Hashtags
            +Avg_Age_Actors+ X3D_Available+ Genre, data = train_data)

prediction1<-predict(svm1, valid_data)
# RMSE and other measures
accuracy(prediction1, valid_data$Blockbuster)
print(prediction1)

#Confusion Matrix
confusionMatrix(as.factor(prediction1), as.factor(valid_data$Blockbuster))


