# Practical Machine Learning - Prediction Assignment
Afsheen Rajendran  
March 27, 2016  

# Summary

The goal of the assignment was to predict exercise quality for some observations taken 
from the 'Qualitative Activity Recognition of Weight Lifting Exercises' study. 
Different classification techniques were used on the given training set to fit models. 
The random forest technique was able to predict with 100% accuracy 
on the given testing set. The out of sample error on the subset of training
set used for validation was 0.45%.




# Background

The 'Qualitative Activity Recognition of Weight Lifting Exercises' study
(http://groupware.les.inf.puc-rio.br/har) aimed to investigate "how (well)"
an activity was performed by a test subject. The "how (well)" investigation 
can potentially provide useful information for a large variety of applications, 
such as sports training.

Six young health participants were asked to perform one set of 10 repetitions 
of the Unilateral Dumbbell Biceps Curl in five different fashions: 
exactly according to the specification (Class A), throwing the elbows to the front (Class B), 
lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) 
and throwing the hips to the front (Class E).

Based on the measurements taken during the repetitions, the investigators tried
to predict how well the participants were performing and to provide feedback on
improving their technique.

# Dataset and Feature Selection



The dataset contains values for 160 variables corresponding to the readings 
collected from sensors in the belt, arm, forearm and dumbbell.
After removing the columns with NA or missing values, 55 variables were left. 

The process of rejection of many variables was made easier by the fact that
those variables were NA or missing in the testing dataset given to validate
our models. If the testing set was not provided or if a different testing set
was provided, the feature selection process might have produced more variables
to be analyzed.

The last column in the dataset is 'classe', a categorical variable that takes 
on one of the five values (class A to class E) described in the previous section.
The following 54 variables will be used to predict the 'classe' variable.


```r
names(mydata)[1:54]
```

```
##  [1] "user_name"            "num_window"           "roll_belt"           
##  [4] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [7] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
## [10] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [13] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [16] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [19] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [22] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [25] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [28] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [31] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [34] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [37] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [40] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [43] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [46] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [49] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [52] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"
```

The following table gives a distribution of the 'classe' values
tabulated for the different users. We see that for each user  
most of the 'classe' values fall under 'Class A'. But there are no obvious
patterns with respect to the other classes.


```r
table(mydata [,c("user_name", "classe")])
```

```
##           classe
## user_name     A    B    C    D    E
##   adelmo   1165  776  750  515  686
##   carlitos  834  690  493  486  609
##   charles   899  745  539  642  711
##   eurico    865  592  489  582  542
##   jeremy   1177  489  652  522  562
##   pedro     640  505  499  469  497
```

# Cross Validation

The number of observations in the given training set is 19622. But the
number of observations in the testing set (against which the accuracy
of any fitted model will be measured) is just 20. So we need to use
parts of the training set itself for validation during the training
process to avoid overfitting the data and to reduce out of sample error.

K-fold cross validation will be used to designate subsets of the training set
that will be used for validation of models before they are used on the testing set.
It is general practice to use folds of 5 or 10. It was decided to use
K-fold validation with k=5 for this assignment.



```r
set.seed(11715) 
trainIndex <- createDataPartition(mydata$classe,p=.60,list=FALSE) 
training <- mydata[trainIndex,] 
testing <- mydata[-trainIndex,]

## cross-validation using 5-folds
trainingControl <- trainControl(method="cv",number=5,allowParallel=TRUE)
```

# Model Selection

Since the response variable is a qualitative (categorical) variable,
we need to use model fitting approaches that pertain to 'classification'
instead of 'regression'.

Commonly used classification approaches are linear discriminant analysis,
classification trees and random forests. If we are not able to achieve satisfactory
results with these approaches, we need to try using more complex approaches
like bagging and boosting. Complex approaches might be more flexible, but we might lose out on
interpretability on the resulting models.

The following sections will go over the accuracy of the different
classification approaches used on this dataset.

## Linear Discriminant Analysis


```r
ldaFit <- train(classe ~ .,data=training, method="lda",trControl=trainingControl)
ldaPred <- predict(ldaFit, training)
```

From the confusion matrix given below, we can see that the accuracy 
of the model built using LDA is 75%. It is able to predict observations
into class A about 86% of the time. The other classes are predicted
with lesser accuracy.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2847  337  187  125   81
##          B   95 1498  178   70  244
##          C  158  286 1414  221  132
##          D  246   84  245 1483  178
##          E    2   74   30   31 1530
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7449          
##                  95% CI : (0.7369, 0.7528)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6769          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8504   0.6573   0.6884   0.7684   0.7067
## Specificity            0.9134   0.9382   0.9180   0.9235   0.9857
## Pos Pred Value         0.7959   0.7185   0.6395   0.6632   0.9178
## Neg Pred Value         0.9389   0.9194   0.9331   0.9531   0.9372
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2418   0.1272   0.1201   0.1259   0.1299
## Detection Prevalence   0.3038   0.1771   0.1878   0.1899   0.1416
## Balanced Accuracy      0.8819   0.7977   0.8032   0.8460   0.8462
```


## Classification Trees (Decision Trees)


```r
rpartFit <- train(classe ~ .,data=training, method="rpart",trControl=trainingControl)
rpartPred <- predict(rpartFit, training)
```

From the confusion matrix given below, we can see that the accuracy 
of the model built using classification is less than 50%. It is able
to correctly predict 'Class A' 90% of the time. But the accuracy for 
the other classes is poor.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3045  960  967  863  305
##          B   48  784   60  357  292
##          C  244  535 1027  710  586
##          D    0    0    0    0    0
##          E   11    0    0    0  982
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4958          
##                  95% CI : (0.4867, 0.5048)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3408          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9095  0.34401  0.50000   0.0000  0.45358
## Specificity            0.6328  0.92029  0.78657   1.0000  0.99886
## Pos Pred Value         0.4959  0.50876  0.33108      NaN  0.98892
## Neg Pred Value         0.9462  0.85393  0.88160   0.8361  0.89029
## Prevalence             0.2843  0.19353  0.17442   0.1639  0.18385
## Detection Rate         0.2586  0.06658  0.08721   0.0000  0.08339
## Detection Prevalence   0.5214  0.13086  0.26342   0.0000  0.08432
## Balanced Accuracy      0.7711  0.63215  0.64328   0.5000  0.72622
```


## Random Forests


```r
rfFit <- train(classe ~ .,data=training, method="rf",trControl=trainingControl)
rfPred <- predict(rfFit, training)
rfFit
```

```
## Random Forest 
## 
## 11776 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9420, 9422, 9421, 9420 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9931221  0.9912993  0.001680113  0.002125416
##   30    0.9968583  0.9960260  0.001065227  0.001347434
##   58    0.9926118  0.9906547  0.004012940  0.005075012
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 30.
```

The randomForest uses different number of predictors in its attempts to
find the most accurate model. From the output above, we can see that for
the optimal model the number of predictors (mtry) used was 30.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

From the confusion matrix given above, we can see that the accuracy 
of the model built using classification is 100% for all classes.

# Out of Sample Error

Among the different approaches on the training set, the 'random forests' technique
was the most accurate. Using that model to classify the cases in the testing set,
we obtain the accuracy as 99.55%. So the out of sample error is 0.45%.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    9    0    0    0
##          B    0 1509    4    0    3
##          C    0    0 1364   13    0
##          D    0    0    0 1272    4
##          E    1    0    0    1 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9938, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9941   0.9971   0.9891   0.9951
## Specificity            0.9984   0.9989   0.9980   0.9994   0.9997
## Pos Pred Value         0.9960   0.9954   0.9906   0.9969   0.9986
## Neg Pred Value         0.9998   0.9986   0.9994   0.9979   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1923   0.1738   0.1621   0.1829
## Detection Prevalence   0.2855   0.1932   0.1755   0.1626   0.1832
## Balanced Accuracy      0.9990   0.9965   0.9975   0.9943   0.9974
```

# Final Predictions
Applying the model fitted using the random forests approach on the given testing set,
we get the predictions below.


```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

This list of answers was entered on the associated quiz and was found to be 100% correct.


# Appendix

All the code snippets used above have been printed out here using the knitr 'echo=TRUE' option.


```r
library(caret)
library(AppliedPredictiveModeling)
library(readr)
library(dplyr)
library(randomForest)
library(MASS)
```


```r
rawData <- read_csv("pml-training.csv")
```

```
## Warning: 32646 problems parsing 'pml-training.csv'. See problems(...) for
## more details.
```

```r
mydata <- rawData[, c(2, 7, 8:10, 11, 37:45, 46:48, 49, 60:68, 84:86, 102, 113:121, 122:124, 140, 151:159, 160)]
mydata$classe <- as.factor(mydata$classe)
mydata[, c(2:54)] <- sapply(mydata[, c(2:54)], as.numeric)

finalRawData <- read_csv("pml-testing.csv")
finalData <- finalRawData[, c(2, 7, 8:10, 11, 37:45, 46:48, 49, 60:68, 84:86, 102, 113:121, 122:124, 140, 151:159)]
finalData[, c(2:54)] <- sapply(finalData[, c(2:54)], as.numeric)
```


```r
names(mydata)[1:54]
```

```
##  [1] "user_name"            "num_window"           "roll_belt"           
##  [4] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [7] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
## [10] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [13] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [16] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [19] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [22] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [25] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [28] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [31] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [34] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [37] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [40] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [43] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [46] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [49] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [52] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"
```


```r
table(mydata [,c("user_name", "classe")])
```

```
##           classe
## user_name     A    B    C    D    E
##   adelmo   1165  776  750  515  686
##   carlitos  834  690  493  486  609
##   charles   899  745  539  642  711
##   eurico    865  592  489  582  542
##   jeremy   1177  489  652  522  562
##   pedro     640  505  499  469  497
```


```r
set.seed(11715) 
trainIndex <- createDataPartition(mydata$classe,p=.60,list=FALSE) 
training <- mydata[trainIndex,] 
testing <- mydata[-trainIndex,]

## cross-validation using 5-folds
trainingControl <- trainControl(method="cv",number=5,allowParallel=TRUE)
```


```r
ldaFit <- train(classe ~ .,data=training, method="lda", trControl=trainingControl)
ldaPred <- predict(ldaFit, training)
```


```r
confusionMatrix(ldaPred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2847  337  187  125   81
##          B   95 1498  178   70  244
##          C  158  286 1414  221  132
##          D  246   84  245 1483  178
##          E    2   74   30   31 1530
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7449          
##                  95% CI : (0.7369, 0.7528)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6769          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8504   0.6573   0.6884   0.7684   0.7067
## Specificity            0.9134   0.9382   0.9180   0.9235   0.9857
## Pos Pred Value         0.7959   0.7185   0.6395   0.6632   0.9178
## Neg Pred Value         0.9389   0.9194   0.9331   0.9531   0.9372
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2418   0.1272   0.1201   0.1259   0.1299
## Detection Prevalence   0.3038   0.1771   0.1878   0.1899   0.1416
## Balanced Accuracy      0.8819   0.7977   0.8032   0.8460   0.8462
```



```r
rpartFit <- train(classe ~ .,data=training, method="rpart", trControl=trainingControl)
rpartPred <- predict(rpartFit, training)
```


```r
confusionMatrix(rpartPred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3045  960  967  863  305
##          B   48  784   60  357  292
##          C  244  535 1027  710  586
##          D    0    0    0    0    0
##          E   11    0    0    0  982
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4958          
##                  95% CI : (0.4867, 0.5048)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3408          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9095  0.34401  0.50000   0.0000  0.45358
## Specificity            0.6328  0.92029  0.78657   1.0000  0.99886
## Pos Pred Value         0.4959  0.50876  0.33108      NaN  0.98892
## Neg Pred Value         0.9462  0.85393  0.88160   0.8361  0.89029
## Prevalence             0.2843  0.19353  0.17442   0.1639  0.18385
## Detection Rate         0.2586  0.06658  0.08721   0.0000  0.08339
## Detection Prevalence   0.5214  0.13086  0.26342   0.0000  0.08432
## Balanced Accuracy      0.7711  0.63215  0.64328   0.5000  0.72622
```



```r
rfFit <- train(classe ~ .,data=training, method="rf", trControl=trainingControl)
rfPred <- predict(rfFit, training)
rfFit
```

```
## Random Forest 
## 
## 11776 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9420, 9422, 9421, 9420 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9931221  0.9912993  0.001680113  0.002125416
##   30    0.9968583  0.9960260  0.001065227  0.001347434
##   58    0.9926118  0.9906547  0.004012940  0.005075012
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 30.
```

The randomForest uses different number of predictors in its attempts to
find the most accurate model. From the output above, we can see that for
the optimal model the number of predictors (mtry) used was 30.


```r
confusionMatrix(rfPred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```



```r
rfPredTest <- predict(rfFit, testing)
confusionMatrix(rfPredTest, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    9    0    0    0
##          B    0 1509    4    0    3
##          C    0    0 1364   13    0
##          D    0    0    0 1272    4
##          E    1    0    0    1 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9938, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9941   0.9971   0.9891   0.9951
## Specificity            0.9984   0.9989   0.9980   0.9994   0.9997
## Pos Pred Value         0.9960   0.9954   0.9906   0.9969   0.9986
## Neg Pred Value         0.9998   0.9986   0.9994   0.9979   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1923   0.1738   0.1621   0.1829
## Detection Prevalence   0.2855   0.1932   0.1755   0.1626   0.1832
## Balanced Accuracy      0.9990   0.9965   0.9975   0.9943   0.9974
```


```r
rfPredFinal <- predict(rfFit, finalData) 
rfPredFinal
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```




