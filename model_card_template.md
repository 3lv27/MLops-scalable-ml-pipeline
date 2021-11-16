# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
**Author:** Elvin Gomez (starter code by Udacity)  
**Date:** November 16, 2021,  
**Model Type:**  Gradient Boosting Trees  
**Model Implemented:** Histogram-based Gradient Boosting Classification Tree  
**Resources:** This implementation is inspired by
[LightGBM](https://github.com/Microsoft/LightGBM)  
**Contact info:** 3lv27.dev@gmail.com

## Intended Use
Primary Intended Use:  
- This model was developed with purpose of the Udacity Machine Learning DevOps Nano Degree's project 3.

Second Intended Use:
- The model could be used as a baseline to compare against other models

Out-of-scope Uses:
- Not for large-scale datasets as FastAPI limit number of requests sent to the pipeline
- Not for enterprise usage

## Training Data
Census Income Data Set: https://archive.ics.uci.edu/ml/datasets/census+income

> Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

```
- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
- salary: >50K, <=50K.
```

## Evaluation Data
20% of the data is used for evaluation.

## Metrics
Metrics used for model evaluation:  
- **Precision:** calculated as (number of true positives) / (number of all labels with positive predictions)  
- **Recall:** calculated as (number of true positives) / (number of all labels with real positive )  
- **Fbeta:** the weighted harmonic mean of precision and recall; in this model, weight on recall = 0.7 and weight on precision = 0.3

## Ethical Considerations
Uses all features from the census dataset including race and sex.


## Caveats and Recommendations
The data set was donated to the UCI Machine Learning Repository in 1996. So the data points are more than 25 years old.  
The data set is somewhat imbalanced with approximately 25% of labels >50K and 75%% <=650K.
