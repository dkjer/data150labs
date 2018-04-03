from sklearn.linear_model import LogisticRegression
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.interactive(True)

default = pd.read_csv("default1.csv")
default.head()
"""
sns.set(font_scale=2)
g1 = sns.lmplot('balance', 'income', data=default.sample(1000), hue='default', markers = ['+', 'o'], fit_reg=False)
g1.fig.set_size_inches(12,6)
plt.ylim([0,70000])
plt.xlim([-100,3500])
plt.figure()
"""
sns.set(font_scale=2)
sns.boxplot(x='default', y='balance', data=default, hue='default')
plt.figure()
sns.boxplot(x='default', y='income', data=default, hue='default')

default['default'] = default['default'].replace(['Yes'],1)
default['default'] = default['default'].replace(['No'],0)
default['student'] = default['student'].replace(['Yes'],1)
default['student'] = default['student'].replace(['No'],0)

# Qualitative logistic regression
lr = LogisticRegression()
X = np.column_stack((np.ones(len(default)), default['student']))
y = default['default']
lr.fit(X, y)
print("What is the association between being a student and the probability of default? Predictor is student (yes) and target is default.")
print("Intercept", lr.intercept_)
print("The beta_1 coefficient for student (yes) is ", lr.coef_[0,1], ". As a positive number, it indicates that students have a higher probability of default than non-students.")
print("Model Score is ", lr.score(X, y))
# Predicting default probabilities
# Probabilities for each of the classes, No Default and Yes Default, for students, returned by .predict_proba in order
# predict default for students
print("Probabilities of No Default and Yes Default, in that order, for a student: ", lr.predict_proba([[1,1]]) )
print("What is the probability of default given that the individual is a student? ", lr.predict_proba([[1, 1]])[0,1]) #
# predict default for non-student
##### CHANGE THE CODE AS NEEDED HERE. You may want to use two steps.
# predict_all_default_classes_non_student = lr.predict_proba([[?,?]])
# predict_for_non_student =
print("What is the probability of default given that the individual is NOT a student? ", lr.predict_proba([[1, 0]])[0,1]) #
print("\n")

# Multivariable logistic regression
lr = LogisticRegression()
X = np.column_stack((np.ones(len(default)), default['balance'], default['student']))
y = default['default']
lr.fit(X, y)
print("What is the association between balance and being a student and the probability of default?")
print("Model Score is ", lr.score(X, y))
print("The beta_1 coefficient for predictor balance is ", lr.coef_[0,1])
print("The beta_2 coefficient for predictor student (yes) is ", lr.coef_[0,2], ". As a negative number, it indicates \n    that students have a lower probability of default than non-students when the balance is taken into account.")
# Predicting default probabilities
# predict default for a student with a balance of $1500
print("What is the probability of default for a student with a balance of $1500? ", lr.predict_proba([[1, 1500,1]])[0,1])
# predict default for a NON-student with a balance of $1500
print("What is the probability of default for a non-student with a balance of $1500? ", lr.predict_proba([[1, 1500,0]])[0,1])
print("\n")


plt.show(block=True)
