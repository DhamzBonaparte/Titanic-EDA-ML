# importing required libraries
import pandas as pd
import seaborn as sns
import numpy as np


# loading the csv file and conveerting the sex to 0 (male) and 1 (female)
def rename(sex):
    if sex == "male":
        return 0
    else:
        return 1


dfmodel = pd.read_csv("data/tested.csv", converters={"Sex": rename})
df = pd.read_csv("data/tested.csv")

# viewing the dataframe
df

# installin the pandas_profiiling to see what tthe relations are between the variables
# !pip install pandas_profiling

# NOTE: pandas_profiling is deprecated and may not work on Python 3.13+
# Install ydata-profiling (supported for Python 3.7–3.13) instead:
#     pip install ydata-profiling
# Then update the import line to:
#     from ydata_profiling import ProfileReport

# from pandas_profiling import ProfileReport
# prof=ProfileReport(df)
# prof.to_file(output_file='titanic.html')

# Univariate EDA (Exploratory Data Analysis)
sns.countplot(x=df["Survived"])

print(df.shape)
df.sample(5)

df.info()

df.isnull().sum()
df.describe()
df.duplicated().sum()
df[["Age", "Fare"]].corr()

sns.countplot(df.sample(15)["Sex"])

df["Sex"].value_counts().plot(kind="pie", autopct="%.2f")

import matplotlib.pyplot as plt

# plt.hist(df['Age'])
# sns.boxplot(df[['Age','Fare']])
sns.kdeplot(df["Age"])
plt.savefig("results/age_distribution.png")
plt.close()

# Bivariate and multivariate EDA
sns.scatterplot(x="Age", y="Fare", data=df, hue="Sex", style="Pclass", size="Survived")
plt.savefig("results/age_fare_scatter.png")
plt.close()

sns.barplot(x="Age", y="Fare", data=df.sample(5), hue="Survived")
plt.savefig("results/age_fare_bar.png")
plt.close()

# sns.kdeplot(df[df["Age"] > 20]["Age"])

a = pd.crosstab(df["Pclass"], df["Survived"])
sns.heatmap(a)
plt.savefig("results/pclass_survival_heatmap.png")
plt.close()

sns.clustermap(a)
plt.savefig("results/pclass_survival_clustermap.png")
plt.close()

sns.pairplot(df)
plt.savefig("results/pairplot.png")
plt.close()

sns.histplot(data=df, x="Age", hue="Survived", bins=10, kde=True)
plt.savefig("results/age_survival_hist.png")
plt.close()

# filling NAN values and getting the reuired columns only
x = dfmodel.iloc[:, [4, 5]]
x.iloc[:, 1] = x.iloc[:, 1].fillna(x.iloc[:, 1].mean())
y = dfmodel["Survived"]

# splitng the dataframe nto 30% test set and 70% tran set
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)
print(xTrain.shape)
print(xTest.shape)

# checking null values
x.isnull().sum()

# using Standardization
# scales using mean and standard deviation sych that mean = 0 and standard deviation = 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(xTrain)
xTrainScaled = scaler.transform(xTrain)
xTestScaled = scaler.transform(xTest)

# turning the scaled data to dataframe to check the scaled values, mean and standard deviaton
xTrainScaled_df = pd.DataFrame(xTrainScaled, columns=["Age", "Sex"])
np.round(xTrainScaled_df.describe())

# usng logstic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(xTrainScaled, yTrain)
prediction = lr.predict(xTestScaled)

# Accuracy Testing
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(yTest, prediction))
print(classification_report(yTest, prediction))
