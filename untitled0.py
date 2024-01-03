import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
print("las librerías hansido instaladas")

sns.set_style("darkgrid")

df=pd.read_csv("HR_comma_sep.csv")
df.head()

df.info()
df.shape

#Renombrando columnas
df.rename(columns={"Work_accident":"work_accident",
                   "average_montly_hours":"average_monthly_hours",
                   "time_spend_company":"tenure",
                   "Department":"department"}, inplace=True)
df.isnull().sum()
df.describe().T
#Verificando valores duplicados
df.duplicated().sum()
df[(df["department"]=="sales") & (df["salary"]=="medium") & (df["satisfaction_level"]==0.37)][df.duplicated()]
#Elimincamos las filas duplicadas 
df=df.drop_duplicates()
df.head()

#Create functions to visualize the data
def countplot(column_name, rotation=0):
    """
    column_name must be categorical
    """
    plt.figure(figsize=(15,6))
    sns.countplot(data=df, x=column_name)
    plt.title(f"{column_name}")
    plt.xticks(rotation=rotation)
    plt.show()

def pieplot(column_name):
    """
    column_name must be categorical with 2 or 3 cat
    """
    plt.pie(df[column_name].value_counts(), labels=df[column_name].value_counts().index, autopct="%1.2f%%", shadow=True)
    plt.show()

def boxplot(column_name, bins=0):
    """
    column_name must be numerical
    """
    fig,ax=plt.subplots(1,2, figsize=(15,6))
    sns.boxplot(data=df, x=column_name, ax=ax[0])
    sns.histplot(data=df, x=column_name, ax=ax[1], bins=bins)
    ax[0].set_ylabel(column_name)
    ax[0].set_xlabel("FRQ")
    ax[1].set_ylabel(column_name)
    plt.show()
    
df.columns.tolist()
df["satisfaction_level"].nunique()
boxplot("satisfaction_level", 4)
boxplot("last_evaluation", 4)
df["number_project"].unique()
countplot("number_project")
boxplot("average_monthly_hours", 4)

df["tenure"].unique()
countplot("tenure")
countplot("left")
countplot("department")

#Definimos funciones para las visualizaciones
def count_plot(column_name, hue=None, rotation=0):
    plt.figure(figsize=(15,6))
    sns.countplot(data=df, x=column_name, hue=hue, dodge=True)
    plt.title(f"{column_name}")
    plt.show()
    
def scatter(column_name_x, column_name_y, hue=None):
    plt.figure(figsize=(15,4))
    sns.scatterplot(data=df, x=column_name_x, y=column_name_y, hue=hue, alpha=.4)
    plt.legend(loc="best")
    plt.title(f"{hue} by {column_name_x} and {column_name_y}")
    plt.show()
    
def boxplot(column_name_x, column_name_y, hue=None):
    plt.figure(figsize=(15,6))
    sns.boxplot(data=df, x=column_name_x, y=column_name_y, hue=hue)
    plt.legend(loc="best")
    plt.title(f"Boxplot entre {column_name_x}, {column_name_y} y {hue}")
    plt.show()
    
    
count_plot("department", "left")
boxplot("tenure", "satisfaction_level", "left")
scatter("department", "average_monthly_hours", "left")
scatter("average_monthly_hours", "satisfaction_level", "left")
scatter("average_monthly_hours", "number_project", "left")
scatter("average_monthly_hours", "last_evaluation", "left")
scatter("average_monthly_hours", "promotion_last_5years", "left")
#Heatmap correlation
var_num=df.describe().columns.tolist()
plt.figure(figsize=(15,6))
sns.heatmap(data=df[var_num].corr(), annot=True, cmap="vlag")
plt.title("Correlación de variables")
plt.show()
#Creando columna
df["overworked"]=df["average_monthly_hours"].apply(lambda x: 1 if x > 175 else 0)
df.head()
#Limpiando outliers
Q1 = df["tenure"].quantile(0.25)
Q3 = df["tenure"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df["tenure"]=df["tenure"].apply(lambda x: lower_limit if x < lower_limit else x)
df["tenure"]=df["tenure"].apply(lambda x: upper_limit if x > upper_limit else x)

df.drop(columns=["average_monthly_hours"], inplace=True)

#Codificando variable Salary 0 = low, 1 = medium, 2=high
df["salary"]=df["salary"].apply(lambda x: 0 if x == "low" else (1 if x == "medium" else (2 if x == "high" else x)))
df.head()

#dummies for departmen column
df=pd.get_dummies(data=df, columns=["department"], dtype=int)

#Realimos split antes de escalar las variables numéricas
y = df["left"]
X = df.drop(columns="left")
var_to_stand = ["satisfaction_level", "last_evaluation", "number_project", "tenure"]

#split
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.25, random_state=0)

#Aplicamos escalado a las variables numéricas de X_train y X_test
scaler=StandardScaler()
X_train[var_to_stand] = scaler.fit_transform(X_train[var_to_stand])
X_test[var_to_stand] = scaler.transform(X_test[var_to_stand])

#Iniciamos el modelo
dt = DecisionTreeClassifier(random_state=0)

#Definimos los hyperparametros
cv_params = {"max_depth":[2,4,6,8,10,20,30,40,None],
            "min_samples_leaf":[2,5,10,20,50],
            "min_samples_split":[2,4,6,8,10]}

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

dt_cv = GridSearchCV(dt, cv_params, scoring=scoring, cv=5, refit="roc_auc")


