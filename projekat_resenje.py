import pandas as pd
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("heart.csv")
categorical_features = data.select_dtypes(include=['object']).columns
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
numerical_features = list(numerical_features)
numerical_features.remove('HeartDisease')

print('\n')
q0 = input("Prikaz provere da li potoje NULL ondosno nepostojece vrednosti? (ukucati y/n)")
if q0 == 'y':
    print(data.info())
    print("\n")
    print(data.isnull().sum())

#####   konverzija stringova u brojeve
for i in data.select_dtypes(include=['object']).columns:
    data[i] = LabelEncoder().fit_transform(data[i])

q0_1 = input("Prikaz zavisnosti ciljnog atributa i nezavisnih promenljivih? (ukucati y/n)")
if q0_1 == "y":
    atributi = ['Age', 'Oldpeak', 'MaxHR', 'RestingBP', 'Cholesterol']
    for atribut in atributi:
        plt.figure()
        sns.histplot(data=data, x=atribut, hue='HeartDisease', kde=True)
        plt.title(f'Distribucija: {atribut} u zavisnosti od HeartDisease')
        plt.show()

####    ANOMALIJE (OUTLIERS)
####    Prikaz podataka pre obrade:
num_atr = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
q1 = input("Prikaz podataka PRE uklanjanja anomalija? (ukucati y/n)")
if q1 == "y":
    for atr in num_atr:
        sns.boxplot(data[atr]).set(title=atr)
        plt.show()
####    Uklanjanje anomalija IQR tehnikom
for atr in num_atr:
    Q1 = data[atr].quantile(0.25)
    Q3 = data[atr].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data = data[(data[atr] > lower) & (data[atr] < upper)]
####    Prikaz podataka nakon obrade:
q2 = input("Prikaz podataka NAKON uklanjanja anomalija? (ukucati y/n)")
if q2 == "y":
    for atr in num_atr:
        sns.boxplot(data[atr]).set(title=atr)
        plt.show()

####    KORELACIJE
q3 = input("Prikaz KORELACIJA? (ukucati y/n)")
if q3 == "y":
    corr = data.corrwith(data['HeartDisease']).sort_values(ascending = False).to_frame()
    corr.columns = ['Correlations']
    plt.subplots(figsize = (5,5))
    sns.heatmap(corr,annot = True,linewidths = 0.4,linecolor = 'black');
    plt.title('Korelacije u odnosu na HeartDisease');
    plt.show()

####    TESTOVI ZA RELEVANTNE ATRIBUTE
print("\n")
q4 = input("Prikaz Chi Squared Score-a? (ukucati y/n)")
if q4 == "y":
    features = data.loc[:,categorical_features[:-1]]
    target = data.loc[:,categorical_features[-1]]

    best_features = SelectKBest(score_func = chi2,k = 'all')
    fit = best_features.fit(features,target)

    featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['Chi Squared Score'])

    plt.subplots(figsize = (5,5))
    sns.heatmap(featureScores.sort_values(ascending = False,by = 'Chi Squared Score'),annot = True,linewidths = 0.4,linecolor = 'black',fmt = '.2f');
    plt.title('Selection of Categorical Features');
    plt.show()

q5 = input("Prikaz ANOVA Score-a? (ukucati y/n)")
if q5 == "y":
    features = data.loc[:,numerical_features]
    target = data.loc[:,categorical_features[-1]]

    best_features = SelectKBest(score_func = f_classif,k = 'all')
    fit = best_features.fit(features,target)

    featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['ANOVA Score'])

    plt.subplots(figsize = (5,5))
    sns.heatmap(featureScores.sort_values(ascending = False,by = 'ANOVA Score'),annot = True,linewidths = 0.4,linecolor = 'black',fmt = '.2f');
    plt.title('Selection of Numerical Features');
    plt.show()

####    PRAVLJENJE MODELA, PODESAVANJE HIPERPARAMETARA I UNAKRSNA VALIDACIJA; UPOREDJIVANJE SA MODELIMA SA NAJBITNIJIM PARAMETRIMA
X_svi = data[data.columns.drop(['HeartDisease'])].values
X_najbitniji = data[data.columns.drop(['HeartDisease','RestingECG','Cholesterol', 'RestingBP'])].values
y = data['HeartDisease'].values
for i in range(2):
    if i == 0:
        X_train, X_test, y_train, y_test = train_test_split(X_svi, y, test_size=0.3, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_najbitniji, y, test_size=0.3, stratify=y)

    models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]

    for model in models:
        name = model.__class__.__name__
        if name == "LogisticRegression":
            hiperparametri = {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 1, 5, 10, 25, 50, 100],
                'solver': ['liblinear', 'saga']
            }
        if name == "KNeighborsClassifier":
            hiperparametri = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        if name == "DecisionTreeClassifier":
            hiperparametri = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }

        grid = GridSearchCV(model, hiperparametri, cv=10)
        grid.fit(X_train, y_train)

        model.set_params(**grid.best_params_)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        if i==0:
            print("Rezultati za algoritam " + name + "\n")
        else:
            print("Rezultati za algoritam " + name + " nakon primene SelectKBest algoritma\n")
        print("Accuracy score: ", "%.2f" % (accuracy_score(y_test, y_predict) * 100), "%")
        print("Precision score: ", "%.2f" % (precision_score(y_test, y_predict) * 100), "%")
        print("Recall score: ", "%.2f" % (recall_score(y_test, y_predict) * 100), "%")
        print("F1 score: ", "%.2f" % (f1_score(y_test, y_predict) * 100), "%")
        print("Confusion matrix: \n", confusion_matrix(y_test, y_predict))
