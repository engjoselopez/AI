# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:21:56 2024

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import operator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import itertools
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn import tree
import seaborn as sns

from IPython.display import Image

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    muestra y grafica matriz de confusión.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def visualize_tree(tr, feature_names):
    """crea un arbol de decisiones.
    """
    
    with open("dt.dot", 'w') as f:
        tree.export_graphviz(tr, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    
##
# =============================================================================
# Data
# =============================================================================
##

header_row = ['age','sex','chest_pain','blood pressure','serum_cholestoral','fasting_blood_sugar',\
               'electrocardiographic','max_heart_rate','induced_angina','ST_depression','slope','vessels','thal','diagnosis']

# read csv file with Cleveland heart diseases data
heart = pd.read_csv('processed.cleveland.data.csv', names=header_row)
heart[:5]

# =============================================================================
# Preprocesamiento de Datos 
# =============================================================================

for c in heart.columns[:-1]:
    heart[c] = heart[c].apply(lambda x: heart[heart[c]!='?'][c].astype(float).mean() if x == "?" else x)
    heart[c] = heart[c].astype(float)
    
set(heart.loc[:, "diagnosis"].values)
    
vecs_1 = heart[heart["diagnosis"] == 1 ].median().values[:-2]
vecs_2 = heart[heart["diagnosis"] == 2 ].median().values[:-2]
vecs_3 = heart[heart["diagnosis"] == 3 ].median().values[:-2]
vecs_4 = heart[heart["diagnosis"] == 4 ].median().values[:-2]

sim = {"(1,2)": np.linalg.norm(vecs_1-vecs_2), \
       "(1,3)": np.linalg.norm(vecs_1-vecs_3),\
       "(1,4)": np.linalg.norm(vecs_1-vecs_4),\
       "(2,3)": np.linalg.norm(vecs_2-vecs_3),\
       "(2,4)": np.linalg.norm(vecs_2-vecs_4),\
       "(3,4)": np.linalg.norm(vecs_3-vecs_4)    
      }
    
sorted_sim = sorted(sim.items(), key=operator.itemgetter(1))
sorted_sim

## Dolor de pecho
heart_d = heart[heart["diagnosis"] >= 1 ]
heart_d[:5]

# heart_d.groupby(["diagnosis", ])["age"].min().astype(str) + ', ' +  heart_d.groupby(["diagnosis", ])["age"].max().astype(str)
# heart_d.groupby(["diagnosis", ])["age"].mean()
# heart_d.groupby(["diagnosis", "sex"])["age"].count()
# heart_d.groupby(["diagnosis", "chest_pain"])["age"].count()

##Presión Arteerial
heart_d.groupby(["diagnosis"])["blood pressure"].min().astype(str) + ', ' +  heart_d.groupby(["diagnosis"])["blood pressure"].max().astype(str)

#Colesterol
heart_d.groupby(["diagnosis"])["serum_cholestoral"].min().astype(str) + ', ' +  heart_d.groupby(["diagnosis"])["serum_cholestoral"].max().astype(str)

#Glicemia en ayunas
heart_d.groupby(["diagnosis", "fasting_blood_sugar"])["age"].count()

#Resultados de electrocardiograma
heart_d.groupby(["diagnosis", "electrocardiographic"])["age"].count()

#Frecuencia Cardiaca máxima
heart_d.groupby(["diagnosis"])["max_heart_rate"].min().astype(str) + ', ' +  heart_d.groupby(["diagnosis"])["max_heart_rate"].max().astype(str)

# =============================================================================
# Preprocesamiento de datos
# =============================================================================

heart.loc[:, "diag_int"] = heart.loc[:, "diagnosis"].apply(lambda x: 1 if x >= 1 else 0)

#Normalización de datos

preprocessing.Normalizer().fit_transform(heart)

#Construcción del set de muestra
heart_train, heart_test, goal_train, goal_test = train_test_split(heart.loc[:,'age':'thal'], \
                                                 heart.loc[:,'diag_int'], test_size=0.33, random_state=0)
#Descomponiendo los sets de atributos

corr = heart.corr()
heart.corr()

cmap = sns.diverging_palette(250, 10, n=3, as_cmap=True)
    
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_table_styles(magnify())    

#Construir el set de entrenamiento y estimar el LSS

loss = ["hinge", "log_loss"]
penalty = ["l1", "l2"]
alpha = [0.1, 0.05, 0.01]
n_iter = [500, 1000]

best_score = 0
best_param = (0,0,0,0)
for l in loss:
    for p in penalty:
        for a in alpha:
            for n in n_iter:
                lss = SGDClassifier(loss=l, penalty=p, alpha=a, max_iter=n)
                lss.fit(heart_train, goal_train)
                scores= cross_val_score(lss, heart.loc[:,'age':'thal'], heart.loc[:,'diag_int'])
                
                if np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_param = (l,p,a,n)

lss_best = SGDClassifier(alpha=a, fit_intercept=True, loss=l, max_iter=500, penalty=p)
lss_best.fit(heart_train, goal_train)

#Evaluación del modelo

cnf_matrix = confusion_matrix(goal_test, lss_best.predict(heart_test))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["Con Enfermedad Cardiaca", "Sin enfermedad Cardiaca"],
                      title='Confusion matrix, Sin normalización')
plt.show()

#Métricas de rendimiento

scores = ['accuracy', 'f1', 'precision', 'recall']

metrics = {score: cross_val_score(lss_best,heart_test, goal_test, scoring=score).mean() for score in scores}

#Si lo deseas puedes hacer:
#metrics

test_df = pd.DataFrame(heart_test, columns = header_row[:-1])

test_df.loc[:, "Disease_probability"] = [x[1] for x in lss_best.predict_proba(heart_test)]
test_df.to_excel("disease_probability.xlsx", index = False)
test_df[:5]

# Coeficientes

w = lss_best.coef_[0]
a = -w[0] / w[1]

coeff_df = pd.DataFrame(columns = ['X_k', 'coeff'])
for c in range(len(heart.loc[:,'age':'thal'].columns)):
    coeff_df.loc[len(coeff_df)] = [heart.loc[:,'age':'thal'].columns[c], w[c]]
    
coeff_df

# =============================================================================
# Árbol de Decisiones
# =============================================================================

best_score_dt = 0

criterion = ['gini', 'entropy']

for c in criterion:             

            clf = tree.DecisionTreeClassifier(criterion=c)

            clf.fit(heart_train, goal_train)
            print("Decision tree Cross-Validation scores:")
            scores = cross_val_score(clf, heart.loc[:,'age':'thal'], heart.loc[:,'diag_int'], cv=10)
            print (scores)
            print("Mean Decision tree Cross-Validation score = ", np.mean(scores))

            if np.mean(scores) > best_score_dt:
                best_score_dt = np.mean(scores)
                best_param_dt = (c)
                    
    
print("The best parameters for model are ", best_param_dt)
print("The Cross-Validation score = ", best_score_dt)

#Desarrollo del modelo

lss_best_dt = tree.DecisionTreeClassifier(criterion = 'entropy')
lss_best_dt.fit(heart_train, goal_train)
print("Decision tree Test score:")
print(lss_best_dt.score(heart_test, goal_test))

# Matriz de confusión
cnf_matrix = confusion_matrix(goal_test, lss_best_dt.predict(heart_test))
np.set_printoptions(precision=2)

# Matriz de confusión no normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["Heart disease", "No heart disease"],
                      title='Confusion matrix, without normalization')
plt.show()

visualize_tree(lss_best_dt, heart.loc[:,'age':'thal'].columns)

from IPython.display import Image  
import pydotplus
dot_data = tree.export_graphviz(lss_best_dt)
graph = pydotplus.graphviz.graph_from_dot_file("dt.dot")
graph.write_pdf("dt.pdf")
Image(graph.create_png())

#Probabilidades para cada x_s

w = lss_best_dt.feature_importances_

prob_df = pd.DataFrame(columns = ['X_k', 'P(X_k)'])
for c in range(len(heart.loc[:,'age':'thal'].columns)):
    prob_df.loc[len(prob_df)] = [heart.loc[:,'age':'thal'].columns[c], w[c]]
    
prob_df

print ("Sum of dependent probabilities = " , prob_df["P(X_k)"].sum())

#Visualización a través de un pie chart
# prob_df.index = prob_df["X_k"].values
# group_names = prob_df["X_k"].values
# counts = pd.Series(prob_df["X_k"].values,prob_df["P(X_k)"].values)

# explode = (0, 0.1, 0.2, 0.25, 0.3, 0.35, 0)
# colors =  ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'violet', 'pink', 'orange', 'red']

# prob_df.plot(kind='pie', fontsize=17, figsize=(8, 7), autopct='%1.1f%%', subplots=True)
# plt.axis('equal')
# plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), prop={'size':12})
# plt.show()

# =============================================================================
# -combinación de ambos modelos
# =============================================================================

def data_parsing(path):
    header_row = ['age','sex','chest_pain','blood pressure','serum_cholestoral','fasting_blood_sugar',\
               'electrocardiographic','max_heart_rate','induced_angina','ST_depression','slope','vessels','thal','diagnosis']


    heart = pd.read_csv(path, names=header_row)
    

    for c in heart.columns[:-1]:
        heart[c] = heart[c].apply(lambda x: heart[heart[c]!='?'][c].astype(float).mean() if x == "?" else x)
        heart[c] = heart[c].astype(float)
        

    heart.loc[:, "diag_int"] = heart.loc[:, "diagnosis"].apply(lambda x: 1 if x >= 1 else 0)
    
    return heart


def subset_decomposition(data):
    # Dividir el dataset en test y train 
    
    heart_train, heart_test, goal_train, goal_test = train_test_split(data.loc[:,'age':'thal'], \
                                                     data.loc[:,'diag_int'], test_size=0.33, random_state=0)
    return heart_train, heart_test, goal_train, goal_test

def model_building(heart):
    
    loss = ["hinge", "log_loss"]
    penalty = ["l1", "l2"]
    alpha = [0.05, 0.01]
    n_iter = [500, 1000]
    heart_train, heart_test, goal_train, goal_test = subset_decomposition(heart)
    
    best_score = 0
    best_param = (0,0,0,0)
    for l in loss:
        for p in penalty:
            for a in alpha:
                for n in n_iter:
                    lss = SGDClassifier(loss=l, penalty=p, alpha=a, max_iter=n)
                    lss.fit(heart_train, goal_train)
                    scores = cross_val_score(lss, heart.loc[:,'age':'thal'], heart.loc[:,'diag_int'], cv=10)

                    if np.mean(scores) > best_score:
                        best_score = np.mean(scores)
                        best_param = (l,p,a,n)

    lss_best = SGDClassifier(loss=l, penalty=p, alpha=a, max_iter=n)
    lss_best.fit(heart_train, goal_train)
    
   
    return lss_best

def self_prediction(heart_test, model):    
    y_pred_proba = model.predict_proba(heart_test)
    y_pred_proba = [x[1] for x in y_pred_proba]
    for i in y_pred_proba[:10]:
        print(i)
    return y_pred_proba

    
if __name__ == '__main__':
    heart = data_parsing('processed.cleveland.data.csv')
    model = model_building(heart)