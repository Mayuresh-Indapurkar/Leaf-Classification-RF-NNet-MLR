#TTU-PradnyaC, #TTU_Max
#************************************IMPORTING LIBRARIES STARTS************************************#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import corrcoef
from pylab import pcolor, show, colorbar, xticks, yticks
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV as cc
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from subprocess import check_output
#************************************IMPORTING LIBRARIES ENDS************************************#
if __name__ == '__main__':
#************************************IMPORTING DATA STARTS************************************#
    train = pd.read_csv('~/train.csv')
    test = pd.read_csv('~/test.csv')
#************************************IMPORTING DATA ENDS************************************#
    
#************************************DATA ANALYSIS AND DATA PARTITIONING STARTS************************************#
    print ("Dimensions of train data: ",train.shape)
    print ("Dimensions of test data: ",test.shape)
    test_ids = test.id 
    levels=train.species
    print("Species present in Train samples: ", levels)
    train.drop(['species', 'id'], axis=1,inplace=True) 
    test.drop(['id'],axis=1,inplace=True)
    print ("The species column saperated out from samples. Dimensions: ", levels.shape)
    print ("Dimensions of train data after split: ",train.shape)
    print ("Dimensions of test data after split: ",test.shape)
    print ("Number of distinct species in train data: ",levels.unique().shape)
    le=LabelEncoder().fit(levels)
    levels=le.transform(levels)
    print ("Encoded species column: ",levels)
    print("A listing of all distinct species: ",list(le.classes_))
#************************************DATA ANALYSIS AND DATA PARTITIONING ENDS************************************#
    
#************************************RANDOM FOREST STARTS************************************#
    Model=RandomForestClassifier(n_estimators=1000)
    Model = cc(Model, cv=3, method='isotonic')
    X_train, X_test, y_train, y_test = train_test_split(train, levels, test_size=0.4, random_state=0)
    Model.fit(X_train, y_train)
    y_predict = Model.predict_proba(X_test)
    clf_score = log_loss(y_test, y_predict)
    print ("log loss =",str(clf_score))
    sub = pd.DataFrame(y_predict, columns=list(le.classes_))
    sub.insert(0, 'id', test_ids)
    sub.reset_index()
    sub.to_csv('~/RF_CSV.csv', index = False)
    sub.head() 
#************************************RANDOM FOREST ENDS************************************#

#************************************NEURAL NETWORK STARTS************************************#
    #NN start - 1 input, 1 hidden and 1 output layer
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    Model2 =  Sequential()
    Model2.add(Dense(1024, input_dim=192, init='uniform', activation='relu'))
    Model2.add(Dropout(0.2))
    Model2.add(Dense(500, activation='sigmoid'))
    Model2.add(Dropout(0.4))
    Model2.add(Dense(99, activation='softmax'))
    #compile model
    Model2.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    # Fit model.
    y_train =to_categorical(levels)
    print(y_train)
    plot_model2 = Model2.fit(train, y_train,nb_epoch=8)
    #Plot error Vs no of iterations
    plt.plot(plot_model2.history['loss'],'o-')
    plt.xlabel('Number of Iterations:')
    plt.ylabel('Categorical Crossentropy:')
    plt.title('Train Error vs Number of Iterations:')
    # Make prediction for test data
    y_prob = Model2.predict_proba(test)
    #csv for submission
    submission = pd.DataFrame(y_prob, index=test_ids, columns=le.classes_)
    submission.to_csv('~/DNN_1Hidden.csv')
    #NN End - 1 input, 1 hidden and 1 output layer
    #NN start - 1 input and 1 output layer
    Model3 =  Sequential()
    Model3.add(Dense(1024, input_dim=192, init='uniform', activation='relu'))
    Model3.add(Dense(99, activation='softmax'))
    #compile model
    Model3.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    # Fit model.
    y_train =to_categorical(levels)
    print(y_train)
    Model3.fit(train, y_train,nb_epoch=5)
    # Make prediction for test data
    y_prob = Model3.predict_proba(test)
    #print(y_prob)
    #csv for submission
    submission = pd.DataFrame(y_prob, index=test_ids, columns=le.classes_)
    submission.to_csv('~/DNN_NoHidden.csv')
    #NN End - 1 input and 1 output layer
#************************************NEURAL NETWORK ENDS************************************#

#************************************MULTINOMIAL LOGISTIC REGRESSION STARTS************************************#
    train = pd.read_csv('~/train.csv')
    test = pd.read_csv('~/test.csv')
    np.random.seed(12345)
    #ENCODE AND NORMALIZE THE LABELS
    x_train = train.drop(['id', 'species'], axis=1).values
    le = LabelEncoder().fit(train['species'])
    y_train = le.transform(train['species'])
    #STANDARDIZE THE INPUT FEATURES BY REMOVING MEAN AND SCALING TO UNIT VARIANCE
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    params = {'C':[1000,1200,1400], 'tol': [0.00001]}
    log_reg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
    clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=-1, cv=5)
    clf.fit(x_train, y_train)
    print("Log loss: " + str(clf.best_score_))
    print("Best Parameters: " + str(clf.best_params_))
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
        print(scores)
    test_ids = test.pop('id')
    x_test = test.values
    scaler = StandardScaler().fit(x_test)
    x_test = scaler.transform(x_test)
    y_test = clf.predict_proba(x_test)
    submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
    submission.to_csv('~/op-mlr.csv')
#************************************MULTINOMIAL LOGISTIC REGRESSION ENDS************************************#