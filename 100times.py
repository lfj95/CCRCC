# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:42:31 2019

@author: lfj
"""

import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import sys
import pickle
from matplotlib import pyplot as plt
#reload(sys)
#sys.setdefaultencoding( "utf-8")
#sys.path.append(path)
from scorecard_fucntions import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
# -*- coding: utf-8 -*-
# =============================================================================
# 

auc=[]
acc=[]
ks_score=[]
auc2=[]
auc3=[]
features_selection_dic={}
####################################
# Group variables into bins#
####################################
#for each categorical variable, if it has distinct values more than 5, we use the ChiMerge to merge it
#for each categorical variable, if it has distinct values more than 5, we use the ChiMerge to merge it
path = './'
trainData1 = pd.read_csv(path+'38mRNA_38gene.csv',header =0, index_col=0, encoding='gbk')
trainData1 =trainData1.stack().unstack(0)
trainData2=trainData1.drop(columns=['target'])
ydata=trainData1['target']
target = trainData1.pop('target')


for iloveyou in range (0,100,1): 
    trainData, X_test, y_train, y_test = train_test_split(trainData2,ydata, test_size=0.2, random_state=iloveyou)
    
    
    trainData['target']=y_train
    allFeatures = list(trainData.columns)
    allFeatures.remove('target')
    
    
    
    #devide the whole independent variables into categorical type and numerical type
    numerical_var = []
    for var in allFeatures:
        uniq_vals = list(set(trainData[var]))
        if np.nan in uniq_vals:
            uniq_vals.remove( np.nan)
        if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
            numerical_var.append(var)
    
    categorical_var = [i for i in allFeatures if i not in numerical_var]
    
    for col in categorical_var:
        trainData[col] = trainData[col].map(lambda x: str(x).upper())
    
    
    '''
    For cagtegorical variables, follow the below steps
    1, if the variable has distinct values more than 5, we calculate the bad rate and encode the variable with the bad rate
    2, otherwise:
    (2.1) check the maximum bin, and delete the variable if the maximum bin occupies more than 90%
    (2.2) check the bad percent for each bin, if any bin has 0 bad samples, then combine it with samllest non-zero bad bin,
            and then check the maximum bin again
    '''
    deleted_features = []  #delete the categorical features in one of its single bin occupies more than 90%
    encoded_features = {}
    merged_features = {}
    var_IV = {}  #save the IV values for binned features
    var_WOE = {}
    for col in categorical_var:
        print('we are processing {}'.format(col))
        if len(set(trainData[col]))>5:
            print('{} is encoded with bad rate'.format(col))
            col0 = str(col)+'_encoding'
            #(1), calculate the bad rate and encode the original value using bad rate
            encoding_result = BadRateEncoding(trainData, col, 'target')
            trainData[col0], br_encoding = encoding_result['encoding'],encoding_result['br_rate']
            #(2), push the bad rate encoded value into numerical varaible list
            numerical_var.append(col0)
            #(3), save the encoding result, including new column name and bad rate
            encoded_features[col] = [col0, br_encoding]
            #(4), delete the original value
            #del trainData[col]
            deleted_features.append(col)
        else:
            maxPcnt = MaximumBinPcnt(trainData, col)
            if maxPcnt > 0.9:
                print('{} is deleted because of large percentage of single bin'.format(col))
                deleted_features.append(col)
                categorical_var.remove(col)
                #del trainData[col]
                continue
            bad_bin = trainData.groupby([col])['target'].sum()
            if min(bad_bin) == 0:
                print('{} has 0 bad sample!'.format(col))
                col1 = str(col) + '_mergeByBadRate'
                #(1), determine how to merge the categories
                mergeBin = MergeBad0(trainData, col, 'target')
                #(2), convert the original data into merged data
                trainData[col1] = trainData[col].map(mergeBin)
                maxPcnt = MaximumBinPcnt(trainData, col1)
                if maxPcnt > 0.9:
                    print('{} is deleted because of large percentage of single bin'.format(col))
                    deleted_features.append(col)
                    categorical_var.remove(col)
                    del trainData[col]
                    continue
                #(3) if the merged data satisify the requirement, we keep it
                merged_features[col] = [col1, mergeBin]
                WOE_IV = CalcWOE(trainData, col1, 'target')
                var_WOE[col1] = WOE_IV['WOE']
                var_IV[col1] = WOE_IV['IV']
                #del trainData[col]
                deleted_features.append(col)
            else:
                WOE_IV = CalcWOE(trainData, col, 'target')
                var_WOE[col] = WOE_IV['WOE']
                var_IV[col] = WOE_IV['IV']
    
    
    '''
    For continous variables, we do the following work:
    1, split the variable by ChiMerge (by default into 5 bins)
    2, check the bad rate, if it is not monotone, we decrease the number of bins until the bad rate is monotone
    3, delete the variable if maximum bin occupies more than 90%
    '''
    var_cutoff = {}
    for col in numerical_var:
        print("{} is in processing".format(col))
        col1 = str(col) + '_Bin'
        #(1), split the continuous variable and save the cutoff points. Particulary, -1 is a special case and we separate it into a group
        if -1 in set(trainData[col]):
            special_attribute = [-1]
        else:
            special_attribute = []
        cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target',special_attribute=special_attribute)
        var_cutoff[col] = cutOffPoints
        trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
    
        #(2), check whether the bad rate is monotone
        BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
        if not BRM:
            for bins in range(4,1,-1):
                cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target',max_interval = bins,special_attribute=special_attribute)
                trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
                BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
                if BRM:
                    break
            var_cutoff[col] = cutOffPoints
    
        #(3), check whether any single bin occupies more than 90% of the total
        maxPcnt = MaximumBinPcnt(trainData, col1)
        if maxPcnt > 0.9:
            #del trainData[col1]
            deleted_features.append(col)
            numerical_var.remove(col)
            print('we delete {} because the maximum bin occupies more than 90%'.format(col))
            continue
        WOE_IV = CalcWOE(trainData, col1, 'target')
        var_IV[col] = WOE_IV['IV']
        var_WOE[col] = WOE_IV['WOE']
        #del trainData[col]
    
    
    
    trainData.to_csv(path+'allData_2a.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)
    
    filewrite = open(path+'var_WOE.pkl','wb+')
    pickle.dump(var_WOE, filewrite)
    filewrite.close()
    
    
    filewrite = open(path+'var_IV.pkl','wb+')
    pickle.dump(var_IV, filewrite)
    filewrite.close()
    # =============================================================================
    # =============================================================================
    #########################################################
    #  Select variables with IV > 0.02 and assign WOE#
    #########################################################
    
    trainData = pd.read_csv(path+'allData_2a.csv', header=0, encoding='gbk')
    
    #num2str = ['SocialNetwork_13','SocialNetwork_12','UserInfo_6','UserInfo_5','UserInfo_10','UserInfo_17','city_match']
    #for col in num2str:
    #    trainData[col] = trainData[col].map(lambda x: str(x))
    
    
    for col in var_WOE.keys():
        print(col)
        col2 = str(col)+"_WOE"
        if col in var_cutoff.keys():
            cutOffPoints = var_cutoff[col]
            special_attribute = []
            if - 1 in cutOffPoints:
                special_attribute = [-1]
            binValue = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
            trainData[col2] = binValue.map(lambda x: var_WOE[col][x])
        else:
            trainData[col2] = trainData[col].map(lambda x: var_WOE[col][x])
    
    trainData.to_csv(path+'allData_3.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)
    
    
    
    
    
    ### (i) select the features with IV above the thresould
    iv_threshould = 0.1
    varByIV = [k for k, v in var_IV.items() if v > iv_threshould]
    
    
    ### (ii) check the collinearity of any pair of the features with WOE after (i)
    
    var_IV_selected = {k:var_IV[k] for k in varByIV}
    var_IV_sorted = sorted(var_IV_selected.items(), key=lambda d:d[1], reverse = True)
    var_IV_sorted = [i[0] for i in var_IV_sorted]
    
    removed_var  = []
    roh_thresould = 0.6
    for i in range(len(var_IV_sorted)-1):
        if var_IV_sorted[i] not in removed_var:
            x1 = var_IV_sorted[i]+"_WOE"
            for j in range(i+1,len(var_IV_sorted)):
                if var_IV_sorted[j] not in removed_var:
                    x2 = var_IV_sorted[j] + "_WOE"
                    roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
                    print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                    if abs(roh) >= roh_thresould:
                        print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                        if var_IV[var_IV_sorted[i]] > var_IV[var_IV_sorted[j]]:
                            removed_var.append(var_IV_sorted[j])
                        else:
                            removed_var.append(var_IV_sorted[i])
    
    var_IV_sortet_2 = [i for i in var_IV_sorted if i not in removed_var]
    
    ### (iii) check the multi-colinearity according to VIF > 10
    for i in range(len(var_IV_sortet_2)):
        x0 = trainData[var_IV_sortet_2[i]+'_WOE']
        x0 = np.array(x0)
        X_Col = [k+'_WOE' for k in var_IV_sortet_2 if k != var_IV_sortet_2[i]]
        X = trainData[X_Col]
        X = np.matrix(X)
        regr = LinearRegression()
        clr= regr.fit(X, x0)
        x_pred = clr.predict(X)
        R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
        vif = 1/(1-R2)
        if vif > 10:
            print("Warning: the vif for {0} is {1}".format(var_IV_sortet_2[i], vif))
    
    
    #
    #############################################################################################################
    # build the logistic regression using selected variables after single analysis and mulitple analysis#
    #############################################################################################################
    
    ### (1) put all the features after single & multiple analysis into logisitic regression
    var_WOE_list = [i+'_WOE' for i in var_IV_sortet_2]
    y = trainData['target']
    X = trainData[var_WOE_list]
    #X['intercept'] = [1]*X.shape[0]
    
#    X[np.isnan(X)] = 0
#    X[np.isinf(X)] = 0
#    y[np.isnan(y)] = 0
#    y[np.isinf(y)] = 0    
    
    LR = sm.Logit(y, X).fit()
    LR.summary()
    pvals = LR.pvalues
    pvals = pvals.to_dict()
    
    ### Some features are not significant, so we need to delete feature one by one.
    varLargeP = {k: v for k,v in pvals.items() if v >= 0.1}
    varLargeP = sorted(varLargeP.items(), key=lambda d:d[1], reverse = True)
    while(len(varLargeP) > 0 and len(var_WOE_list) > 0):
        # In each iteration, we remove the most insignificant feature and build the regression again, until
        # (1) all the features are significant or
        # (2) no feature to be selected
        varMaxP = varLargeP[0][0]
        if varMaxP == 'intercept':
            print('the intercept is not significant!')
            break
        var_WOE_list.remove(varMaxP)
        y = trainData['target']
        
        X = trainData[var_WOE_list]
    #    X['intercept'] = [1] * X.shape[0]
    
        LR = sm.Logit(y, X).fit()
        summary = LR.summary()
        pvals = LR.pvalues
        pvals = pvals.to_dict()
        varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
        varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    
    
    '''
    Now all the features are significant and the sign of coefficients are negative
    var_WOE_list = ['UserInfo_15_encoding_WOE', u'ThirdParty_Info_Period6_10_WOE', u'ThirdParty_Info_Period5_2_WOE', 'UserInfo_16_encoding_WOE', 'WeblogInfo_20_encoding_WOE',
                'UserInfo_7_encoding_WOE', u'UserInfo_17_WOE', u'ThirdParty_Info_Period3_10_WOE', u'ThirdParty_Info_Period1_10_WOE', 'WeblogInfo_2_encoding_WOE',
                'UserInfo_1_encoding_WOE']
    '''
    
    
    saveModel =open(path+'LR_Model_Normal.pkl','wb+')
    pickle.dump(LR,saveModel)
    saveModel.close()
    
    
    features_selection = var_WOE_list
 
    ### read the saved WOE encoding dictionary ###
    fread = open(path+'var_WOE.pkl','rb')
    WOE_dict = pickle.load(fread)
    fread.close()
    
    ### the below features are selected into the scorecard model in Step 5
    
    #other features can be mapped to WOE directly
    var_others = [i.replace('_WOE','').replace('_encoding','') for i in var_WOE_list if i.find('_encoding') < 0]
    for col in var_others:
        print(col)
        col2 = str(col) + "_WOE"
        if col in var_cutoff.keys():
            cutOffPoints = var_cutoff[col]
            special_attribute = []
            if - 1 in cutOffPoints:
                special_attribute = [-1]
            binValue = X_test[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
            X_test[col2] = binValue.map(lambda x: WOE_dict[col][x])
        else:
            X_test[col2] = X_test[col].map(lambda x: WOE_dict[col][x])
    
    
    
    #### load the training model
    saveModel =open(path+'LR_Model_Normal.pkl','rb')
    LR = pickle.load(saveModel)
    saveModel.close()
    LR.summary2()
    
    pvals=LR.pvalues
    #X_test=X_test[selected_var].copy()
    X_test3=X_test
    X_test=X_test[var_WOE_list].copy()
    
    
    y_pred = LR.predict(X_test)
    auc.append(roc_auc_score(y_test, y_pred))
#    acc.append(accuracy_score(y_test, y_pred))


    
    print("AUC1:",auc)
#    print("AUC2:",auc2)
#    print("AUC3:",auc3)
    features_selection_dic[iloveyou]=features_selection
    print("feature selection:",features_selection_dic)
    scorecard_result = pd.DataFrame({'prob':y_pred, 'target':y_test})
    # we check the performance of the model using KS and AR both indices should be above 30%
#    performance = KS_AR(scorecard_result,'prob','target')
#    print("KS and AR for the scorecard in the test dataset are %.0f%% and %.0f%%"%(performance['AR']*100,performance['KS']*100))
    performance = KS_AR(scorecard_result,'prob','target')
    print("KS and AR for the scorecard in the test dataset are %.0f%% and %.0f%%"%(performance['AR']*100,performance['KS']*100))
    ks_score.append(performance['KS'])
    print("KS1:",ks_score)
#    scores = Prob2Score(y_pred, 200, 100)
#    plt.hist(scores,bins=100)
#    ROC_AUC(y_test,y_pred)
#    perf_model = KS_AR(scorecard_result,'prob','target')
#print("AUC for the scorecard in the test dataset are ",auc)


















