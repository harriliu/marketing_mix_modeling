import statsmodels.api as sm
import statsmodels.tsa as tsa
import scipy.stats as stats
import numpy as np
from datetime import datetime, timedelta
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa as tsa
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, StratifiedKFold, cross_validate, cross_val_score, cross_val_predict
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV, LassoCV, Ridge, Lasso, LinearRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, RepeatedKFold



class stepwise_model(object):
    
    def __init__(self,
                 alpha_lower = 0.88,
                 alpha_upper = 0.98,
                 decay_lower = 0.0,
                 decay_upper = 1,
                 lag_lower = 0,
                 lag_upper = 15,
                 beta = 500000,
                 n_fold = 5):
        
        self.alpha_lower  = alpha_lower
        self.alpha_upper = alpha_upper
        self.decay_lower = decay_lower
        self.decay_upper = decay_upper
        self.lag_lower = lag_lower
        self.lag_upper = lag_upper
        self.beta = beta
        self.n_fold = n_fold
    

    # create leg function
    def lag_function(self, data, column_name, lag_num):
        '''
        data: dataframe
        column_name: column to perform lag transformation
        lag_num: depends on the data, if the data is at weekly level,the lag_num is number of week lag
        '''
        column_name1=column_name+'_lag_'+str(lag_num)
        data[column_name1]=data[column_name].shift(lag_num).fillna(0)
        return column_name1

    # create decay function
    def transform_decay(self, data, column_name, adstock_factor):
        column_name_adstock=column_name+'_decay_'+str(adstock_factor).replace('.','')
        data[column_name_adstock]=tsa.filters.filtertools.recursive_filter(data[column_name],adstock_factor)
        return column_name_adstock

    # create S-Curve transform function 
    def transform_s_curve (self, data, column_name, alpha, beta):
        media_input_index=data[column_name]/np.max(data[column_name]) * 100
        beta1=np.float(beta*(10**-10))
        column_name1=str(column_name)+'_alpha_'+str(alpha).replace('.','')
        data[column_name1]=beta1**(alpha**media_input_index)
        return column_name1

    def stepwise_model(self, data, target_col, feature_col, media_var):
    """
        
    INPUTS:
    - data: Design Matrix as a pd.DataFrame
    - target_col: response variable
    - feature_col: a list of all dependent variables
    - media_var: a list of media variables (these variables will be transformed automatically using the three functions above)

    OUTPUTS:
    1. model_result: coef of each variables
    2. cv_result: cross validation result
    3. X_t: transformed media variable with optimal parameter
    4. model_module: sklearn RidgeCV object
    5. optimal_param: pd.DataFrame contains the optimal alpha, decay, and lag for each media variable.
    """    

      ### S-Curve Parameter 
      alpha_lower_bound = self.alpha_lower 
      alpha_upper_bound = self.alpha_upper
      alpha_step = 0.01
      beta = self.beta 

      ### decay/adstock Parameter
      decay_lower_bound = self.decay_lower
      decay_upper_bound = self.decay_upper
      decay_step=0.1

      ### lag Parameter
      lag_lower_bound = self.lag_lower
      lag_upper_bound = self.lag_upper 
      lag_step=1

      X_t, y = data[feature_col], data[target_col]

      print(f"Features:{feature_col}")
      print("\n")
      print("#"*80)
      print("finding the best parameters.....")
      # S curve optimization
      curve_optimized=pd.DataFrame(columns=['mse','score','coef','candidates'])
      curve_log=pd.DataFrame()
      for curve_var in media_var:
          curve_list=[]
          for alpha_numb in np.around(np.arange(alpha_lower_bound, alpha_upper_bound, alpha_step),decimals=4):
              curve_list.append(transform_s_curve(data, curve_var, alpha_numb, beta))               

              scores_with_candidates_s_curve = []
              for candidate in curve_list:
                  candidate_removed=list([i for i in media_var if i not in curve_var])
                  fit_list = candidate_removed +[var for var in feature_col if var not in media_var]
                  fit_list.append(candidate)
                  X_t = data[fit_list]
                  y = data[target]
                  lasso=LassoCV(cv=5 ,fit_intercept=True, normalize=True).fit(X_t,y)
                  df_coef=pd.DataFrame(list(zip(data[fit_list].columns, lasso.coef_)),columns=['variables','coef'])
                  coef=np.round(df_coef[df_coef['variables']==candidate].iloc[0]['coef'], decimals=6)
                  y_pred = lasso.predict(X_t)
                  mse=mean_squared_error(y_pred, y)
                  score=lasso.score(X_t, y)
                  scores_with_candidates_s_curve.append((mse, score, coef, candidate))
              alpha = alpha_numb
              curve_result = pd.DataFrame(scores_with_candidates_s_curve,columns=['mse','score','coef','candidates'])
              final_pick_curve = curve_result[curve_result.mse==curve_result.mse.min()]
          curve_log = curve_log.append(curve_result)
          curve_optimized = curve_optimized.append(final_pick_curve[final_pick_curve['mse']==final_pick_curve['mse'].min()].iloc[0])
          print(list(curve_optimized['candidates']))
          pass
      curve_selected = list(curve_optimized['candidates'])
      # parsing out the optimal alpha by variable from the column name
      curve_optimized['alpha']=curve_optimized['candidates'].map(lambda x:x.split('alpha_')[-1] if len(x.split('alpha_')[-1])>2 else x.split('alpha_')[-1]+'0').astype(int)/100
      curve_optimized['var']=curve_optimized['candidates'].map(lambda x:x.split('_alpha')[0])

      # decay optimization
      decay_optimized=pd.DataFrame(columns=['score','coef','candidates'])
      decay_log=pd.DataFrame()
      for curve_opt_var in curve_selected:
          decay_list=[]
          for decay_numb in np.around(np.arange(decay_lower_bound,decay_upper_bound,decay_step),decimals=4):
              decay_list.append(transform_decay(data,curve_opt_var,decay_numb))             

              scores_with_candidates_decay = []
              for candidate in decay_list:
                  candidate_removed=list([i for i in media_var if i not in curve_opt_var])
                  fit_list=candidate_removed + [var for var in feature_col if var not in media_var]
                  fit_list.append(candidate)
                  X_t=data[fit_list]
                  y=data[target]
                  lasso=LassoCV(cv=5, fit_intercept=True, normalize=True).fit(X_t,y)
                  df_coef=pd.DataFrame(list(zip(data[fit_list].columns, lasso.coef_)),columns=['variables','coef'])
                  coef=np.round(df_coef[df_coef['variables']==candidate].iloc[0]['coef'], decimals=6)
                  y_pred = lasso.predict(X_t)
                  mse = mean_squared_error(y_pred, y)
                  score = lasso.score(X_t,y)
                  scores_with_candidates_decay.append((mse, score, coef, candidate))
              decay_result = pd.DataFrame(scores_with_candidates_decay, columns=['mse','score','coef','candidates'])
              final_pick_decay = decay_result[(decay_result.coef >= 0)]
          decay_log=decay_log.append(decay_result)
          decay_optimized = decay_optimized.append(final_pick_decay[final_pick_decay['mse']==final_pick_decay['mse'].min()].iloc[0])
          print(list(decay_optimized['candidates']))
          pass
      decay_selected = list(decay_optimized['candidates'])

      # lag optimization
      lag_optimized=pd.DataFrame(columns=['score','coef','candidates'])
      lag_log=pd.DataFrame()
      for decay_opt_var in decay_selected:
          lag_list=[]
          for lag in np.arange(lag_lower_bound,lag_upper_bound,lag_step):
              lag_list.append(lag_function(data,decay_opt_var,lag))

              scores_with_candidates_lag = []
              for candidate in lag_list:
                  candidate_removed=list([i for i in media_var if i not in decay_opt_var])
                  fit_list=candidate_removed + [var for var in feature_col if var not in media_var]
                  fit_list.append(candidate)
                  X_t=data[fit_list]
                  y = data[target]
                  lasso=LassoCV(cv=5, fit_intercept=True, normalize=True).fit(X_t,y)
                  df_coef=pd.DataFrame(list(zip(data[fit_list].columns, lasso.coef_)),columns=['variables','coef'])
                  coef=np.round(df_coef[df_coef['variables']==candidate].iloc[0]['coef'], decimals=6)
                  y_pred = lasso.predict(X_t)
                  mse = mean_squared_error(y_pred, y)
                  score = lasso.score(X_t,y)
                  scores_with_candidates_lag.append((mse, score, coef, candidate))
              lag_result = pd.DataFrame(scores_with_candidates_lag,columns=['mse','score','coef','candidates'])
              final_pick_lag = lag_result[(lag_result.coef >= 0)]
          lag_log=lag_log.append(lag_result)
          lag_optimized = lag_optimized.append(final_pick_lag[final_pick_lag['mse']==final_pick_lag['mse'].min()].iloc[0])
          print(list(lag_optimized['candidates']))
          pass
      lag_selected = list(lag_optimized['candidates'])

      final_features = lag_selected  + [var for var in feature_col if var not in media_var]
      ###### Coefficient
      X_t= data[final_features]

      # repeated KFold - split dataset into 3 folds and repeat 5 times with different randomization in each repetition
      kf=RepeatedKFold(n_splits = self.n_fold, n_repeats= 2, random_state=666)
      y = data[target]

      ridge_model = RidgeCV(cv=kf, fit_intercept=True, normalize=True , scoring='neg_mean_squared_error').fit(X= X_t,y= y)
      y_train_pred = ridge_model.predict(X_t)
      y_train = y

      # model coefficient
      result=pd.DataFrame(list(zip(data[final_features].columns, ridge_model.coef_.ravel())),columns=['variables','coef'])
      intercept=[]
      intercept.append(('intercept', ridge_model.intercept_))
      df_intercept=pd.DataFrame(intercept,columns=['variables','coef'])
      model_result=result.append(df_intercept).reset_index(drop=True)
      print("\n")
      print("#"*80)
      print("Model Coefficient")
      print(model_result)

      ###### Cross-validation
      # repeated k-fold validation
      scorings = ['r2', 'neg_mean_absolute_error']
      scores = cross_validate(ridge_model, 
                               X_t,
                               y.fillna(0).values.ravel(), 
                               cv = kf,
                               scoring = scorings,
                               return_train_score = True)

      cv_results=pd.DataFrame(scores).reset_index()
      cv_results['test_neg_mean_absolute_error']=abs(cv_results['test_neg_mean_absolute_error'])
      cv_results['train_neg_mean_absolute_error']=abs(cv_results['train_neg_mean_absolute_error'])
      cv_results['test_mape']=cv_results['test_neg_mean_absolute_error']/np.average(y)
      cv_results['train_mape']=cv_results['train_neg_mean_absolute_error']/np.average(y)
      average={'metrics': list(cv_results.columns), 'avg' : list(cv_results.mean())}
      df_cv_results=pd.DataFrame(average)
      print("\n")
      print("#"*80)
      print("Cross validation")
      print(df_cv_results)

      print("\n")
      print("#"*80)
      print("Variable VIF")
      print(get_vif(X_t, data))

      # get optimal alpha, decay and lag
      transformed_mkt_var = [val for val in final_features if 'SPEND' in val]
      alpha_list = [float(val.split("alpha_")[1].split("_decay")[0])/100 for val in transformed_mkt_var]
      decay_list = [float(val.split("_decay_")[1].split("_lag")[0])/10 for val in transformed_mkt_var]
      lag_list = [int(val.split("lag_")[1].split("_lag")[0]) for val in transformed_mkt_var]

      optimal_param = pd.DataFrame({'var_name': media_var,
                                    'alpha': alpha_list,
                                    'decay': decay_list,
                                    'lag': lag_list})

      return model_result, df_cv_results, X_t, ridge_model, optimal_param
