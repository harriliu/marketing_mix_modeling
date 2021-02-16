import statsmodels.api as sm
import statsmodels.tsa as tsa
import scipy.stats as stats
import numpy as np
from datetime import datetime, timedelta
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, RepeatedKFold
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class cncr_model(object):
  
  def __init__(self,
               seed = None, 
               tol = .01, 
               maxiter = 1000, 
               polish = True, 
               popsize = 15, 
               strategy = 'best1bin', 
               mutation = (0.5, 1), 
               recombination = 0.7, 
               disp = False, 
               init = 'latinhypercube', 
               atol = 0, 
               updating = 'immediate'):
    
    self.seed = seed
    self.tol = tol
    self.maxiter = maxiter
    self.polish = polish
    self.popsize = popsize
    self.strategy = strategy
    self.mutation = mutation
    self.recombination = recombination
    self.disp = disp
    self.init = init
    self.atol = atol
    self.updating = updating
    
  
  def get_opt_param(self, fit_df, y, media_var, beta_list, validate=0, fit_intercept =True, params_bounds=None):
    """
    This function identity the optimial paramter for 3 MKT transformations.
    
    MKT transformations:
    1. Saturation Curve Parameters: Alpha, Beta
    2. Decay/adstock: adstock ratio
    3. Leg: number of days
    
    It transforms marketing variables using decay, s-curve, and lag, while optimizing the respective parameters using differential evolution. 
    Since adding Beta (S-curve) will make it difficult to train the model and beta does not change the transformation much. we perform grid search on Beta instead 
    optimizing it in differential evolution.
    
    The function also can validate transformations on subsets of the data.
    
    INPUTS:
    - fit_df: Design Matrix as a pd.DataFrame (for now need to include intercept)
    - y: response variable
    - media_var: string of media variables (these variables will be transformed automatically)
    - beta_list: a list of beta parameter to perform grid search (defaut list [0.000005, 0.000001, 0.000025, 0.0000015])
    - fit_intercept: whether to inlcude intercept in the model. (boolean)
    - validate: subset the last validate (k)  days as a test set (will train on first N - k days)
    - params_bounds: default to None, list of tuples otherwise of length equal to number of independent variables
    see more details on differential evolution - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    
    OUTPUTS:
    - A dataframe with optimal Alpha, decay, leg for each media variable and its correspoding beta in the beta_list
    after running this fucntion the model object will carry following intances which can be called anytime
    1.best_beta: the best Beta seleced from the grid search. (Numeric)
    2.best_mkt_param: a list of alpha, decay, and leg values for each media variable. (List)
    3.beta_grid_res: Beta grid search resul. (Dataframe)
    """
    
    p = len(media_var)

    #this makes sure that the media variables are the first p columns
    non_media_var = [i for i in fit_df.columns if i not in media_var]
    fit_df_new = fit_df.copy()[media_var + non_media_var]

    if fit_intercept == True:
      fit_df_new["intercept"] = 1    

    #creates the train/test data sets. Train sample size: N-validate. Test sample size: validate (last validate days).
    if validate > 0:
      y_temp = y.copy()
      X = fit_df_new.copy()
      X_test, y_test = fit_df_new[-validate:], y[-validate:]
      fit_df_new, y = fit_df_new[:-validate], y[:-validate]

    if params_bounds == None:
      params_bounds = ([(.87,.98),(0,1),(0,15)] *  len(media_var)) + ([(0,1000000)] *  len(media_var)) + ([(-1000000,1000000)] *  (fit_df_new.shape[1] - len(media_var)))
    
    if beta_list == None:
      beta_list = [0.000005, 0.000001, 0.000025, 0.0000015]
      
    def loss_w_transforms(theta, fit_df, y, media_var, beta):
      p = len(media_var)
      X = fit_df.copy()
      for i,m_var in enumerate(media_var):
        # adstock transformation
        X[m_var] = tsa.filters.filtertools.recursive_filter(X[m_var],theta[(p*i)+1])
        # scaling transformation
        X[m_var] = (X[m_var]/np.max(X[m_var])) * 100
        # response curve and lag transformation
        X[m_var] = (beta**(theta[p*i]**X[m_var])).shift(int(theta[(p*i)+2])).fillna(0)

      diff = y - (X @ theta[3*p:])
      return np.inner(diff,diff)
    
    results_log = pd.DataFrame([i + j for i, j in zip(np.repeat(media_var,3).tolist(), (["_alpha","_adstock","_lag"] * len(media_var)))], columns = ["name"])
    
    for beta in beta_list:
      #preforms differential evolution
      self.results = differential_evolution(loss_w_transforms, 
                                       bounds = params_bounds, 
                                       args = (fit_df_new,y,media_var,beta), 
                                       seed = self.seed, 
                                       tol = self.tol, 
                                       maxiter = self.maxiter,
                                       polish = self.polish,
                                       popsize = self.popsize,
                                       strategy = self.strategy,
                                       mutation = self.mutation,
                                       recombination = self.recombination,
                                       disp = self.disp,
                                       init = self.init,
                                       atol = self.atol,
                                       updating = self.updating)

      print(self.results.success)
      #creates table of transformation parameters
      res_out = pd.DataFrame([i + j for i, j in zip(np.repeat(media_var,3).tolist(), (["_alpha","_adstock","_lag"] * len(media_var)))], columns = ["name"])
      res_out["estimates"] = self.results.x[:(3*len(media_var))]
      res_out.loc[res_out["name"].str.contains("lag"), "estimates"] = np.floor(res_out.loc[res_out["name"].str.contains("lag"), "estimates"])
      res_out.loc[res_out["name"].str.contains("alpha"), "estimates"] = np.round(res_out.loc[res_out["name"].str.contains("alpha"), "estimates"],2)
      res_out.loc[res_out["name"].str.contains("adstock"), "estimates"] = np.round(res_out.loc[res_out["name"].str.contains("adstock"), "estimates"],2)
      new_name = "beta =" + str(beta)
      res_out = res_out.rename(columns = {"estimates": new_name})
      results_log = results_log.merge(res_out)
      
    # grid search a list of beta value provided and selected based on max test_r2
    beta_perform_final = pd.DataFrame()
    for beta in beta_list:
      perform_out = self.evaluate_performance(results_log.iloc[:len(media_var)*3,1].tolist(), fit_df, y, media_var, beta, coef=False)
      beta_perform = perform_out.set_index('name').T
      beta_perform['beta_var'] = beta
      beta_perform = beta_perform.reset_index(drop=True)
      beta_perform_final = beta_perform_final.append(beta_perform)
      beta_perform_final = beta_perform_final[['beta_var','test_r2', 'train_r2', 'test_rmse', 'train_rmse', 'test_mae', 'train_mae', 'test_mape', 'train_mape']]
    self.beta_grid_res = beta_perform_final.copy()
    self.best_beta = beta_perform_final[beta_perform_final['test_r2'] == max(beta_perform_final['test_r2'])]['beta_var'][0]
    self.best_mkt_param = results_log["beta ="+ str(self.best_beta)].tolist()
    
    return results_log
  
  # create prediction interval 
  def get_prediction_interval(self, prediction, y_actual, pi=.95):
    '''
    Get a prediction interval for a linear regression.

    INPUTS: 
    - prediction: model prediction
    - y_actual
    - pi: Prediction interval threshold (default = .95) 
    OUTPUTS: 
    - Prediction interval for single prediction
    '''

    #get standard deviation of y_test
    sum_errs = np.sum((y_actual - prediction)**2)
    stdev = np.sqrt(1 / (len(y_actual) - 2) * sum_errs)
    #get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    #generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval
    return lower, upper, interval
  
  def update_media_variables(self, transformations, beta, data, media_var):
    '''
    this function transform take a list of optimal MKT transformation parameter and transform the model data
    
    INPUTS:
    - transformations: a list of alpha, decay, and leg values for each media variable (can be obtained from best_mkt_param instance from get_opt_param function)
    - beta: best_beta instance from get_opt_param function
    - data: model data set
    - media_var: a list of media_var
    
    OUPUTS:
    - an updated dataframe with optimized MKT transformations
    '''
    self.data_updated = data.copy()
    media_vars_transformed = ["1","2","3"]
    transformations2 = np.round(transformations, 2)
    p = len(media_vars_transformed)
    for i,m_var in enumerate(media_var):
        media_vars_transformed[i] = m_var + "_" + str(transformations2[3*i]) + "_" + str(transformations2[3*i + 1]) + "_" + str(int(transformations2[3*i+2]))
        self.data_updated[media_vars_transformed[i]] = tsa.filters.filtertools.recursive_filter(self.data_updated[m_var],transformations[(p*i)+1])
        self.data_updated[media_vars_transformed[i]] = (self.data_updated[media_vars_transformed[i]]/np.max(self.data_updated[media_vars_transformed[i]])) * 100
        self.data_updated[media_vars_transformed[i]] = (beta**(transformations[p*i]**self.data_updated[media_vars_transformed[i]])).shift(int(transformations[(p*i)+2])).fillna(0)
    return self.data_updated, media_vars_transformed
  
  
  def nn_ridge(self, X, y, x0, penalty, media_var, fit_intercept = True):
    '''
    INPUTS:
    - X: Design Matrix
    - y: True response
    - x0: initialize parameters
    - penatly: Ridge Regression Penalty
    - media_var: list of media variables
    - fit_intercept: add intercept to model
    
    OUTPUTS:
    - reg_coef: regression coefficients estimates
    - y_hat: predicted values
  
    Note: set penalty = 0 for nonnegative linear regression
    '''
    def ridge(beta,X,y, penalty):
      diff = y - (X @ beta)
      return np.inner(diff,diff) + (penalty * (np.inner(beta,beta)))
  
    p = len(media_var)
    non_media_var = [i for i in X.columns if i not in media_var]
    X_new = X.copy()[media_var + non_media_var]

    if fit_intercept == True:
      X_new["intercept"] = 1
      x0 = x0 + [0]
         
    params_bounds = ([(0,None)] *  p) + ([(None,None)] *  (X_new.shape[1] - p))
    #print(params_bounds)
    results = minimize(ridge, x0 = x0, bounds = params_bounds, args = (X_new, y, penalty))
  
    #print("Convergence:", results.success)
    reg_coef = pd.DataFrame(X_new.columns, columns = ["name"]) 
    reg_coef["estimates"] = results.x
    y_hat = X_new @ results.x
    return reg_coef, y_hat
  
  def evaluate_performance(self, transformations, fit_df, y, media_var, beta, seed = None, coef = True, standardize = False, test_size = .2):
    """
    This function is wrapped inside the get_opt_param function to evaluate performance of each Beta in the grid
    
    INPUTS:
    - transformations: a list of optimal MKT parameter
    - fit_df: 
    - y: response
    - media_var: list of media variables
    - beta: Saturation curve parameter
    - coef: if coef is included in the output (boolean)
    - test_size: ratio of test dataset
    
    OUTPUTS:
    - a dataframe with list of evalation metrics of given transformed media variable.
    """
    p = len(media_var)
    X, media_vars_trans = self.update_media_variables(transformations = transformations,
                                    beta = beta,
                                    data = fit_df,
                                    media_var = media_var)
    
    if standardize == True:
      for i,m_var in enumerate(media_vars_trans):
        X[m_var] = (X[m_var] - X[m_var].mean())/X[m_var].std()
        
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 100)
    x0 = [0] * (X_train.shape[1] - p)
    reg_coef, y_train_hat = self.nn_ridge(X_train[media_vars_trans], y_train, x0, 0, media_var= media_vars_trans, fit_intercept = True)
    
    X_test["intercept"] = 1
    y_test_hat = X_test.drop(media_var, axis = 1) @ reg_coef["estimates"].tolist()
    preform = pd.DataFrame([["test_r2",r2_score(y_test, y_test_hat)],
                          ["train_r2",r2_score(y_train, y_train_hat)],
                          ["test_rmse",np.sqrt(mean_squared_error(y_test,y_test_hat))],
                          ["train_rmse",np.sqrt(mean_squared_error(y_train,y_train_hat))],
                          ["test_mae",mean_absolute_error(y_test,y_test_hat)],
                          ["train_mae",mean_absolute_error(y_train,y_train_hat)],
                          ["test_mape",mean_absolute_error(y_test,y_test_hat)/np.average(y)],
                          ["train_mape",mean_absolute_error(y_train,y_train_hat)/np.average(y)]],
                           columns= ["name","estimates"])
    final_res = preform.copy()
  
    if coef == True:
      final_res = preform.append(reg_coef)

    return final_res
  
  def fit(self, media_vars_transformed, X_df, y, n_splits = 3, n_repeats=2, random_state = 666, alphas = [0,0.1,0.5,1,2,5,10]):
    """
    the function fit the data with optimal MKT transformation generated from the get_opt_param function,
    function fit a non negetive ridge model and cross validate using repeatedKFold method on the alhpa L2 panalty term.
    
    """
    kf = RepeatedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = random_state)
    r2 = []
    for alpha_try in alphas:
      r2_temp = []
      for train_index, test_index in kf.split(X_df):
        X_train, X_test = X_df.iloc[train_index,:], X_df.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
  
        coefs, y_hat_train = self.nn_ridge(X_train, y_train, [0] * X_df.shape[1], alpha_try, media_vars_transformed, fit_intercept = True)
        X_test["intercept"] = 1
        y_hat_test = X_test @ coefs["estimates"].tolist()
        r2_temp += [r2_score(y_test,y_hat_test)]
    
      r2 += [np.mean(r2_temp)]
      print("L2 penalty =", alpha_try)
      print("r^2 at each fold:", r2_temp)
      print("r^2=", np.mean(r2_temp))
      print("\n")
      
    print("_" * 50)
    print("Best Model:") 
    print(f"L2 penalty = {alphas[np.argmax(r2)]}")
    print(f"r^2 = {np.max(r2)}")
    self.cv_coef_ = self.nn_ridge(X_df, y, [0] * X_df.shape[1], alphas[np.argmax(r2)], media_vars_transformed, fit_intercept = True)[0]
    return self

  def predict(self, x):
    if [i for i in self.cv_coef_['name'].tolist() if 'intercept' == i][0] == 'intercept':
      x["intercept"] = 1
    else: x 
    yhat = x @ self.cv_coef_['estimates'].values
    return yhat
  
  def plot_diagnostic_chart(self, data, ds, target, yhat):
    """

    INPUTS:
    - data: model data set
    - ds: date/time column name
    - target: reponse
    - yhat: model prediction
    
    OUTPUTS:
    - the function generates following plots:
    1. fitted vs. actual time series plot
    2. residual trend
    3. residual plote
    4. Normal Q-Q plot
    
    """
  
    df_chart= data[[ds,target]]
    df_chart[ds] = pd.to_datetime(pd.to_datetime(df_chart[ds]).map(lambda x:x.strftime("%Y-%m-%d")))
    df_chart['pred_MKT_seasonality_trf'] = yhat.tolist().copy()
    df_chart['residual']= df_chart[target] - df_chart['pred_MKT_seasonality_trf']
    df_chart['lower']= self.get_prediction_interval(df_chart['pred_MKT_seasonality_trf'],df_chart[target])[0]
    df_chart['upper']= self.get_prediction_interval(df_chart['pred_MKT_seasonality_trf'],df_chart[target])[1]

    # ACTUAL VS PREDICT PLOT
    sns.set(style='whitegrid')
    plt.figure(figsize=(25, 5))
    plt.plot(ds, target, data = df_chart, linewidth=4, label = 'Actual')
    plt.plot(ds, 'pred_MKT_seasonality_trf', data = df_chart, color = '#33FF33', linewidth = 3, label='Model: fn(MKT) + Controls')
    plt.fill_between('DS', 'lower','upper',data = df_chart, alpha =.3, label = 'Prediction Interval')
    plt.suptitle('Fitted vs. Actual', fontsize = 20)
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("Target Volume", fontsize = 15)
    plt.xticks(rotation = 0)
    plt.legend(loc = 'upper left')
    plt.show()

    # residual trend
    plt.figure(figsize = (25, 5))
    sns.barplot(x = ds,y = 'residual', data = df_chart)
    plt.title("Residual Trend",fontsize = 18)
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("Residual", fontsize = 15)
    plt.xticks([])
    plt.show()

    fig = plt.figure(figsize = (20, 5))
    # residual plot
    ax1 = plt.subplot(1, 2, 1)
    residual = df_chart['residual']
    ax1.scatter(df_chart[target], residual, edgecolors = (0,0,0), lw = 2, s = 80)
    ax1.plot(0, 'k--', lw = 2)
    ax1.set_title("Residual Plot", fontsize = 18)
    ax1.set_xlabel("Dependent Variables", fontsize = 15)
    ax1.set_ylabel("Residual", fontsize = 15)

    # QQ PLOT
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Normal Q-Q Plot", fontsize = 18)
    ax2.grid(1)
    figure = sm.qqplot(residual, stats.t, fit = True, line='45', ax = ax2)
    ax2.figure
    plt.tight_layout()
