# marketing_mix_modeling (MMM)

### In a marketing mix model, we embeded three types of marketing variable transformations in the modeling process to better describe relationship between marketing variables and business metrics:
1. saturation curve: To account for diminishing return of marketing activities. <img src="https://render.githubusercontent.com/render/math?math=S%20=%20\beta^{\alpha^x}">
2. decay/adstock: Advertising adstock is a term used for measuring the memory effect carried over from the time of first starting advertisements.
3. lag: A lag effect is used to represent the effect of a previous value of a lagged variable when there is some inherent ordering of the observations of this variable. 

### Two modeling methods implemented in this repo:
* Rigde regression with stepwise forward selection marketing hyperparamter tunning (**stepwise_mkt_param** folder)
  - This model performs a forward stepwise search for optimal marketingtransformation parameters
  - Brute force to select the media variable with positive coefficients in each iteration
  
* Ridge regression with concurrent marketing hyperparameter tunning (**cncr_mkt_param** folder)
  - This model finds optimal marketing transformation parameter simultaneously
  - After optimal parameters are identified, fit into a customized non-negative ridge regression (using scipy ```differental_evulation``` to minized the loss function with postive bounds on the marketin variables)
  


