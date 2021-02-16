from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import seaborn as sns
import math

class driver_analysis:
    
    def __init__(self, beta):
        self.beta = beta*(10**10)

    def get_sat_lvl(self, data, ds, alpha, media_var):
        '''
        Returns a indexed response curve with different saturation level:
        1. Breakthrough
        2. Optimal
        3. Saturation Begin
        4. Full Saturation

        Saturation level is calculated by taking 1st, 2nd, 3rd integral of the curve function
        Note saturation level is default at weekly level
        '''
        data[ds] = pd.to_datetime(data[ds])
        data['week'] = data[ds].map(lambda x:x - timedelta(days=x.isoweekday() % 7))
        data = data[['week', media_var]].groupby("week").sum().reset_index()

        df_curve= pd.DataFrame()

        index=((np.mean(data[media_var])/10)*100)/np.max(data[media_var])
        df_curve['Index']=np.arange(0,300,index)
        df_curve['var_volume']=df_curve['Index']*np.max(data[media_var])/100

        def s_curve_chart (data, column_name, alpha, beta):
            media_input_index = data['Index']
            beta1 = np.float(beta*(10**-10))
            column_name1 = str(column_name)+'_alpha_'+str(alpha).replace('.','')
            data[column_name1] = round(beta1**(alpha**media_input_index),8)
            return column_name1

        df_curve['var_curve'] = s_curve_chart(df_curve,'var_volume',alpha, self.beta)
        df_curve['max_var'] = np.max(data[media_var])
        df_curve['mean_var'] = np.mean(data[media_var])
        df_curve.drop('var_curve',axis = 1,inplace = True)
        df_curve.sort_values(by = 'var_volume',inplace = True)


        ########################################################################
        ##########Calculate optimal point 1st derivative of the curve###########
        ########################################################################

        def deri_1st(data,var_column,index_column):
            data['deri_1st']=alpha**(data[index_column])*data[var_column]*np.log(alpha)*np.log(np.float(self.beta*(10**-10)))
        deri_1st(df_curve,'var_volume_alpha_'+str(alpha).replace('.',''),'Index')
        self.opt_x=df_curve[df_curve['deri_1st']==df_curve['deri_1st'].max()]['var_volume']
        self.opt_y=df_curve[df_curve['deri_1st']==df_curve['deri_1st'].max()]['var_volume_alpha_'+str(alpha).replace('.','')]
        df_curve['opt_x'] = self.opt_x
        df_curve['opt_y'] = self.opt_y

        ############################################################
        #######Calculate breakthrough point 2nd derivative #########
        ############################################################
        def deri_2nd(data,var_column,index_column,frist_column):
            data['deri_2nd']=data[frist_column]*np.log(alpha)+\
            alpha**(2*data[index_column])*data[var_column]*\
            np.log(alpha)*np.log(alpha)*np.log(np.float(self.beta*(10**-10)))*np.log(np.float(self.beta*(10**-10)))  

        deri_2nd(df_curve,'var_volume_alpha_'+str(alpha).replace('.',''),'Index','deri_1st')
        self.bt_x=df_curve[df_curve['deri_2nd']==df_curve['deri_2nd'].max()]['var_volume']
        self.bt_y=df_curve[df_curve['deri_2nd']==df_curve['deri_2nd'].max()]['var_volume_alpha_'+str(alpha).replace('.','')]
        df_curve['bt_x']=self.bt_x
        df_curve['bt_y']=self.bt_y

        ##################################################################
        #########Calculate saturation begins point 3rd derivative#########
        ##################################################################
        def deri_3rd(data,var_column,index_column,frist_column):
            data['deri_3rd']=data[frist_column]*(alpha**(2*data[index_column])*np.log(np.float(self.beta*(10**-10))**2)+\
            3*alpha**data[index_column]*np.log(np.float(self.beta*(10**-10)))+1)                                                       

        deri_3rd(df_curve,'var_volume_alpha_'+str(alpha).replace('.',''),'Index','deri_1st')    
        self.sb_x=df_curve[df_curve['deri_3rd']==df_curve['deri_3rd'].max()]['var_volume']
        self.sb_y=df_curve[df_curve['deri_3rd']==df_curve['deri_3rd'].max()]['var_volume_alpha_'+str(alpha).replace('.','')]
        df_curve['sb_x']=self.sb_x
        df_curve['sb_y']=self.sb_y

        #################################################
        #########Calculate full saturation point#########
        #################################################

        self.fs_x=df_curve[df_curve['var_volume_alpha_'+str(alpha).replace('.','')]>=0.992]['var_volume'][0:1]
        self.fs_y=df_curve[df_curve['var_volume_alpha_'+str(alpha).replace('.','')]>=0.992]['var_volume_alpha_'+str(alpha).replace('.','')][0:1]
        df_curve['fs_x']=self.fs_x
        df_curve['fs_y']=self.fs_y

        return df_curve
    
    
    def readable_number(self, n):
    
        mill_lst = ['',' Thousand',' Million',' Billion',' Trillion']

        n = float(n)
        millidx = max(0,min(len(mill_lst)-1, int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

        return '{:.1f}{}'.format(n / 10**(3 * millidx), mill_lst[millidx])

    def plot_sat_lvl(self, df_curve, model_df, ds, var):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        plt.style.use('ggplot')

        #plot curve line
        lm = sns.lineplot(x='var_volume', y = [col for col in df_curve.columns if "alpha" in col][0], 
                          data = df_curve, color = '#37536d', ax = ax1)

        # formatting number into readable format
        y_ticks = lm.get_yticks()
        x_ticks = lm.get_xticks()
        lm.set_yticklabels(['{:,.0%}'.format(i) for i in y_ticks])
        lm.set_xticklabels([self.readable_number(i) for i in x_ticks])

        # plot saturation levels
        ax1.plot(df_curve['bt_x'], df_curve['bt_y'],'ro',label="Break Through",marker='o', markersize=10,color='m')
        ax1.plot(df_curve['opt_x'], df_curve['opt_y'], 'ro',label="Optimal",marker='o', markersize=10,color='g')
        ax1.plot(df_curve['sb_x'], df_curve['sb_y'], 'ro',label="Satuation Begins",marker='o', markersize=10,color='r')
        ax1.plot(df_curve['fs_x'], df_curve['fs_y'], 'ro',label="Full Satuation",marker='o', markersize=10,color='c')
        # # Set plot options and show plot
        ax1.set_xlabel('Variable Volumes',fontsize=20)
        ax1.set_ylabel('Response Index',fontsize=20)
        ax1.set_title(var +' Response Curve',fontsize=20)
        ax1.legend(loc='center right', fancybox=False, framealpha=0)

        # creating dataframe for plotting volume against saturation level plot
        df_volume = pd.DataFrame()
        df_volume['period'] = pd.to_datetime(pd.to_datetime(model_df[ds]).map(lambda x:x.strftime("%Y-%m-%d")))
        df_volume['week'] = df_volume['period'].map(lambda x:x - timedelta(days=x.isoweekday() % 7))
        df_volume['week'] = pd.to_datetime(df_volume['week']).map(lambda x:x.strftime("%Y-%m-%d"))
        df_volume['var_volume'] = model_df[var]
        df_volume = df_volume[['week', 'var_volume']].groupby("week").sum().reset_index()
        max_x=df_volume['var_volume'].max()


        df_volume['Optimal']=int(df_curve['opt_x'].unique()[1])
        df_volume['Break Through']=int(df_curve['bt_x'].unique()[1])
        df_volume['Satuation Begins']=int(df_curve['sb_x'].unique()[1])

        try:
            df_volume['Full Satuation']=int(df_curve['fs_x'].unique()[1])
        except:
            print('out of range')
            fs_x=0
            pass

        df_volume['Max'] = max_x
        df_volume['var_name'] = var

        # plot volume against saturation level
        textstr = '\n'.join((
            r'Breakthrough: ${}'.format(self.readable_number(int(df_volume['Break Through'].unique()[0])), ),
            r'Optimal: ${}'.format(self.readable_number(int(df_volume['Optimal'].unique()[0])), ),
            r'Saturation Begins: ${}'.format(self.readable_number(int(df_volume['Satuation Begins'].unique()[0])),),
            r'Full Saturation: ${}'.format(self.readable_number(int(df_volume['Full Satuation'].unique()[0])),),

        ))

        ax2 = sns.barplot(x=df_volume['week'], y = df_volume['var_volume'], color = '#37536d', ax = ax2)
        y_ticks2 = ax2.get_yticks()
        ax2.set_yticklabels([self.readable_number(i) for i in y_ticks2])

        ax2.plot('week','Break Through',data=df_volume, color='m', linewidth=5,linestyle='dashed')
        ax2.plot('week','Optimal', data=df_volume, color='g', linewidth=5,linestyle='dashed')
        ax2.plot('week','Satuation Begins', data=df_volume, color='r', linewidth=5,linestyle='dashed')
        ax2.plot('week','Full Satuation', data=df_volume, color='c', linewidth=5,linestyle='dashed')
        ax2.set_title(var +' Volume Against Weekly Saturation Levels',fontsize=20)
        ax2.set_xlabel("Week",fontsize=20)
        ax2.set_xticks(df_volume['week'])
        ax2.set_xticklabels(df_volume['week'], rotation=40, ha='right')
        ax2.set_ylabel("Volume",fontsize=20)
        ax2.set_yticks(y_ticks2)

        props = dict(boxstyle='round', alpha=0.5)
        ax2.text(0.6, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax2.legend(loc='upper right', fancybox=True, framealpha=5, bbox_to_anchor=(1, 0.95))

        plt.tight_layout(pad=5)
        plt.show()
