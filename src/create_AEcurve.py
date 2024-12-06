#!/usr/python
'''
create the AE curve
'''
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import scipy
import datetime
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.colors
mpl.use('Agg')
#=====================================================
def read_costFunction(expname, ens_num, div=1.0, odir='/scratch/menaka/LakeCalibration/out'):
    fname=odir+"/"+expname+"_%02d/OstModel0.txt"%(ens_num)
    # print (fname)
    df=pd.read_csv(fname,sep="\s+",low_memory=False)
    # print (df.head())
    return (df['obj.function'].iloc[-1]/float(div))*-1.0
#=====================================================
def read_costFunction(expname, ens_num, div=1.0, odir='/scratch/menaka/LakeCalibration/out'):
    fname=odir+"/"+expname+"_%02d/OstModel0.txt"%(ens_num)
    # print (fname)
    df=pd.read_csv(fname,sep="\s+",low_memory=False)
    # print (df.head())
    return (df['obj.function'].iloc[-1]/float(div))*-1.0
#=====================================================
def read_rvt_file(file_path):
    '''
    # Function to read the file and create a dataframe
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the initial date (ignoring the first line which is the header)
    initial_entry = lines[1].split()
    date = initial_entry[0] + " " + initial_entry[1]

    # Read subsequent values, ignoring the last line (':EndObservationData')
    values = [float(line.strip()) for line in lines[2:-1]]

    # Create a date range starting from the initial date
    date_range = pd.date_range(start=date, periods=len(values), freq='D')

    # Create the dataframe
    df = pd.DataFrame({'date': date_range, 'value': values})
    return df
#=====================================================
odir='/scratch/menaka/LakeCalibration/out'
expname="E0b"
ens_num=10
#========================================================================================
# costFunc=[]
# for num in range(1,ens_num+1):
#     # print (num)
#     # metric.append(np.concatenate( (read_diagnostics(expname, num), read_WaterLevel(expname, num))))
#     # print (list(read_diagnostics(expname, num).flatten()).append(read_costFunction(expname, num))) #np.shape(read_diagnostics(expname, num)), 
#     # row=list(read_diagnostics(expname, num, odir=odir).flatten())
#     CostFun=read_costFunction(expname, num, div=2.0, odir=odir)
#     costFunc.append(CostFun)
#     print (num,CostFun)

# # get the max costfunction
# maxInd = costFunc.index(max(costFunc)) + 1
# print (costFunc.index(max(costFunc)) + 1)
maxInd = 10
output = 'output'

# read final cat 
# final_cat=pd.read_csv(odir+"/"+expname+"_%02d/finalcat_hru_info_updated.csv"%(maxInd))
final_cat=pd.read_csv("/project/def-btolson/menaka/LakeCalibration/OstrichRaven/finalcat_hru_info_updated.csv")
# add columns to finalcat_hru_info_updated.csv
final_cat['slope']=np.nan
final_cat['intercept']=np.nan
final_cat['R2']=np.nan
final_cat['p_value']=np.nan
final_cat['minWL']=np.nan
final_cat['maxWL']=np.nan

# Raven output WL
fnameWL=odir+"/"+expname+"_%02d/best_Raven/RavenInput/output2/Petawawa_ReservoirStages.csv"%(maxInd)
print (fnameWL)
df_WL_org=pd.read_csv(fnameWL)
df_WL_org['date']=pd.to_datetime(df_WL_org['date'])
print (df_WL_org.columns)

# initiate dict
ae_str_dict={}
# get details from best calibrated model
for lake in final_cat[final_cat['Obs_WA_RS1']==1]['HyLakeId'].dropna().unique()[0::]:
    # subid
    subid=final_cat[final_cat['HyLakeId']==lake]['SubId'].unique()[0]
    subid_col='sub'+str(subid)+' '
    if subid_col not in df_WL_org.columns:
        print ('no observation', lake, subid_col)
        continue

    # print (lake, subid)

    df_WL=df_WL_org.loc[:,['date',subid_col]]

    # Raven output Lake Area
    fnameWA=odir+"/"+expname+"_%02d/best_Raven/RavenInput/obs/WA_RS_%d_%d.rvt"%(maxInd,lake,subid)
    df_WA = read_rvt_file(fnameWA)

    # Merge the two dataframes on the date column
    df = pd.merge(df_WL, df_WA, on='date', suffixes=('_WL', '_WA'))
    df.rename(columns={subid_col:'WL', 'value':'WA'},inplace=True)

    # Remove all rows with -1.2345 in any column
    df = df[(df['WL'] != -1.2345) & (df['WA'] != -1.2345)]

    # print (df.head())

    # linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df['WL'], df['WA'])
    # print (slope, intercept, r_value**2, p_value)

    minWL=df_WL_org[subid_col].min()-1.0
    maxWL=df_WL_org[subid_col].max()+1.0

    if slope > 0.0:
        final_cat.loc[final_cat['HyLakeId']==lake,'slope'] = abs(slope)
        final_cat.loc[final_cat['HyLakeId']==lake,'intercept'] = intercept
        final_cat.loc[final_cat['HyLakeId']==lake,'R2'] = r_value**2
        final_cat.loc[final_cat['HyLakeId']==lake,'p_value'] = p_value
        final_cat.loc[final_cat['HyLakeId']==lake,'minWL'] = minWL
        final_cat.loc[final_cat['HyLakeId']==lake,'maxWL'] = maxWL
    else:
        # linear regression
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([minWL,maxWL], [df['WA'].min(),df['WA'].max()])
        final_cat.loc[final_cat['HyLakeId']==lake,'slope'] = abs(slope)
        final_cat.loc[final_cat['HyLakeId']==lake,'intercept'] = intercept
        final_cat.loc[final_cat['HyLakeId']==lake,'R2'] = -9999
        final_cat.loc[final_cat['HyLakeId']==lake,'p_value'] = -9999
        final_cat.loc[final_cat['HyLakeId']==lake,'minWL'] = minWL
        final_cat.loc[final_cat['HyLakeId']==lake,'maxWL'] = maxWL

print (final_cat.head())
print (len(final_cat[final_cat['Obs_WA_RS1']==1]['HyLakeId'].dropna().unique()))
print (len(final_cat['HyLakeId'].dropna().unique()))

lake1=final_cat[final_cat['Obs_WA_RS1']==1]['HyLakeId'].dropna().unique()
lake1 = final_cat[final_cat['Obs_WA_RS1']==1]['HyLakeId'].dropna().unique()
lake2 = final_cat['HyLakeId'].dropna().unique()
lake2 = lake2[~np.isin(lake2, lake1)]

print (lake2)

for lake in lake2:
    lakeArea=final_cat[final_cat['HyLakeId']==lake]['LakeArea'].dropna().unique()[0]
    maxWL=df_WL_org[subid_col].max()
    minWL=df_WL_org[subid_col].min()
    maxA=lakeArea*1.1
    minA=lakeArea*0.9
    # linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([minWL,maxWL], [minA,maxA])
    # print (slope, intercept, r_value**2, p_value)
    final_cat.loc[final_cat['HyLakeId']==lake,'slope'] = abs(slope)
    final_cat.loc[final_cat['HyLakeId']==lake,'intercept'] = intercept
    final_cat.loc[final_cat['HyLakeId']==lake,'R2'] = -9999
    final_cat.loc[final_cat['HyLakeId']==lake,'p_value'] = -9999
    final_cat.loc[final_cat['HyLakeId']==lake,'minWL'] = minWL
    final_cat.loc[final_cat['HyLakeId']==lake,'maxWL'] = maxWL

# save file
print (final_cat.head())
# final_cat=final_cat[~final_cat['Unnamed: 0']]
final_cat.to_csv('./out/finalcat_hru_info_updated_AEcurve.csv',index=False)


    
    # # print ('='*20)
    # # string=[':AreaStageRelation LOOKUP_TABLE']
    # # string.append('   '+str(len(np.arange(minWL,maxWL+1,0.1))))
    # # for WL in np.arange(minWL,maxWL+1,0.1):
    # #     result = intercept + slope * WL
    # #     # Format values to 2 decimal places
    # #     string.append(f"   {WL:.2f}   {result:.2f}")
    # # string.append(':EndAreaStageRelation')

    # # ae_str_dict[lake]=string
    # # print ("\n".join(string))