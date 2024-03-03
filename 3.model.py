
# %% 整理并合并中介变量
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
def modify_stock_code(code):  
    if code.startswith(('60', '68')):  
        return 'sh.' + code  
    elif code.startswith(('30', '00')):  
        return 'sz.' + code  
    else:  
        return code  

ana = pd.read_excel('ana.xlsx')
ana['AnaAttention'].fillna(0, inplace=True)
ana['code'] = [str(num).zfill(6) for num in ana['code']] 
ana['code'] = [modify_stock_code(code) for code in ana['code']] 
df = pd.merge(df, ana, on=['code', 'year'], how='left')
codes_to_drop = df[df['AnaAttention'].isnull()]['code'].unique()
df = df[~df['code'].isin(codes_to_drop)] # ~表示逻辑非

df['lnma'] = np.log(df['manip']+1)
df['AnaAttention'] = np.log(df['AnaAttention']+1)

# inst = pd.read_excel('institu.xlsx')
# inst['InstitutionNum'].fillna(0, inplace=True)
# inst['code'] = [str(num).zfill(6) for num in inst['code']] 
# inst['code'] = [modify_stock_code(code) for code in inst['code']] 
# inst['ReportDate'] = pd.to_datetime(inst['ReportDate'])
# inst.set_index('ReportDate', inplace=True)
# inst['year'] = (inst.index).year
# inst_sum_by_year = inst.groupby(['code','year'])['InstitutionNum'].sum()
# inst_sum_by_year = pd.DataFrame(inst_sum_by_year)
# df = pd.merge(df, inst_sum_by_year, on=['code', 'year'], how='left')
# codes_to_drop = df[df['InstitutionNum'].isnull()]['code'].unique()
# df = df[~df['code'].isin(codes_to_drop)] # ~表示逻辑非
# df['InstitutionNum'] = np.log(df['InstitutionNum']+1)
# print(df.isnull().any().any()) # 没有空值


# %% 整理控制变量
# manip 当年
# read 下一年
# anaatten 当年
# 市净率 MB 当年
mbdata = pd.read_excel('mb.xlsx')
mbdata['date'] = pd.to_datetime(mbdata['date'])
mbdata = mbdata.set_index(mbdata['date'])
mbdata['year'] = (mbdata.index).year
mb_mean_by_year = mbdata.groupby(['code','year'])['M/B'].mean()
mb = pd.DataFrame(mb_mean_by_year)
mb = mb.reset_index()
mb['code'] = [str(num).zfill(6) for num in mb['code']] 
mb['code'] = [modify_stock_code(code) for code in mb['code']] 

# 换手率turnover 当年
to = pd.read_excel('turnover.xlsx')

# 净资产收益率ROE 下一年-当年
roe = pd.read_excel('roe.xlsx')

# 资产负债率 lev 下一年-
lev = pd.read_excel('lev.xlsx')

# 公司规模 size 下一年-
size = pd.read_excel('size.xlsx')

# 资本收益率roc等 当年
rocs = pd.read_excel('ROC.xls')

# 员工密集度dens 下一年-
# 机构投资者持股institu 下一年
# 管理层持股mng 下一年
mng = pd.read_excel('dens.xlsx')

# beta 当年 剔除上交所的影响 当年
beta = pd.read_excel('beta.xlsx')

# 合并
contr = pd.merge(to, roe, on=['code', 'year'], how='left')
contr = pd.merge(contr, lev, on=['code', 'year'], how='left')
contr = pd.merge(contr, size, on=['code', 'year'], how='left')
contr = pd.merge(contr, rocs, on=['code', 'year'], how='left')
contr = pd.merge(contr, mng, on=['code', 'year'], how='left')
contr = pd.merge(contr, beta, on=['code', 'year'], how='left')
contr['code'] = [str(num).zfill(6) for num in contr['code']] 
contr['code'] = [modify_stock_code(code) for code in contr['code']] 
contr = pd.merge(contr, mb, on=['code', 'year'], how='left')

df = pd.merge(df, contr, on=['code', 'year'], how='left')
df = df[df['year'] != 2023]
df = df[df['year'] != 2013]
df = df.set_index(['code','year'])
df = df.iloc[:,1:]
df = df.fillna(method='ffill')
print(df.isnull().any().any()) # 没有空值
df.to_csv('data_all2.csv')



# %% 可读性指标重建-基于信息披露质量
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS  
from scipy.stats import spearmanr # 不需要假定正态分布

pl = pd.read_excel('正式数据/披露质量.xlsx')
inde = pd.read_csv('正式数据/read_zhibiao.csv')
df = pd.merge(pl,inde, on=['code', 'year'], how='left')
df = df.set_index(['code', 'year']) 
X = df.iloc[:,1:]
y = df['eva']

coef = [0] * 8
pp = [0] *8
for i in range(8):
    coef[i], pp[i] = spearmanr(y,X.iloc[:,i] )

# read['Readty'] = 0.0263*read['word_count']-0.0239*read['sentence_count']+0.0279*read['avg_sentence_length']-0.0655*read['average_stroke_count']+0.0587*read['passive_sentence_count']+0.0103*read['concessive_conjunction_count']+0.0218*read['uncommon_words_count']+0.1499*read['accounting_terms_count']
df = pd.read_csv('data_all2.csv')
df = df.drop('Readty',axis=1)
read = pd.read_excel('readty.xlsx')
read['Readty'] = coef[0]*read['word_count']+coef[1]*read['sentence_count']+coef[2]*read['avg_sentence_length']+coef[3]*read['average_stroke_count']+coef[4]*read['passive_sentence_count']+coef[5]*read['concessive_conjunction_count']+coef[6]*read['uncommon_words_count']+coef[7]*read['accounting_terms_count']
read['year'] = read['year'] +1
read = read[['code' , 'year', 'Readty']]
read['code'] = [str(num).zfill(6) for num in read['code']] 
read['code'] = [modify_stock_code(code) for code in read['code']] 
df = pd.merge(df, read, on=['code', 'year'], how='left')
print(df.isnull().any().any()) # 没有空值
df = df.set_index(['code','year'])
df.to_csv('data_all2.csv')


# %% 建模
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS  

# df = pd.read_csv('df.csv')
df = pd.read_csv('data_all2.csv')
df = df.set_index(['code', 'year'])

# 描述统计
statistics = {}  
for column in df.columns:  
    statistics[column] = {  
        'Sample Size': len(df[column]),  
        'Mean': df[column].mean(),  
        'Median': df[column].median(),  
        'Standard Deviation': df[column].std(),  
        'Min': df[column].min(),  
        'Max': df[column].max()  
    }  
stats_df = pd.DataFrame(statistics).T  
stats_df.to_csv('stats_df.csv')

# 一元回归
model = PanelOLS(df['Manip'],df['Readty'], entity_effects=True, time_effects=True)  
result = model.fit()  
print(result)


# 加控制变量
ind = df[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] # 一个一个加，直到共线性再去掉，最后再把不显著的一个一个删除
model = PanelOLS(df['Manip'], ind, entity_effects=True, time_effects=True)  
result = model.fit()
print(result)

# 中介效应
ind = df[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model0 = PanelOLS(df['Analy'], ind, entity_effects=True, time_effects=True)  
result0 = model0.fit()  
print(result0)

# 稳健型检验：增加变量：每股现金流量额Ncfops,Readty 依旧不变
ind = df[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',  'Ncfops' ,'IO', 'MO',  'Beta']] # 一个一个加，直到共线性再去掉，最后再把不显著的一个一个删除
model = PanelOLS(df['Manip'], ind, entity_effects=True, time_effects=True)  
result = model.fit()  
print(result)
# 此外，还加资产负债率'Lev', 每股收益'PE', 员工密集度'ED',  市净率'PB'等后均不变


# 固定行业变量 好（导完包执行下面的就可以）
df = pd.read_csv('data_all2.csv')
indus = pd.read_excel('indus.xlsx')
indus['code'] = [str(num).zfill(6) for num in indus['code']] 
indus['code'] = [modify_stock_code(code) for code in indus['code']] 
df = pd.merge(df,indus, on=['code'], how='left')
df = df.set_index(['sector','year']) 
ind = df[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] # 一个一个加，直到共线性再去掉，最后再把不显著的一个一个删除
model = PanelOLS(df['Manip'], ind, entity_effects=True, time_effects=True)  
result = model.fit()  
print(result)

# 工具变量法检验内生性
# 第一阶段
df = pd.read_csv('data_all2.csv')
top = pd.read_excel('TOPSalary.xlsx')
top['TOPSalary'] = np.log(top['TOPSalary']+1) # 加1取对数
top = top.fillna(method='ffill')
top['code'] = [str(num).zfill(6) for num in top['code']] 
top['code'] = [modify_stock_code(code) for code in top['code']] 
df = pd.merge(df,top, on=['code','year'], how='left')
df = df.set_index(['code','year'])
ind = df[['TOPSalary','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] # 一个一个加，直到共线性再去掉，最后再把不显著的一个一个删除
model1 = PanelOLS(df['Readty'], ind, entity_effects=True, time_effects=True)  
result1 = model1.fit()  
print(result1)

# 第二阶段
df['Readty_predicted'] = result1.predict(ind)
ind = df[['Readty_predicted','Turn','Roe', 'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] # 一个一个加，直到共线性再去掉，最后再把不显著的一个一个删除
model2 = PanelOLS(df['Manip'], ind, entity_effects=True, time_effects=True)  
result2 = model2.fit()  
print(result2)

# 替换被解释变量：用lnma，仍然成立
ind = df[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] # 一个一个加，直到共线性再去掉，最后再把不显著的一个一个删除
model = PanelOLS(df['Lnma'], ind, entity_effects=True, time_effects=True)  
result = model.fit()
print(result)

# 替换解释变量：用Readty2，去掉相关最大的+coef[3]*read['average_stroke_count']，仍然成立
read = pd.read_excel('readty.xlsx')
read['Readty2'] = coef[0]*read['word_count']+coef[1]*read['sentence_count']+coef[2]*read['avg_sentence_length']+coef[4]*read['passive_sentence_count']+coef[5]*read['concessive_conjunction_count']+coef[6]*read['uncommon_words_count']+coef[7]*read['accounting_terms_count']
read['year'] = read['year'] +1
read = read[['code' , 'year', 'Readty2']]
read['code'] = [str(num).zfill(6) for num in read['code']] 
read['code'] = [modify_stock_code(code) for code in read['code']] 
df = pd.merge(df, read, on=['code', 'year'], how='left')
df = df.set_index(['code', 'year'])
ind = df[['Readty2','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] # 一个一个加，直到共线性再去掉，最后再把不显著的一个一个删除
model = PanelOLS(df['Manip'], ind, entity_effects=True, time_effects=True)  
result = model.fit()  
print(result)

# 产权性质（是否国有） 异质：1=国企、0=其他
df = pd.read_csv('data_all2.csv')
equ =pd.read_excel('equity.xlsx')
equ['code'] = [str(num).zfill(6) for num in equ['code']] 
equ['code'] = [modify_stock_code(code) for code in equ['code']] 
df = pd.merge(df,equ, on=['code','year'], how='left')
df['Equity'] = np.where(df['Equity'] == '1', 1, 0)
df = df.set_index(['code', 'year'])
df1 = df[df['Equity']  == 1]
df2 = df[df['Equity']  == 0]

ind = df1[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model1 = PanelOLS(df1['Manip'], ind, entity_effects=True, time_effects=True)  
result1 = model1.fit()  
print(result1)

ind = df2[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model2 = PanelOLS(df2['Manip'], ind, entity_effects=True, time_effects=True)  
result2 = model2.fit()  
print(result2) # 非国有不显著


# 事长与总经理两职是否同一人 异质 
df = pd.read_csv('data_all2.csv')
same =pd.read_excel('same.xlsx')
same['code'] = [str(num).zfill(6) for num in same['code']] 
same['code'] = [modify_stock_code(code) for code in same['code']] 
df = pd.merge(df,same, on=['code','year'], how='left')
df['Same'] = df['Same'].fillna(method='ffill')
df['Same'] = np.where(df['Same'] == 'Y', 1, 0)
df = df.set_index(['code', 'year'])
df1 = df[df['Same']  == 1]
df2 = df[df['Same']  == 0]

ind = df1[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model1 = PanelOLS(df1['Manip'], ind, entity_effects=True, time_effects=True)  
result1 = model1.fit()  
print(result1) # 事长与总经理两职是同一人的时候不显著

ind = df2[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model2 = PanelOLS(df2['Manip'], ind, entity_effects=True, time_effects=True)  
result2 = model2.fit()  
print(result2) 


# 数字技术应用程度 异质 比均值高的是1
df = pd.read_csv('data_all2.csv')
dig =pd.read_excel('Dig.xlsx')
dig['code'] = [str(num).zfill(6) for num in dig['code']] 
dig['code'] = [modify_stock_code(code) for code in dig['code']] 
df = pd.merge(df,dig, on=['code','year'], how='left')
df['Digital'] = np.where(df['Digital'] > np.mean(df['Digital']), 1, 0)
df = df.set_index(['code', 'year'])
df1 = df[df['Digital']  == 1]
df2 = df[df['Digital']  == 0]

ind = df1[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model1 = PanelOLS(df1['Manip'], ind, entity_effects=True, time_effects=True)  
result1 = model1.fit()  
print(result1) # 都显著，但高的时候影响更大

ind = df2[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model2 = PanelOLS(df2['Manip'], ind, entity_effects=True, time_effects=True)  
result2 = model2.fit()  
print(result2) 

# 审计质量（是否是四大） 异质1 是
df = pd.read_csv('data_all2.csv')
big4 =pd.read_excel('big4.xlsx')
big4['code'] = [str(num).zfill(6) for num in big4['code']] 
big4['code'] = [modify_stock_code(code) for code in big4['code']] 
df = pd.merge(df,big4, on=['code','year'], how='left')
df['Big4'] = df['Big4'].fillna(method='ffill')
df = df.set_index(['code', 'year'])
df1 = df[df['Big4']  == 1]
df2 = df[df['Big4']  == 0]

ind = df1[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model1 = PanelOLS(df1['Manip'], ind, entity_effects=True, time_effects=True)  
result1 = model1.fit()  
print(result1) 

ind = df2[['Readty','Turn', 'Roe',  'Size',  'Roc',  'Eps',   'IO', 'MO',  'Beta']] 
model2 = PanelOLS(df2['Manip'], ind, entity_effects=True, time_effects=True)  
result2 = model2.fit()  
print(result2) # 都显著，但是不是的更显著

# 从结果中提取R²和相关信息  
r_squared_over = -0.0170#  R-squared (Overall)
n = len(df)  # 观测值的总数  
k = 9 # 独立变量的数量  
1 - (1 - r_squared_over) * ((n - 1) / (n - k - 1))  

print(df.isnull().any().any()) # 没有空值
df.to_csv('df.csv')