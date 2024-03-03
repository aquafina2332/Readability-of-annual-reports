# %% 输出以后把年份不全的数据去掉

import pandas as pd

# df = pd.read_csv("your_data.csv")

# 计算每个唯一值在"code"列中的重复数量
code_counts = df['code'].value_counts()

# 找到那些重复数量小于10的唯一值
values_to_remove = code_counts[code_counts < 10].index

# 使用isin方法过滤掉"code"列中对应值为values_to_remove的样本
df_filtered = df[~df['code'].isin(values_to_remove)]

# 现在，df_filtered中包含了"code"列中至少有10个重复值的样本

# 如果需要将处理后的数据保存到文件中，可以使用to_csv方法
# df_filtered.to_csv("filtered_data.csv", index=False)

# %% news
import pandas as pd

# 读取Excel文件
df = pd.read_excel(r"D:/360MoveData/Users/aquaf/Desktop/ER_NewsInfo3.xlsx")

# 将日期时间列转换为日期时间格式
df['DeclareDate'] = pd.to_datetime(df['DeclareDate'])

# 定义时间范围
start_time = pd.to_datetime('09:30:00').time()
end_time = pd.to_datetime('14:45:00').time()

# # 筛选出符合时间范围的数据
# filtered_df = df[(df['DeclareDate'].dt.time >= start_time) & (df['DeclareDate'].dt.time <= end_time)]
# 筛选出不在指定时间范围内的数据
filtered_df = df[(df['DeclareDate'].dt.time < start_time) | (df['DeclareDate'].dt.time > end_time)]

# 删除Symbol列为空白值的样本
df = filtered_df.dropna(subset=['Symbol'], how='any')

# # 将日期时间列转换为日期时间格式
# df['DeclareDate'] = pd.to_datetime(df['DeclareDate'])

# 定义时间范围
start_time1 = pd.to_datetime('00:00:00').time()
end_time1 = pd.to_datetime('09:30:00').time()
start_time2 = pd.to_datetime('14:45:00').time()
end_time2 = pd.to_datetime('23:59:59').time()

# 添加新的time列
df['time'] = df['DeclareDate'].apply(lambda x: x.date() if (start_time2 <= x.time() <= end_time2) else (x.date() - pd.DateOffset(1)) if (start_time1 <= x.time() < end_time1) else pd.NaT)
# 将'time'列的数据全部转换为日期格式
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# 将Symbol列中包含逗号的单元格拆分成多行
df_split = df.assign(Symbol=df['Symbol'].str.split(',')).explode('Symbol')

# 保存筛选后的数据到新的Excel文件，如果需要可以覆盖原始文件
df_split.to_excel('5.xlsx', index=False)

# 读取两个Excel文件
df1 = pd.read_excel('1.xlsx')
df2 = pd.read_excel('2.xlsx')
df3 = pd.read_excel('3.xlsx')
df4 = pd.read_excel('4.xlsx')
df5 = pd.read_excel('5.xlsx')

# 按列合并两个DataFrame
merged_df = pd.concat([df1, df2,df3, df4,df5], axis=1)

merged_df.to_excel('news.xlsx', index=False)


# %% 获得每天的每日开盘价、收盘价、收盘前15分钟的价格
import baostock as bs
import pandas as pd
import numpy as np

# 登陆账户
lg = bs.login()

# 读入数据并预处理
df = pd.read_excel('list.xlsx')
stock_codes = [str(num).zfill(6) for num in df.code]  

# 60和688开头的股票代码后面应该加sh.，而00和300开头的股票代码后面应该加sz.
def modify_stock_code(code):  
    if code.startswith(('60', '68')):  
        return 'sh.' + code  
    elif code.startswith(('30', '00')):  
        return 'sz.' + code  
    else:  
        return code    

codes = [modify_stock_code(code) for code in stock_codes]  

# 读入新闻文件并预处理
news_data = pd.read_excel('news.xlsx') 
news_data['time'] = pd.to_datetime(news_data['time'])
news_data.set_index('time', inplace=True)
news_data['code'] = [str(num).zfill(6) for num in news_data['code']] 
news_data['code'] = [modify_stock_code(code) for code in news_data['code']]  
news_data.index.name = None

# 读入result的表格
res_data = pd.read_excel('manip.xlsx')
res_data['code'] = [str(num).zfill(6) for num in res_data['code']] 
res_data['code'] = [modify_stock_code(code) for code in res_data['code']]  
res_data.set_index('code', inplace=True)
res_data.index.name = None


def manip_count(symbol):

    # 设置股票代码和起止日期
    # symbol = "sz.000029"  
    start_date = "2013-11-01" # 因为后面涉及前三十个交易日，所以往前多算几天
    end_date = "2023-12-31"

    # 获取15分钟级别行情数据
    rs = bs.query_history_k_data_plus(symbol, "date,time,open,close", start_date=start_date, end_date=end_date, frequency="15", adjustflag="3")
    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    # 将日期和时间列合并，并转换为 datetime 类型
    result['datetime'] = pd.to_datetime(result['date'] + ' ' + result['time'].str[8:12])  
    result.set_index('datetime', inplace=True)

    # 获取每天9:30、14:45和15:00的价格
    pr_open = result[result.index.time == pd.to_datetime('09:45:00').time()][['open']] # 开盘价是在9:30这个时间点确定的
    pr_close15 = result[result.index.time == pd.to_datetime('14:45:00').time()][['close']]
    pr_close = result[result.index.time == pd.to_datetime('15:00:00').time()][['close']]
    pr_open.index = pd.to_datetime(pr_open.index).date  
    pr_close15.index = pd.to_datetime(pr_close15.index).date  
    pr_close.index = pd.to_datetime(pr_close.index).date 
    pr_open.columns = ['open']
    pr_close15.columns = ['close15']
    pr_close.columns = ['close']
    # (pr_close15.index).equals(pr_close.index)  
    price = pr_open.join(pr_close15).join(pr_close)

    # mani-编写计算条件
    # 先计算前两个
    price = price.apply(pd.to_numeric, errors='coerce') # 转换数值
    price['delta_close'] = (price['close']-price['close15'])/price['close15']
    price['ba_30'] = price['delta_close'].rolling(window=30).mean()
    price['sigma_30'] = price['delta_close'].rolling(window=30).std()
    price['reverse'] = (price['close'] - price['open'].shift(-1))/(price['close']-price['close15'])
    # price['reverse'].fillna(method='ffill', inplace=True) # 空值用前一天的代替
    price['reverse'] = price['reverse'].ffill()
    price.index = pd.to_datetime(price.index).date 
    price = price[price.index >= pd.Timestamp('2014-01-01').date() ]
    price['con1'] = np.where(
        (price['delta_close'] > (price['ba_30'] + 3 * price['sigma_30'])) |
        (price['delta_close'] < (price['ba_30'] - 3 * price['sigma_30'])),
        1, 0)
    price['con2'] = np.where(price['reverse'] > 0.5, 1, 0)

    # 剔除当天交易结束前第15分钟至下一交易日开盘前有公告发布的股票
    price.index = pd.to_datetime(price.index)
    filtered_news_data = news_data[news_data['code'] == symbol]
    price['con3'] = np.where(price.index.isin(filtered_news_data.index), 0, 1)

    price['manip_day'] = np.where((price['con1'] == 1) & (price['con2'] == 1) & (price['con3'] == 1), 1, 0)
    # price[price['manip_day'] == 1]

    # 写入新的数据框res_data
    # 计算每年的 manip_day 的和
    price['year'] = (price.index).year
    manip_sum_by_year = price.groupby('year')['manip_day'].sum()
    manip_sum_by_year_df = manip_sum_by_year.reset_index()
    res_data.loc[symbol][manip_sum_by_year_df['year']] = manip_sum_by_year_df['manip_day'] 
    
    return

# %% 主部分
# # 写一个遍历
for symbol in codes:
    manip_count(symbol)
    
# 登出 Baostock
bs.logout()

res_data.to_csv('res.csv')


# %% 把manip二维变一维，再用readity与之匹配，最后把有缺失值的去掉
manip = pd.read_csv('manip.csv') # 前面的res.csv，并做了些修改
readty = pd.read_excel('readty.xlsx') 
readty = readty[['code' , 'year', 'Readty']]
readty['code'] = [str(num).zfill(6) for num in readty['code']] 
readty['code'] = [modify_stock_code(code) for code in readty['code']]  
readty['year'] = readty['year']+1 # 效应延迟

manip = pd.melt(manip, id_vars=['code'], value_vars=['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], 
                var_name='year', value_name='manip') # 转换为一维

# 合并两个变量
manip['year'] = manip['year'].astype('int64')
df = pd.merge(manip, readty, on=['code', 'year'], how='left')
print(df.isnull().any().any()) # 没有空值

codes_to_drop = df[df['Readty'].isnull()]['code'].unique()
df = df[~df['code'].isin(codes_to_drop)] # ~表示逻辑非

df.to_csv('data.csv')

# # 画一下相关图
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))

# plt.plot(df.index, df['manip'], label='manip', color='blue')
# plt.plot(df.index, df['Readty'], label='Readty', color='red')

# plt.title('regression')
# plt.xlabel('x')
# plt.ylabel('y')

# plt.legend()

# plt.show()

# 很丑，想想办法


