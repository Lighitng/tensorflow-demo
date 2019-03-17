import xlrd
import xlwt
import pandas as pd
from xlrd import xldate_as_tuple
from datetime import date,datetime

# 读取水稻数据
# apiUse {json} data
# {
#     title: '水稻名',
#     data: dataFrame
# }
# apiSuccessExample {json} datas
# [...data]
def read_excel_sync():
    # 打开文件
    workbook = xlrd.open_workbook(r'./rawdata.xlsx')
    # 获取一张表
    sheet = workbook.sheet_by_index(0)

    # 获取整行和整列的值（数组）
    # titles: 四个水稻种类
    # attributes: 水稻的属性 时间+六个基础属性
    titles = [title for title in sheet.row_values(0) if title is not '']
    attributes = ['time'] + [attr for attr in sheet.row_values(1)[1:7]]
    # datas: 水稻的数据 startCols: 每种水稻数据的起始列
    datas = [{'title': titles[index], 'data': []} for index in range(4)]
    startCols = [1, 7, 13, 19]
    # 格式化data，分为四组
    for row in range(sheet.nrows - 2):
        rowdata = sheet.row_values(row + 2)
        for (index, obj) in enumerate(datas):
            #列表第一个为时间 第2-7个为六个田间数据
            obj['data'].append([datetime(*xldate_as_tuple(rowdata[0], 0))] + rowdata[startCols[index]:startCols[index]+6])
    dataFrames = []
    # 构建dataFrame数据
    for data in datas:
        dataFrames.append({
            'title': data['title'],
            'data': pd.DataFrame(data['data'], columns=attributes)
        })
    return read_excel_sync()
if __name__ == '__main__':
    read_excel_sync()
