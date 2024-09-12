import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def is_align(chunk, body):
    for i in range(len(chunk)):
        if chunk[i] -body[i] > 50000 or chunk[i] -body[i] < 0:
            return False
    return True

def process_files(chunkfile, bodyfile, output_file):
    chunk_list = []
    body_list = []
    bodyfile_data = []
    bodyfile_body = []
    # 读取第二个文件的第0列
    with open(bodyfile, 'r') as f2:
        next(f2)  # 跳过头部
        for line in f2:
            columns = line.strip().split(',')
            bodyfile_data.append(columns[0][-11:])  # 取第0列的最后11个字符
            bodyfile_body.append(list(map(int,columns[3].split('/'))))  # 取第3列

    result_list = []
    num=0

    
    with open(output_file, 'w') as out_file:
        
        with open(chunkfile, 'r') as f1:
            for line in f1:
                url = line.strip().split(',')[0][-11:]
                chunk = list(map(int, line.strip().split(',')[5].split('/')))[2:] # 获取chunk
                if url in bodyfile_data and url not in result_list:
                    result_list.append(url)
                    body = bodyfile_body[bodyfile_data.index(url)][3:] # 按url获取body
                    min_len = min(len(chunk), len(body)) - 1
                    if is_align(chunk[:min_len], body[:min_len]):
                        num+=1
                        # 写入 chunk 和 body 到新文件
                        out_file.write(url+','+ '/'.join(map(str, chunk[:min_len]))+','+ '/'.join(map(str, body[:min_len])) +'\n')

                        chunk_list.extend(chunk[:min_len])
                        body_list.extend(body[:min_len])
                    
                        
                
    print(f"共{num}组数据")
    return chunk_list, body_list


def linear_regression_and_plot(chunk_list, body_list):
    # 确保chunk_list和body_list等长
    if len(chunk_list) != len(body_list):
        raise ValueError("chunk_list and body_list must have the same length.")
    
    # 将列表转换为numpy数组并调整形状
    X = np.array(chunk_list).reshape(-1, 1)
    y = np.array(body_list)
    
    # 创建线性回归模型并拟合数据
    model = LinearRegression()
    model.fit(X, y)
    
    # 获取回归系数a和截距b
    a = model.coef_[0]
    b = model.intercept_
    print("a=", a, "b=", b)
    
    # 打印a和b
    print(f"Linear relationship: body = {a:.4f} * chunk + {b:.4f}")
    
    # 绘制散点图和回归线
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, model.predict(X), color='red', label='Regression line')
    plt.xlabel('Chunk')
    plt.ylabel('Body')
    plt.title('Linear Regression: body = a * chunk + b')
    plt.legend()
    plt.grid(True)
    plt.show()





chunkfile = r'data\quic_chunk\online_chunk_quic.csv'
bodyfile = r'data\quic_body\final_body.csv'
output_file = r'data\quic_chunk\aligned_chunk_body.csv'

chunk_list, body_list = process_files(chunkfile, bodyfile, output_file)
linear_regression_and_plot(chunk_list, body_list)
