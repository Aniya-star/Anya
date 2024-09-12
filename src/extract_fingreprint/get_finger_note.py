import csv, os
import socket
import dpkt
import matplotlib.pyplot as plt


# 处理PCAP文件的函数
def process_pcap(pcap, host_ip):
    # 创建一个空字典，用于存储网络流
    flows = {}

    # 以二进制模式打开PCAP文件
    with open(pcap, 'rb') as f:
        # 使用dpkt库读取PCAP文件
        r = dpkt.pcap.Reader(f)

        # 遍历PCAP文件中的每一个包
        for ts, buf in r:
            
            try:
                # 解析以太网帧
                packet = dpkt.ethernet.Ethernet(buf)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError) as e:
                # 如果解析失败，跳过这个包
                return None, None, None

            # 检查包是否是IP包，并且是否包含UDP协议
            if isinstance(packet.data, dpkt.ip.IP) and hasattr(packet.data, 'udp'):
                # 排除 rdpudp 包
                if packet.data.data.sport == 3389 or packet.data.data.dport == 3389:
                    continue

                # 获取源IP地址和目的IP地址
                src_ip = socket.inet_ntoa(packet.data.src)
                dst_ip = socket.inet_ntoa(packet.data.dst)
                # 获取源端口和目的端口
                src_port = packet.data.data.sport
                dst_port = packet.data.data.dport

                # 判断是否是目的IP为主机IP且源端口为443（HTTPS）的流
                if dst_ip == host_ip and src_port == 443:
                    # 使用源IP、目的IP、源端口和目的端口作为键来标识流
                    flow_key = (src_ip, dst_ip, src_port, dst_port)
                    if flow_key not in flows:
                        flows[flow_key] = []  # 如果流键不存在，初始化为空列表
                    flows[flow_key].append(packet)  # 将包添加到流列表中

    # 获取所有流的键
    flows_list = flows.keys()
    videoflows = []  # 用于存储视频流的列表
    for i in flows_list:
        sumlen = 0
        for j in flows[i]:
            sumlen = sumlen + j.data.data.ulen  # 计算流的总数据长度
        if sumlen > 10 * 1024 * 1024:  # 如果流的数据总长度大于10MB
            videoflows.append((i, sumlen))  # 认为这是一个视频流，将其添加到视频流列表中
            print(pcap.split('\\')[-1], sumlen)

    P = {}  # 用于存储视频流的包信息
    P_all = []  # 用于存储所有相关的视频流包
    for (videoflow, sumlen) in videoflows:
        P[videoflow] = []  # 初始化每个视频流的包列表

    # 再次读取PCAP文件，提取详细的包信息
    with open(pcap, 'rb') as f:
        r = dpkt.pcap.Reader(f)
        ind = 0
        for ts, buf in r:
            ind += 1

            try:
                packet = dpkt.ethernet.Ethernet(buf)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

            if isinstance(packet.data, dpkt.ip.IP) and hasattr(packet.data, 'udp'):
                # 提取包的相关信息
                p = {}
                p['src_ip'] = socket.inet_ntoa(packet.data.src)
                p['dst_ip'] = socket.inet_ntoa(packet.data.dst)
                p['src_port'] = packet.data.data.sport
                p['dst_port'] = packet.data.data.dport
                p['time'] = ts
                p['datalen'] = packet.data.data.ulen

                flag = 0
                for (videoflow, sumlen) in videoflows:
                    # 检查当前包是否属于视频流
                    if (p['src_ip'], p['dst_ip'], p['src_port'], p['dst_port']) == videoflow or (
                            p['dst_ip'], p['src_ip'], p['dst_port'], p['src_port']) == videoflow:
                        P[videoflow].append(p)  # 将包添加到对应的视频流中
                        flag = 1
                if flag == 1:
                    P_all.append(p)  # 将包添加到所有视频流包列表中

    return P, videoflows, P_all  # 返回处理结果，包括视频流信息和所有相关包


# 分析视频流块的函数
def request_chunk(P, host_ip):
    chunk_list = []  # 用于存储每个视频块的大小
    len_P = len(P)
    sum = 0
    for i in range(len_P):
        # 判断包是否来自主机IP
        if P[i]['src_ip'] == host_ip:
            if P[i]['datalen'] > 300:  # 如果包的数据长度大于100字节，认为是一个块的结束 # 一个重要的超参数！！！！可在100~250间取值，如果取得小容易把一个chunk分为两部分，如果取得合适，就会出现和body序列一一对应的结果，即正确。
                chunk_list.append(sum)
                sum = 0
        else:
            sum = sum + P[i]['datalen']  # 累加非主机IP的包数据长度

    video = []  # 存储视频块大小的列表
    audio = []  # 存储音频块大小的列表
    for i in chunk_list:
        # if i < 600 * 1024:  # 如果块的大小小于600KB，认为是音频块
        #     audio.append(i)
        # else:
        if i > 10000: 
            video.append(i)  # 否则认为是视频块
    return video  # 返回视频块大小的列表


def generate_online_results(folder_path, host_ip, prosessed_url_file, output_file):
    # processed_url=set()
    # with open(prosessed_url_file, 'r', encoding='utf-8') as infile:
    #     processed_url.update([line.strip()[-11:] for line in infile])

    
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name.endswith('--PCAP'):
                pcap_folder = os.path.join(root, dir_name)
                for file_name in os.listdir(pcap_folder):
                    
                    if file_name.endswith('.pcap'):
                        base_name = file_name[0:11]
                        # if base_name in processed_url:
                        #     print(f"{base_name} processed~")
                        #     continue
                        print(f"{base_name} begin~")
                        # with open(prosessed_url_file, 'a+', encoding='utf-8') as outfile:
                        #     outfile.write(base_name+"\n")

                        resolution = file_name.split('--')[-2]
                        timestamp = file_name.split('--')[-1].replace('.pcap', '')

                        # 处理PCAP文件，提取视频流信息
                        pcap_path = os.path.join(pcap_folder, file_name)
                        P, videoflows, P_all = process_pcap(pcap_path, host_ip)

                        if videoflows:
                            for (videoflow, sumlen) in videoflows:
                                # 分析视频流块
                                video_chunks = request_chunk(P[videoflow], host_ip)
                                if len(video_chunks) >= 5:  # Assuming a threshold of 10 chunks
                                    # if is_chunk(video_chunks): # 严格过滤
                                    # 将结果写入在线分析文件
                                    url = f"https://www.youtube.com//watch?v={base_name}"
                                    flow_info = f"{videoflow[1]}.{videoflow[3]}>{videoflow[0]}.{videoflow[2]}"
                                    chunk_sizes = "/".join(map(str, video_chunks)) 
                                    result_line = f"{url},{resolution},{timestamp},{flow_info},{sumlen},{chunk_sizes}\n"
                                    with open(output_file, 'a+', encoding='utf-8') as outfile:
                                        outfile.write(result_line)

# 是否严格过滤
def is_chunk(chunk_list):
    for  chunk in chunk_list: # 一个块一般都不会小于100000
        if 0 < chunk < 60000: 
            print(chunk)
            return False
    return True


# 主函数入口
if __name__ == '__main__':
    # 设置主机IP、临时文件、数据路径和输出文件
    host_ip = '192.168.38.62'
    folder_path = r'data\720_quic_traffic_result2'  
    output_file = r'data\quic_chunk\online_chunk_quic_train.csv'
    prosessed_url_file = r'data\quic_chunk\prosessed_url.csv' # 可设定一个已经处理过的url文件，避免每次重新提取块浪费时间
    generate_online_results(folder_path, host_ip, prosessed_url_file, output_file)

    
