from collections import defaultdict
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os, re

plt.rcParams['font.sans-serif'] = ['SimHei']   
plt.rcParams['axes.unicode_minus'] = False     

class FpPair:
    def __init__(self, row) -> None:
        self.ID = row['ID']
        self.url = row['url']
        self.video_itag = row['video_itag']
        self.video_quality = row['video_quality']
        self.video_format = row['video_format']
        self.audio_itag = row['audio_itag']
        self.audio_quality = row['audio_quality']
        self.audio_format = row['audio_format']
        self.video_fp = list(map(int, row['video_fp'].split('/')))
        self.video_timeline = list(map(int, row['video_timeline'].split('/')))
        self.audio_fp = list(map(int, row['audio_fp'].split('/')))
        self.audio_timeline = list(map(int, row['audio_timeline'].split('/')))
        self.sorted_fp, self.format_fp = self.get_sorted_fp()
        self.prefix_fp, self.prefix_fp_bin = self.get_prefix_fp()
        self.prefix_dict, self.prefix_dict_bin = self.get_prefix_dict()

    # 获取音视频片段按照时间线排序后的序列
    def get_sorted_fp(self): 
        i, j = 0, 0
        sorted_fp, format_fp = [], []
        while i < len(self.video_fp)-1 and j < len(self.audio_fp):
            while i < len(self.video_fp)-1  and self.video_timeline[i+1] < self.audio_timeline[j]: # 每个视频段结束时间，要早于下一音频段开始时间
                sorted_fp.append(self.video_fp[i])
                format_fp.append('v')
                i +=1
            sorted_fp.append(self.audio_fp[j])
            format_fp.append('a')
            j += 1
        return sorted_fp[:100], format_fp[:100]
    
    # 获取前缀和序列
    def get_prefix_fp(self):
        prefix_list = [0] * (len(self.sorted_fp) + 1)
        for i in range(0, len(self.sorted_fp)):
            prefix_list[i+1] = prefix_list[i] + self.sorted_fp[i]
        # 全部/C
        prefix_list_bin = [item//C for item in prefix_list]
        return prefix_list, prefix_list_bin
    
    # 获取前缀和字典
    def get_prefix_dict(self):
        prefix_dict, prefix_dict_bin = {},{}
        for idx, value in enumerate(self.prefix_fp):
            prefix_dict[value] = idx
            prefix_dict_bin[value//C] = idx
        return prefix_dict, prefix_dict_bin
    

        

class ChunkList:
    def __init__(self, row) -> None:
        self.url = row['url']
        self.quality = row['quality']
        self.chunk_list = list(map(int, row['body_list'].split('/')))



class OnlineMatch:
    def __init__(self, min_dist_to_body) -> None:
        self.min_dist_to_body = min_dist_to_body # 和真实body的偏差上限
    

    def chunk_match(self, chunk_idx, chunk):
        for fp_idx, fp_obj in enumerate(FP_LIST):
            if not MATCH_STATE[fp_idx]:
                # 初始化
                self.last_start_index = -1 
                self.last_end_index = -1 
            else:
                self.last_start_index = MATCH_STATE[fp_idx][-1]['start_index'] # 记录上一个找到的区间的开始索引
                self.last_end_index = MATCH_STATE[fp_idx][-1]['end_index']  # 记录上一个找到的区间的结束索引

            self.subarray_sum_with_tolerance(fp_idx, fp_obj, chunk_idx, chunk, tolerance = self.min_dist_to_body)

    def subarray_sum_with_tolerance(self, fp_idx, fp_obj, chunk_idx, target, tolerance = 2000):
        '''
        输入：fp_obj:原始序列，前缀和序列；一个目标chunk
        输出：MATCH_STATE:{idx:, chunk:, start_idx:, end_idx:, subarray:, sum:, difference:,}

        '''
        sequence = fp_obj.sorted_fp
        format_list = fp_obj.format_fp
        prefix_sum_list, bin_prefix_sum_list = fp_obj.prefix_fp, fp_obj.prefix_fp_bin
        prefix_sum_dict, bin_prefix_sum_dict = fp_obj.prefix_dict, fp_obj.prefix_dict_bin

        n = len(sequence)
        # 遍历每个目标值
        found = False
        bin_target = target//C
        tolerance = tolerance//C

        for i in range(n): #self.last_end_index+1, n): # O(n) ##
            bin_prefix_sum = bin_prefix_sum_list[i]

            for k in range(0, tolerance): 
                if bin_prefix_sum - bin_target + k in bin_prefix_sum_dict:
                    start_index = bin_prefix_sum_dict[bin_prefix_sum - bin_target + k]
                    
                    if i - start_index <= 8:
                        subarray = sequence[start_index:i] # 前闭后开
                        format_subarray = format_list[start_index:i]
                        audio_count = format_subarray.count('a')
                        video_count = format_subarray.count('v')
                        # if audio_count <= video_count: # 音频片段数 < 视频片段数
                        subarray_sum = prefix_sum_list[i] - prefix_sum_list[start_index] # O(1)
                        diff = target - subarray_sum
                        
                        # 更新MATCH_STATE
                        MATCH_STATE[fp_idx].append({
                            'idx' : chunk_idx,
                            'target': target,
                            'start_index': start_index,
                            'end_index': i-1,
                            'subarray': subarray,
                            'format_subarray': format_subarray,
                            'audio_count': audio_count,
                            'video_count': video_count,
                            'sum': subarray_sum,
                            'difference':  diff
                        })
                        
                        found = True
                        break


    def result_parse(self):
        all_continuous_intervals = [None]*len(FP_LIST) # 长度和指纹数一样
        for i in range(len(MATCH_STATE)):
            if not MATCH_STATE[i]:
                continue
            longest_interval = self.find_continuous_intervals(MATCH_STATE[i])
            all_continuous_intervals[i]=longest_interval
        return all_continuous_intervals
        
    
    def find_continuous_intervals(self, results):
        '''
        输入：从一条无序的results，[]
        输出：最长连续区间，[]
        '''
        results.sort(key=lambda x: (x['idx'], x['start_index']))

        all_intervals = []

        for result in results:
            placed = False
            for interval in all_intervals:
                if result['idx'] == interval[-1]['idx'] + 1 and result['start_index'] == interval[-1]['end_index'] + 1:
                    interval.append(result)
                    placed = True
                    # break

            if not placed:
                all_intervals.append([result])

        longest_interval = max(all_intervals, key=len)
        return longest_interval

    def find_longest_intervals(self, all_intervals):
        '''
        输入：匹配结果区间，[None, [], None,[]...]
        输出：最终结果区间，[（idx,[]）]
        '''
        
        # 获取最大长度
        max_length = max((len(interval) for interval in all_intervals if interval is not None), default=0)
        if max_length == 0:
            return []
        longest_intervals = []
        # 返回所有长度等于最大长度的子列表 # 滤掉
        for index, intervals in enumerate(all_intervals):
            if not intervals :# not self.is_legal(intervals)
                continue
            if len(intervals) == max_length:
                longest_intervals.append((index, intervals))
                
        return longest_intervals # 最大长度的所有连续区间列表的列表
    
    def is_legal(self, intervals):
        '''
        '''
        legal = True
        for interval in intervals:
            if interval['start_index'] == interval['end_index'] and (interval['difference'] > 300 or interval['difference'] < 200):
                legal = False
        return legal
    
    # 返回列表最大值的下标
    def get_all_max_indices(self, lst):
        '''
        '''
        if not lst:
            raise ValueError("列表为空")
        
        max_value = max(lst)
        max_indices = [index for index, value in enumerate(lst) if value == max_value]
        return max_indices

# 自定义排序函数，提取键中的数字部分
def sort_key(k):
    # 使用正则表达式提取 x 和 y 的数字部分
    match = re.match(r"(\d+)V(\d+)A", k)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x, y)
    return (0, 0)

def plot_box(data, filename):
    '''
    输入：统计数据字典
    输出：箱图
    '''
    plt.figure(figsize=(12, 8))

    categories = list(data.keys())
    values = list(data.values())

    sns.boxplot(data=values, showfliers=False)

    plt.xticks(range(len(categories)), categories)

    plt.title('Box Plot of Categories')
    plt.ylabel('Values')

    plt.savefig(filename) #, bbox_inches='tight'
    print(f"图像已保存到 {filename}")

def plot_bar(counter, filename, title):
    combinations = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(10, 6))
    plt.bar(combinations, counts, color='skyblue')

    plt.title(title)
    plt.xlabel('组合')
    plt.ylabel('出现次数')

    plt.xticks(rotation=45)  # 旋转 x 轴标签以便更好地展示
    plt.tight_layout()  # 自动调整布局

    plt.savefig(filename) #, bbox_inches='tight'
    print(f"图像已保存到 {filename}")


def plot_pie(counter, filename, title):
    combinations = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=combinations, autopct='%1.1f%%', startangle=140, colors=plt.get_cmap('tab20').colors)

    plt.title(title)
    plt.savefig(filename) #, bbox_inches='tight'
    print(f"图像已保存到 {filename}")

if __name__=='__main__':
    
    ########################################################
    ########################################################
    # 在线指纹
    online_chunk_file = r'data\fingerprint\chunk-_for_test.csv'
    # 指纹库
    offline_fp_file = r'data\fingerprint\yt_fp_for_test.csv'

    global MATCH_STATE # 记录当前匹配状态[[int, (int,int), dict]]
    global FP_LIST, C
    tolerance = 1200 
    C = 400
    
    pic_dir = r"data\match_result\pic"
    log_dir = r'data\match_result\log'
    result_file_dir = r'data\match_result\result_file'
    
    extra_info = f"C{C}"

    box_filename = f"box_diff_distribution_{tolerance} {extra_info}.png"
    bar_filename = f"bar_diff_distribution_{tolerance} {extra_info}.png"
    pie_filename = f"pie_diff_distribution_{tolerance} {extra_info}.png"
    itag_bar_filename = f"bar_itag_pair_count_{tolerance} {extra_info}.png"
    itag_pie_filename = f"pie_itag_pair_count_{tolerance} {extra_info}.png"
    chunk_bar_filename = f"bar_chunk_num_{tolerance} {extra_info}.png"
    chunk_pie_filename = f"pie_chunk_num_{tolerance} {extra_info}.png"
    
    diff_file_path = result_file_dir + rf'\diff_distribution_{tolerance} {extra_info}.txt'
    itag_pair_path = result_file_dir + rf'\itag_pair_count_{tolerance} {extra_info}.txt'
    chunk_num_path = result_file_dir + rf'\chunk_num_{tolerance} {extra_info}.txt'
    time_num_path = result_file_dir + rf'\time_num_{tolerance} {extra_info}.txt'

    log_path = log_dir + rf'\log_{tolerance}_{datetime.now():%Y%m%d%H%M} {extra_info}.txt'


    wrong_url = r'data\match_result\wrong_url_1200.csv'
    ########################################################
    ########################################################
   
    with open(offline_fp_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        FP_LIST = [FpPair(row) for row in reader ] 

    start = time.time()
    with open(online_chunk_file, 'r', newline='') as file:
        reader = csv.DictReader(file)

        # 初始化
        correct = 0 # 匹配成功的指纹数
        num = 0 # 总指纹数
        diff_distribution = defaultdict(list) # 记录v a 组合与body的差值分布
        pair_count = defaultdict(int) # 记录v a 组合数
        itag_pair_count = defaultdict(int) # 记录v a 的itag组合数
        chunk_num = defaultdict(int) # 记录匹配成功用的chunk数
        time_num = defaultdict(int) # 记录匹配成功用的时长

        with open(log_path, 'a', encoding='utf-8') as log:
            
            for idx, row in enumerate(reader):
                num += 1
                # 初始化
                MATCH_STATE = [[] for _ in range(len(FP_LIST))] # [[一行指纹和所有已传入的块的匹配结果{1}{1}{2}{2}{3}{3}],[],[]]
                chunk_list_obj = ChunkList(row)

                print('#'*20 + f"开始识别chunk序列 标答：{chunk_list_obj.url} {chunk_list_obj.quality}" + '#'*20)
                log.write('#'*20 + f"开始识别chunk序列 标答：{chunk_list_obj.url} {chunk_list_obj.quality}" + '#'*20+'\n')
                online_match = OnlineMatch(
                        min_dist_to_body = tolerance
                    )
                
                for chunk_idx, chunk in enumerate(chunk_list_obj.chunk_list):
                    if chunk_idx == 0 or chunk_idx == 1:
                        continue
                    if chunk_idx ==10:
                        print("=-=【匹配失败】=-=")
                        log.write("=-=【匹配失败】=-=\n")
                        with open(wrong_url, 'a', encoding = 'utf-8') as f2:
                            f2.write(chunk_list_obj.url+'\n')
                        break

                    online_match.chunk_match(chunk_idx, chunk)
                    all_continuous_intervals = online_match.result_parse()
                    longest_intervals = online_match.find_longest_intervals(all_continuous_intervals)
                    # 增加判断：longest_intervals 代表唯一url
                    url_set = set()
                    for interval_tuple in longest_intervals:
                        fp_obj = FP_LIST[interval_tuple[0]]
                        url_set.add(fp_obj.url)
                    if len(url_set) == 1: # 有可能是唯一url,但多条指纹（v_itag+251或者v_itag+251-drc),有相同匹配长度,默认取后一个drc,取决于经验
                        tuple_ = longest_intervals[-1]
                        fp_obj = FP_LIST[tuple_[0]]
                        
                        print(f"【chunk_idx={chunk_idx} 匹配结果】：1个URL: Line{tuple_[0]}: {fp_obj.url} {fp_obj.video_itag} {fp_obj.audio_itag}")
                        log.write(f"【chunk_idx={chunk_idx} 匹配结果】：1个URL: Line{tuple_[0]}: {fp_obj.url} {fp_obj.video_itag} {fp_obj.audio_itag}\n")

                        # 匹配成功
                        if fp_obj.url == chunk_list_obj.url:
                            correct += 1
                            time_key = 0
                            
                            # 模式一： 匹配成功输出，模式二：匹配结果只要唯一不论对错就输出
                            for result in tuple_[1]:
                                idx = result['idx']
                                target = result['target']
                                start_index = result['start_index']
                                end_index = result['end_index']
                                subarray = result['subarray']
                                a_count = result['audio_count']
                                v_count = result['video_count']
                                sum_subarray = result['sum']
                                diff = result['difference']
                                print(f"chunk_idx={idx}:{target} : 片段索引: {start_index}--{end_index}, 子序列: {subarray} 格式：{v_count}V{a_count}A 差值: {diff}")
                                log.write(f"chunk_idx={idx}:{target} : 片段索引: {start_index}--{end_index},  子序列: {subarray} 格式：{v_count}V{a_count}A 差值: {diff}\n")


                                # 模式一 记录{v_count}V{a_count}A
                                key = f"{v_count}V{a_count}A"
                                diff_distribution[key].append(diff)
                                pair_count[key] += 1

                                time_key += a_count * 10
                            # 记录V:{fp_obj.video_itag}/A:{fp_obj.audio_itag}
                            itag_key = f"V:{fp_obj.video_itag}/A:{fp_obj.audio_itag}"
                            itag_pair_count[itag_key] += 1
                            
                            # 记录匹配成功用的chunk数
                            num_key = len(tuple_[1])
                            chunk_num[num_key] += 1

                            # 记录匹配成功用的时间（音频个数）
                            time_num[time_key] += 1 



                            break # 结束当前chunk序列
                        else:
                            # 模式二：对于碰撞错误，也打印结果分析错误的diff和xVyA的关系
                            for result in tuple_[1]:
                                idx = result['idx']
                                target = result['target']
                                start_index = result['start_index']
                                end_index = result['end_index']
                                subarray = result['subarray']
                                a_count = result['audio_count']
                                v_count = result['video_count']
                                sum_subarray = result['sum']
                                diff = result['difference']

                            print("=-=【匹配失败】=-=")
                            log.write("=-=【匹配失败】=-=\n")

                            with open(wrong_url, 'a', encoding = 'utf-8') as f2:
                                f2.write(chunk_list_obj.url+'\n')
                            break
                    
                    else:
                        print(f"【chunk_idx={chunk_idx} 临时结果】：{len(longest_intervals)}个url")
                        log.write(f"【chunk_idx={chunk_idx} 临时结果】：{len(longest_intervals)}个url\n")
            # 打印结果
            correct = correct/num
            print(f"容忍度={tolerance} 在线指纹数={num} 准确率={correct:.5f}")
            log.write(f"容忍度={tolerance} 在线指纹数={num} 准确率={correct:.5f}\n")
            end = time.time()
            print(f"总耗时：{end-start:.5f}平均耗时={(end-start)/num:.5f}")
            log.write(f"总耗时：{end-start:.5f}平均耗时={(end-start)/num:.5f}\n")
        
        #  {v_count}V{a_count}A画图
        
        # 按照自定义的排序规则对字典进行排序
        sorted_diff_distribution = dict(sorted(diff_distribution.items(), key=lambda item: sort_key(item[0])))
        sorted_pair_count = dict(sorted(pair_count.items(), key=lambda item: sort_key(item[0])))
        plot_box(sorted_diff_distribution, os.path.join(pic_dir, box_filename))
        plot_bar(sorted_pair_count, os.path.join(pic_dir, bar_filename), '每种 xVyA 组合的出现次数')
        plot_pie(sorted_pair_count, os.path.join(pic_dir, pie_filename), '每种 xVyA 组合的出现次数')
        plot_bar(itag_pair_count, os.path.join(pic_dir, itag_bar_filename), '每种 V:itag/A:itag组合的出现次数')
        plot_pie(itag_pair_count, os.path.join(pic_dir, itag_pie_filename), '每种 V:itag/A:itag组合的出现次数')
        plot_bar(chunk_num, os.path.join(pic_dir, chunk_bar_filename), '匹配成功需要的chunk个数')
        plot_pie(chunk_num, os.path.join(pic_dir, chunk_pie_filename), '匹配成功需要的chunk个数')

        # 将字典写入文件
        with open(diff_file_path, 'w') as file:
            for key in sorted_diff_distribution:
                count = sorted_pair_count[key]
                value = sorted_diff_distribution[key]
                value = '/'.join(list(map(str, value)))
                file.write(f'{key},{count},{value}\n')
        print(f"字典已成功写入文件：{diff_file_path}")

        with open(itag_pair_path, 'w') as file:
            for key in itag_pair_count:
                count = itag_pair_count[key]
                file.write(f'{key},{count}\n')
        print(f"字典已成功写入文件：{itag_pair_path}")
        
        all_num = 0
        c_num = 0
        with open(chunk_num_path, 'w') as file:
            for key in chunk_num:
                count = chunk_num[key]
                all_num += int(key) * int(count)
                c_num += int(count)
                file.write(f'{key},{count}\n')
        print(f"字典已成功写入文件：{chunk_num_path}")


        with open(time_num_path, 'w') as file:
            for key in time_num:
                count = time_num[key]
                file.write(f'{key},{count}\n')
        print(f"字典已成功写入文件：{time_num_path}")

        ave_num = all_num/c_num
        print("ave_num=", ave_num)




