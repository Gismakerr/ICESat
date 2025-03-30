# -*- coding: utf-8 -*-
"""
# -*- coding: utf-8 -*-
@Time: 2024/4/14 11:19
@Author: LXX
@File: 0308关联.py
@IDE：PyCharm
@Motto：ABC(Always Be Coding)
"""

import h5py
import os
import arcpy
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import Feature as F

# 使用matplotlib的颜色映射
cmap = plt.get_cmap('jet')

'''
关联 ATL08 与 ATL03 数据产品:

1.遍历 ATL08 、ATL03数据产品的每个光子；
2.利用光子区间号匹配(ph_segment_id=segment_id)，找到两组数据的相同区间；
3.逐个匹配每个光子，ATL03 数据的光子序号等于 ATL03 数据的起始光子序号与 ATL08 数据的相对光子序号相加
 (classed_pc_indx+ph_index_beg-1)

由此可以将ATL08的每个光子对应在ATL03 数据产品中
'''


class ATLDataLoader:
    def __init__(self, atl03Path, atl08Path, gtx):
        # 初始化
        self.atl03Path = atl03Path
        self.atl08Path = atl08Path
        self.gtx = gtx
        self.load()

    def load(self):

        '''
        步骤:
        1)打开并读取 ATL03 文件，提取经度、纬度、分段起始索引、分段ID、高度和信号置信度等数据;
        2)打开并读取 ATL08 文件，提取光子分类索引、分类标识、分段ID和高度等数据;
        3)使用 ismember 方法匹配 ATL03 和 ATL08 的分段，以确定相同分段之间的对应关系;
        4)根据匹配结果，确定新的映射关系，并根据映射关系将 ATL08 的分类信息和高度信息与 ATL03 对应起来;
        5)创建一个 DataFrame 存储处理后的数据。
        '''

        # 读取ATL03分段数据
        f = h5py.File(self.atl03Path, 'r')
        atl03_lat = np.array(f[self.gtx + '/heights/lat_ph'][:])
        atl03_lon = np.array(f[self.gtx + '/heights/lon_ph'][:])
        atl03_ph_index_beg = np.array(f[self.gtx + '/geolocation/ph_index_beg'][:]) #20米分段光子起始编号
        atl03_segment_id = np.array(f[self.gtx + '/geolocation/segment_id'][:])  #20米分段编号
        atl03_heights = np.array(f[self.gtx + '/heights/h_ph'][:])   #光子高程
        atl03_conf = np.array(f[self.gtx + '/heights/signal_conf_ph'][:]) #光子置信度
        f.close()

        # 读取ATL08分段数据
        f = h5py.File(self.atl08Path, 'r')
        atl08_classed_pc_indx = np.array(f[self.gtx + '/signal_photons/classed_pc_indx'][:])  #该光子在20米分段中的相对索引号
        atl08_classed_pc_flag = np.array(f[self.gtx + '/signal_photons/classed_pc_flag'][:])  #该光子在20米分段中的分类
        #latitudes = np.array(f[self.gtx + '/land_segments/latitude'][:])
        atl08_segment_id = np.array(f[self.gtx + '/signal_photons/ph_segment_id'][:]) #20米的轨道段
        
        
        atl08_ph_h = np.array(f[self.gtx + '/signal_photons/ph_h'][:])
        f.close()

        # 利用光子区间号匹配(ph_segment_id=segment_id)，找到两组数据的相同区间
        atl03SegsIn08TF, atl03SegsIn08Inds = self.ismember(atl08_segment_id, atl03_segment_id)  #相同区间位置索引，bool，数字索引

        # 获取ATL08分类的索引和标识值
        atl08classed_inds = atl08_classed_pc_indx[atl03SegsIn08TF]   #相同20米分段区间中的光子相对索引号
        atl08classed_vals = atl08_classed_pc_flag[atl03SegsIn08TF]   #相同20米分段区间中的光子的分类
        atl08_hrel = atl08_ph_h[atl03SegsIn08TF]                     #相同20米分段区间中的光子高程

        # 确定ATL03数据的新映射
        atl03_ph_beg_inds = atl03SegsIn08Inds
        atl03_ph_beg_val = atl03_ph_index_beg[atl03_ph_beg_inds]     #相同20米分段区间中光子起始编号
        newMapping = atl08classed_inds + atl03_ph_beg_val - 2        #相同20米分段区间中的光子索引号

        # 获取输出数组的最大大小
        # sizeOutput = np.max(newMapping)
        sizeOutput = len(atl03_lat) - 1

        # 用零预填充所有光子类阵列
        allph_classed = (np.zeros(sizeOutput + 1)) - 1
        allph_hrel = np.full(sizeOutput + 1, np.nan)

        # 加入ATL08分类信息
        allph_classed[newMapping] = atl08classed_vals
        allph_hrel[newMapping] = atl08_hrel

        # 匹配ATL03大小
        allph_classed = allph_classed[:len(atl03_heights)]
        allph_hrel = allph_hrel[:len(atl03_heights)]

        # 创建DataFrame存放数据
        self.df = pd.DataFrame()
        self.df['lon'] = atl03_lon  # longitude
        self.df['lat'] = atl03_lat  # latitude
        self.df['z'] = atl03_heights  # elevation
        self.df['h'] = allph_hrel  # 相对于参考面的高度
        self.df['conf'] = atl03_conf[:, 0]  # confidence flag（光子置信度）
        self.df['classification'] = allph_classed  # atl08 classification（分类标识）
        self.df['beam'] = self.gtx

    def ismember(self, a_vec, b_vec, method_type='normal'):
        """ MATLAB equivalent ismember function """
        # 该函数主要用于判断一个数组中的元素是否存在于另一个数组中，并返回匹配的索引

        if (method_type.lower() == 'rows'):

            # 将a_vec转换为字符串数组
            a_str = a_vec.astype('str')
            b_str = b_vec.astype('str')

            # #将字符串连接成一维字符串数组
            for i in range(0, np.shape(a_str)[1]):
                a_char = np.char.array(a_str[:, i])
                b_char = np.char.array(b_str[:, i])
                if (i == 0):
                    a_vec = a_char
                    b_vec = b_char
                else:
                    a_vec = a_vec + ',' + a_char
                    b_vec = b_vec + ',' + b_char

        matchingTF = np.isin(a_vec, b_vec)
        common = a_vec[matchingTF]
        common_unique, common_inv = np.unique(common, return_inverse=True)  # common = common_unique[common_inv]
        b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
        common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
        matchingInds = common_ind[common_inv]

        return matchingTF, matchingInds

    def extract_08_ground(self, extent=None):
        """
        提取 ATL08 数据中 classification == 1 的地面点数据，并根据 extent 进行空间过滤（如果提供）。
        """
        # 先筛选 classification == 1 且 conf 不是 0 或 1
        df_filtered = self.df[(self.df['classification'] == 1) & ~self.df['conf'].isin([0, 1, 2])].copy()

        # 如果提供了 extent，进一步筛选
        if extent is not None:
            xmin, ymin, xmax, ymax = extent  # 解包元组
            df_filtered = df_filtered[
                (df_filtered['lon'] >= xmin) & (df_filtered['lon'] <= xmax) &
                (df_filtered['lat'] >= ymin) & (df_filtered['lat'] <= ymax)
            ]

        # 重置索引
        self.df_class_1 = df_filtered.reset_index(drop=True)
        return self.df_class_1
            
    def create_shapefile_from_df(self, data, output_shp, lon_field, lat_field, spatial_ref=4326):
        """
        使用 arcpy 将 Pandas DataFrame 数据转换为 SHP 点文件。
        
        参数：
        df         --  Pandas DataFrame，必须包含用户指定的经纬度字段
        output_shp --  输出的 shapefile 文件路径
        lon_field  --  DataFrame 中的经度字段名称（字符串）
        lat_field  --  DataFrame 中的纬度字段名称（字符串）
        spatial_ref -- 坐标参考系统，默认 WGS84 (EPSG:4326)
        """
        # if type == "08":
        #     df = self.df_class_1
        # elif type == "03":
        #     df = self.df
        
        df = data
        # 检查 DataFrame 是否包含经纬度字段
        if lon_field not in df.columns or lat_field not in df.columns:
            raise ValueError(f"DataFrame 中缺少 '{lon_field}' 或 '{lat_field}' 字段！")

        # 获取除经纬度外的所有字段
        other_fields = [col for col in df.columns if col not in [lon_field, lat_field]]

        # 设置 ArcPy 环境
        arcpy.env.overwriteOutput = True
        spatial_reference = arcpy.SpatialReference(spatial_ref)  # WGS 1984 坐标系统

        # 创建 SHP 点要素类
        arcpy.CreateFeatureclass_management(
            out_path=output_shp.rsplit("\\", 1)[0],  # SHP 存储的文件夹
            out_name=output_shp.rsplit("\\", 1)[1],  # SHP 文件名称
            geometry_type="POINT",
            spatial_reference=spatial_reference
        )

        # 根据 df 自动添加字段（数据类型自动识别）
        field_types = {"int64": "LONG", "float64": "DOUBLE", "object": "TEXT"}
        for field in other_fields:
            field_type = field_types.get(str(df[field].dtype), "TEXT")  # 默认文本类型
            arcpy.AddField_management(output_shp, field, field_type)

        # 构建字段列表
        insert_fields = ["SHAPE@"] + other_fields

        # 插入数据
        with arcpy.da.InsertCursor(output_shp, insert_fields) as cursor:
            for _, row in df.iterrows():
                point = arcpy.Point(row[lon_field], row[lat_field])
                row_values = [row[field] for field in other_fields]
                cursor.insertRow([point] + row_values)
    
    
    def save_to_csv_with_progress(self, data, filename):
        # if type == "08":
        #     data = self.df_class_1
        # elif type == "03":
        #     data = self.df
        
        total_rows = len(data)
        with tqdm(total=total_rows, desc="关联数据导出至CSV", unit="row") as pbar:
            start_time = time.time()
            data.to_csv(filename, index=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            pbar.set_postfix({"Time (s)": elapsed_time})
            pbar.update(total_rows)
            

class DataVisualizer:
    def __init__(self, csv_path, class_dict):
        self.df = pd.read_csv(csv_path)
        self.class_dict = class_dict

    def plot(self):
        fig, ax = plt.subplots()
        for c in self.class_dict.keys():
            mask = self.df['classification'] == c
            ax.scatter(self.df[mask]['lat'], self.df[mask]['z'],
                       color=self.class_dict[c]['color'], label=self.class_dict[c]['name'], s=1)
        ax.set_xlabel('Latitude (°)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Ground Track')
        ax.legend(loc='best')
        plt.show()


def get_tif_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]


# if __name__ == "__main__":
#     # 单个条带处理测试（批处理见末尾）:
#     # atl03Path = "D:/other/icesat/text/1/shan/processed_ATL03_20200915100047_12530806_006_02.h5"
#     # atl08Path = "D:/other/icesat/text/2/processed_ATL08_20200915100047_12530806_006_02.h5"
#     atl03Path = r"G:\ICE\数据下载测试\2019\ATL03\extent_1\ATL03_20190101171428_00650207_006_02.h5"
#     atl08Path = r"G:\ICE\数据下载测试\2019\ATL08\extent_1\ATL08_20190101171428_00650207_006_02.h5"

#     gtx = "gt1l"
#     loader = ATLDataLoader(atl03Path, atl08Path, gtx)
#     extents = F.get_feature_extents(r"G:\ICE\数据下载测试\data\tibet.shp")
    
#     loader.extract_08_ground(extents[1])
#     # 将结果保存到CSV
#     loader.save_to_csv_with_progress("output.csv","08")
    
#     loader.create_shapefile_from_df("08", r"D:\青藏高原地形超分\青藏高原\数据下载测试\Code\test\test.gdb\data", "lon", "lat")

#     # 作图
#     # 颜色参考ATL08官方颜色
#     class_dict = {
#         -1: {'color': (194 / 255, 197 / 255, 204 / 255), 'name': 'Unclassified'},
#         0: {'color': (194 / 255, 197 / 255, 204 / 255), 'name': 'Noise'},
#         1: {'color': (210 / 255, 184 / 255, 38 / 255), 'name': 'Ground'},
#         2: {'color': (69 / 255, 128 / 255, 26 / 255), 'name': 'Canopy'},
#         3: {'color': (133 / 255, 243 / 255, 52 / 255), 'name': 'Top of canopy'}
#     }
#     csv_path = 'output.csv'
#     visualizer = DataVisualizer(csv_path, class_dict)
#     visualizer.plot()
#     print('数据处理完成！')
    
    
atl03folder = r"G:\ICE\数据下载测试\2019\ATL03"
atl08folder = r"G:\ICE\数据下载测试\2019\ATL08"
shpfolder = r"G:\ICE\数据下载测试\2019\Point"
csv_path = r"G:\ICE\数据下载测试\2019\CSV"
tibetshp = r"G:\青藏高原\青藏高原渔网\Tibet0_45.shp" 
uniqueids, extents = F.get_feature_extents(tibetshp)
for i in range(4,10):
    uniqueid = uniqueids[i]
    extent = extents[i]

    atl03extentfolder = os.path.join(atl03folder, f"extent_{uniqueid}")
    atl08extentfolder = os.path.join(atl08folder, f"extent_{uniqueid}")
    csvextentfolder = os.path.join(csv_path, f"extent_{uniqueid}")
    gdb_name = f"extent_{uniqueid}"
    shpextentfolder = os.path.join(shpfolder, f"{gdb_name}.gdb")

    # 创建 CSV 目录
    if not os.path.exists(csvextentfolder):
        os.makedirs(csvextentfolder)
    
    # 创建 GDB 目录
    if not os.path.exists(shpextentfolder):
        arcpy.CreateFileGDB_management(shpfolder, gdb_name)

    alt08files = get_tif_files(atl08extentfolder)

    for alt08file in alt08files:
        name08 = os.path.basename(alt08file).split(".h5")[0]
        name03 = name08.replace("ATL08", "ATL03")
        alt03file = os.path.join(atl03extentfolder, f"{name03}.h5")

        csvFile = os.path.join(csvextentfolder, f"{name08}.csv")
        shpfile = os.path.join(shpextentfolder, name08)

        # 如果 CSV 和 Shapefile 都存在，跳过处理
        if os.path.exists(csvFile) and arcpy.Exists(shpfile):
            print(f"文件 {name08} 已处理，跳过。")
            continue

        # 处理数据
        if os.path.exists(alt03file):
            print("Processing file:", name08)
            D = []

            for beam in ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']:
                loader = ATLDataLoader(alt03file, alt08file, beam)
                data_08 = loader.extract_08_ground(extents[i])
                D.append(data_08)

            newData = pd.concat(D, ignore_index=True)

            # 保存 CSV
            if not os.path.exists(csvFile):
                loader.save_to_csv_with_progress(newData, csvFile)

            # 保存 Shapefile
            if not os.path.exists(shpfile):
                loader.create_shapefile_from_df(newData, shpfile, "lon", "lat")

            
        
    

    '''
    批处理：循环处理六个激光波束

    if __name__ == "__main__":
    import os
    # 测试文件路径
    atl03Path = "D:/other/icesat/text/1/shan/processed_ATL03_20200915100047_12530806_006_02.h5"
    atl08Path = "D:/other/icesat/text/2/processed_ATL08_20200915100047_12530806_006_02.h5"
    # 循环处理六个激光波束
    for beam in ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']:
        print("Processing beam:", beam)
        # 创建 ATLDataLoader 对象
        loader = ATLDataLoader(atl03Path, atl08Path, beam) 
        # 将结果保存到 CSV 文件
        csvFile = atl03Path.replace(".h5", '_' + beam + ".csv").replace(".hdf", '_' + beam + ".csv")
        loader.df.to_csv(csvFile, index=False)
        print("Results saved to:", csvFile)
    '''