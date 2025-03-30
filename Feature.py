import arcpy

def get_feature_extents(feature_class):
    """
    获取要素类中所有要素的矩形范围（extent）。
    :param feature_class: 输入的要素类路径
    :return: 包含每个要素的extent的列表，每个extent是一个字典
    """
    # 创建空列表存储每个要素的extent
    extents = []
    ID = []

    # 使用搜索游标遍历要素
    with arcpy.da.SearchCursor(feature_class, ["SHAPE@", "Id"]) as cursor:
        for row in cursor:
            # 获取要素的几何对象
            geometry = row[0]

            # 获取要素的extent
            extent = geometry.extent
            extents.append((
                 extent.XMin,
                 extent.YMin,
                 extent.XMax,
                 extent.YMax
            ))
            ID.append(row[1])
    return ID, extents

