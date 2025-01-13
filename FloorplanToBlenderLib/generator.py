# 导入必要的模块和库
import abc
import cv2
import math
import numpy as np

# 自定义模块的导入
from . import detect        # 负责检测功能的模块
from . import transform     # 负责几何变换的模块
from . import IO            # 负责输入输出操作的模块
from . import const         # 常量定义模块
from . import draw          # 绘图模块
from . import calculate     # 数值计算模块

"""
Generator
This file contains structures for different floorplan detection features.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
核心功能概述
基类 Generator:

定义了通用的生成逻辑和接口。
包含了关键的几何计算方法，比如 get_shape()，用于根据点集（顶点 verts）计算3D形状的大小。
子类职责:

Floor: 生成地板的3D模型。
Wall: 生成墙壁的3D模型，包括垂直和水平的墙面。
Room: 生成房间区域的3D模型。
Door: 检测门的位置，生成门的3D模型。
模块功能分工:

detect: 提供特征检测功能，例如外部轮廓、墙壁、房间和门的检测。
transform: 实现从检测结果到3D模型数据（顶点和面）的转换，包括缩放和平移。
IO: 保存生成的数据到文件中，供后续处理或导出到3D建模工具（如 Blender）。
calculate: 包括数学和几何计算方法，如欧几里得距离计算和点的归一化。
draw: 可视化调试工具，用于绘制和显示检测结果。
常量管理:

const: 定义了多个常量值（如墙高、像素到3D单位的比例、门宽等），为代码提供配置和约束。
核心逻辑说明
1. Generator.get_shape() 方法
计算顶点的范围并按比例缩放：
python
复制代码
def get_shape(self, verts):
    ...
    return [
        (high[0] - low[0]) * self.scale[0],
        (high[1] - low[1]) * self.scale[1],
        (high[2] - low[2]) ** self.scale[2],
    ]
目的: 确定生成物体的3D尺寸，返回一个包含 [宽, 高, 深] 的列表。
2. 各子类的 generate() 方法
每个子类实现了自己的 generate() 方法，以下是重点：

Floor.generate(): 从灰度图检测外轮廓并生成地板的顶点和面。
Wall.generate():
通过过滤器提取墙体图像并检测精确的盒子（墙壁）。
利用 transform 模块生成墙体的顶点和面。
Room.generate():
检测房间的区域，生成房间的顶点和面。
Door.generate():
检测门的位置，计算门的轮廓和连接点，最终生成门的3D模型。
"""


class Generator:
    __metaclass__ = abc.ABCMeta  # 定义抽象基类

    # 3D点的集合，用于创建网格
    verts = []
    # 每个平面的面集合，描述生成网格的顺序
    faces = []
    # 墙的高度
    height = const.WALL_HEIGHT
    # 将像素值缩放为3D位置的比例
    pixelscale = const.PIXEL_TO_3D_SCALE
    # 对象的缩放比例
    scale = np.array([1, 1, 1])
    # 文件路径，用于保存结果
    path = ""

    def __init__(self, gray, path, scale, info=False):
        # 初始化路径、形状和缩放比例
        self.path = path
        self.shape = self.generate(gray, info)
        self.scale = scale

    def get_shape(self, verts):
        """
        verts = [
            [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],  # 底面
            [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]   # 顶面
        ]
        [1.0, 2.0, 4.0]
        获取形状信息，并按指定比例调整盒子的大小。
        @param verts: 输入的盒子顶点集合
        @return: 调整比例后的盒子大小
        """
        if len(verts) == 0:
            return [0, 0, 0]

        poslist = transform.verts_to_poslist(verts)  # 转换为位置列表
        high = [0, 0, 0]
        low = poslist[0]

        for pos in poslist:
            # 更新最高点和最低点
            high = [max(high[i], pos[i]) for i in range(3)]
            low = [min(low[i], pos[i]) for i in range(3)]

        # 计算并返回缩放后的大小
        return [
            (high[0] - low[0]) * self.scale[0],
            (high[1] - low[1]) * self.scale[1],
            (high[2] - low[2]) * self.scale[2],
        ]

    @abc.abstractmethod
    def generate(self, gray, info=False):
        """抽象方法，用于生成数据"""
        pass


class Floor(Generator):
    def __init__(self, gray, path, scale, info=False):
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        # 检测外轮廓（简单的地板或屋顶解决方案）
        contour, _ = detect.outer_contours(gray)

        # 创建顶点，按比例缩放
        self.verts = transform.scale_point_to_vector(
            boxes=contour,
            scale=self.scale,
            pixelscale=self.pixelscale,
            height=self.height,
        )

        # 创建面
        '''
        # 定义一些示例顶点
        verts = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # 顶面
        ]
        
        # 正确地定义面，这里假设我们想要创建一个立方体，使用三角形面
        faces = [
            [0, 1, 2], [2, 3, 0],  # 底面
            [4, 5, 6], [6, 7, 4],  # 顶面
            [0, 4, 5], [5, 1, 0],  # 前面
            [2, 6, 7], [7, 3, 2],  # 后面
            [0, 2, 6], [6, 4, 0],  # 左面
            [5, 7, 3], [3, 1, 5]   # 右面
        ]
        '''
        self.faces = list(range(len(self.verts)))

        if info:
            print("近似公寓大小 :", cv2.contourArea(contour))

        # 保存顶点和面到文件
        IO.save_to_file(self.path + const.FLOOR_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.FLOOR_FACES, self.faces, info)

        return self.get_shape(self.verts)


class Wall(Generator):
    def __init__(self, gray, path, scale, info=False):
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        # 创建墙的二值图像（滤除小的物体）
        wall_img = detect.wall_filter(gray)

        # 检测墙的盒子
        boxes, _ = detect.precise_boxes(wall_img)

        # 检测外轮廓
        contour, _ = detect.outer_contours(gray)

        # 移除轮廓外的墙
        boxes = calculate.remove_walls_not_in_contour(boxes, contour)

        # 将盒子转换为垂直的顶点和面
        self.verts, self.faces, wall_amount = transform.create_nx4_verts_and_faces(
            boxes=boxes,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
        )

        if info:
            print("创建的墙数量 :", wall_amount)

        # 保存垂直顶点和面
        IO.save_to_file(self.path + const.WALL_VERTICAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WALL_VERTICAL_FACES, self.faces, info)

        # 将盒子转换为水平的顶点和面
        self.verts, self.faces, wall_amount = transform.create_4xn_verts_and_faces(
            boxes=boxes,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
        )

        # 保存水平顶点和面
        IO.save_to_file(self.path + const.WALL_HORIZONTAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WALL_HORIZONTAL_FACES, self.faces, info)

        return self.get_shape(self.verts)


class Room(Generator):
    def __init__(self, gray, path, scale, info=False):
        # 将房间略微高于地板
        self.height = const.WALL_HEIGHT - const.ROOM_FLOOR_DISTANCE
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        gray = detect.wall_filter(gray)  # 过滤墙的图像
        gray = ~gray  # 取反
        rooms, colored_rooms = detect.find_rooms(gray.copy())  # 检测房间
        gray_rooms = cv2.cvtColor(colored_rooms, cv2.COLOR_BGR2GRAY)

        # 获取房间的盒子位置
        boxes, gray_rooms = detect.precise_boxes(gray_rooms, gray_rooms)

        # 创建房间的顶点和面
        self.verts, self.faces, counter = transform.create_4xn_verts_and_faces(
            boxes=boxes,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
        )

        if info:
            print("检测到的房间数量 :", counter)

        IO.save_to_file(self.path + const.ROOM_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.ROOM_FACES, self.faces, info)

        return self.get_shape(self.verts)


class Door(Generator):
    def __init__(self, gray, path, image_path, scale_factor, scale, info=False):
        self.image_path = image_path  # 图像路径
        self.scale_factor = scale_factor  # 缩放因子
        super().__init__(gray, path, scale, info)

    def get_point_the_furthest_away(self, door_features, door_box):
        """
        计算门上距离门框最远的点
        """
        best_point = None
        dist = 0
        center = calculate.box_center(door_box)

        for f in door_features:
            distance = abs(calculate.euclidean_distance_2d(center, f))
            if best_point is None or dist < distance:
                best_point = f
                dist = distance
        return best_point

    def get_closest_box_point_to_door_point(self, wall_point, box):
        """
        计算离门点最近的盒子点
        """
        best_point = None
        dist = math.inf

        (x, y, w, h) = cv2.boundingRect(box)  # 计算盒子的边界矩形

        # 获取盒子的边点
        box_side_points = (
            [[x + w / 2, y], [x + w / 2, y + h]]
            if w < h
            else [[x, y + h / 2], [x + w, y + h / 2]]
        )

        for fp in box_side_points:
            distance = calculate.euclidean_distance_2d(wall_point, fp)
            if best_point is None or distance < dist:
                best_point = fp
                dist = distance

        return (int(best_point[0]), int(best_point[1]))

    def generate(self, gray, info=False):
        doors = detect.doors(self.image_path, self.scale_factor)  # 检测门

        door_contours = []
        for door in doors:
            door_features = door[0]  # 门的特征点
            door_box = door[1]  # 门框

            # 计算最远的点和最近的点

            # find door to space point
            space_point = self.get_point_the_furthest_away(door_features, door_box)

            # find best box corner to use as attachment
            closest_box_point = self.get_closest_box_point_to_door_point(
                space_point, door_box
            )

            # Calculate normal
            normal_line = [
                space_point[0] - closest_box_point[0],
                space_point[1] - closest_box_point[1],
            ]

            # Normalize point
            normal_line = calculate.normalize_2d(normal_line)

            # Create door contour
            x1 = closest_box_point[0] + normal_line[1] * const.DOOR_WIDTH
            y1 = closest_box_point[1] - normal_line[0] * const.DOOR_WIDTH

            x2 = closest_box_point[0] - normal_line[1] * const.DOOR_WIDTH
            y2 = closest_box_point[1] + normal_line[0] * const.DOOR_WIDTH

            x4 = space_point[0] + normal_line[1] * const.DOOR_WIDTH
            y4 = space_point[1] - normal_line[0] * const.DOOR_WIDTH

            x3 = space_point[0] - normal_line[1] * const.DOOR_WIDTH
            y3 = space_point[1] + normal_line[0] * const.DOOR_WIDTH

            c1 = [int(x1), int(y1)]
            c2 = [int(x2), int(y2)]
            c3 = [int(x3), int(y3)]
            c4 = [int(x4), int(y4)]

            door_contour = np.array([[c1], [c2], [c3], [c4]], dtype=np.int32)
            door_contours.append(door_contour)

        if const.DEBUG_DOOR:
            print("Showing DEBUG door. Press any key to continue...")
            img = draw.contoursOnImage(gray, door_contours)
            draw.image(img)

        # Create verts for door

        self.verts, self.faces, door_amount = transform.create_nx4_verts_and_faces(
            boxes=door_contours,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
        )

        if info:
            print("Doors created : ", int(door_amount / 4))

        IO.save_to_file(self.path + "door_vertical_verts", self.verts, info)
        IO.save_to_file(self.path + "door_vertical_faces", self.faces, info)

        self.verts, self.faces, door_amount = transform.create_4xn_verts_and_faces(
            boxes=door_contours,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
            ground_height=const.WALL_GROUND,
        )

        # One solution to get data to blender is to write and read from file.
        IO.save_to_file(self.path + "door_horizontal_verts", self.verts, info)
        IO.save_to_file(self.path + "door_horizontal_faces", self.faces, info)

        return self.get_shape(self.verts)


class Window(Generator):
    # TODO: also fill small gaps between windows and walls
    # TODO: also add verts for filling gaps

    def __init__(self, gray, path, image_path, scale_factor, scale, info=False):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.scale = scale
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        windows = detect.windows(self.image_path, self.scale_factor)

        # Create verts for window, vertical
        v, self.faces, window_amount1 = transform.create_nx4_verts_and_faces(
            boxes=windows,
            height=const.WINDOW_MIN_MAX_GAP[0],
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=0,
        )  # create low piece
        v2, self.faces, window_amount2 = transform.create_nx4_verts_and_faces(
            boxes=windows,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=const.WINDOW_MIN_MAX_GAP[1],
        )  # create higher piece

        self.verts = v
        self.verts.extend(v2)
        parts_per_window = 2
        window_amount = len(v) / parts_per_window

        if info:
            print("Windows created : ", int(window_amount))

        IO.save_to_file(self.path + const.WINDOW_VERTICAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WINDOW_VERTICAL_FACES, self.faces, info)

        # horizontal

        v, f, _ = transform.create_4xn_verts_and_faces(
            boxes=windows,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
            ground_height=const.WALL_GROUND,
        )
        v2, f2, _ = transform.create_4xn_verts_and_faces(
            boxes=windows,
            height=const.WINDOW_MIN_MAX_GAP[0],
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
            ground_height=const.WINDOW_MIN_MAX_GAP[1],
        )

        self.verts = v
        self.verts.extend(v2)
        self.faces = f
        self.faces.extend(f2)

        # One solution to get data to blender is to write and read from file.
        IO.save_to_file(self.path + const.WINDOW_HORIZONTAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WINDOW_HORIZONTAL_FACES, self.faces, info)

        return self.get_shape(self.verts)
