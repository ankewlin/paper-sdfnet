"""Single-stream SDF-Net ablation: early fusion before any learned layer."""  # 文档说明：在任何可学习层之前先融合输入的单流SDF-Net消融
import torch  # 导入PyTorch张量计算、自动求导及CUDA接口
from torch import nn  # 导入神经网络模块接口



class SpatialNorm(nn.Module):  # 定义逐阵元执行通道LayerNorm
    def __init__(self, width):  # 初始化逐阵元执行通道LayerNorm的参数与子模块
        super().__init__()  # 初始化父类
        self.norm = nn.LayerNorm(width)  # 对每个空间位置的通道做归一化

    def forward(self, x):  # 定义逐阵元执行通道LayerNorm的前向计算
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()  # 调整通道轴、归一化并恢复原布局


class ResidualCNN(nn.Module):  # 定义单流二维残差卷积块
    def __init__(self, width, dropout):  # 初始化单流二维残差卷积块的参数与子模块
        super().__init__()  # 初始化父类
        self.block = nn.Sequential(  # 构建顺序计算模块
            SpatialNorm(width), nn.Conv2d(width, width, 3, padding=1), nn.GELU(),  # 第一层归一化、卷积和激活
            nn.Dropout(dropout), SpatialNorm(width), nn.Conv2d(width, width, 3, padding=1))  # 建立本行指定通道、核大小及步长的二维卷积

    def forward(self, x):  # 定义单流二维残差卷积块的前向计算
        return x + self.block(x)  # 残差相加并返回


class TokenPool(nn.Module):  # 定义单流token注意力或均值汇聚
    def __init__(self, width, mode):  # 初始化单流token注意力或均值汇聚的参数与子模块
        super().__init__()  # 初始化父类
        self.score = nn.Linear(width, 1) if mode == "attention" else None  # 注意力模式建立可学习打分层

    def forward(self, x):  # 定义单流token注意力或均值汇聚的前向计算
        if self.score is None:  # 判断是否使用均值池化
            return x.mean(1)  # 沿token轴取均值，得到每个样本的汇总表示
        return (x * self.score(x).softmax(dim=1)).sum(1)  # 对token计算可学习分数，归一化后加权求和


class SingleStreamSDFNet(nn.Module):  # 定义输入提前融合的单流CNN和Transformer定位模型
    def __init__(self, physical_dim=5, max_paths=15, cnn_width=32, cnn_blocks=2,  # 初始化输入提前融合的单流CNN和Transformer定位模型的参数与子模块
                 d_model=96, num_heads=4, num_layers=2, ffn_ratio=2,  # 配置 Transformer 宽度、头数、层数
                 dropout=0.1, pooling="attention", shortcut=True):  # 配置随机丢弃概率、pooling、shortcut，具体取值见本行
        super().__init__()  # 初始化父类
        if min(physical_dim, max_paths, cnn_width, d_model, num_heads, ffn_ratio) < 1:  # 检查主要维度是否为正
            raise ValueError("Model dimensions must be positive")  # 维度错误则报错
        if max_paths > 15 or min(cnn_blocks, num_layers) < 0 or d_model % num_heads:  # 检查路径数、层数和头数整除关系
            raise ValueError("Invalid paths, block counts or head dimensions")  # 校验失败时抛出异常并停止，错误信息指出对应输入或配置问题
        if not 0 <= dropout < 1 or pooling not in ("attention", "mean"):  # 检查本行所列维度、数量或概率参数的合法范围
            raise ValueError("Invalid dropout or pooling")  # 校验失败时抛出异常并停止，错误信息指出对应输入或配置问题
        self.physical_dim, self.max_paths = physical_dim, max_paths  # 保存物理维数和最大路径数
        self.input_channels = 7 + max_paths * (physical_dim + 1)  # 输入通道数为8个统计量，加所有路径参数和每条路径的有效标记
        self.stem = nn.Conv2d(self.input_channels, cnn_width, 1)  # 用1×1卷积统一处理提前拼接后的全部输入通道
        self.cnn = nn.Sequential(*[ResidualCNN(cnn_width, dropout) for _ in range(cnn_blocks)])  # 按配置堆叠单流二维CNN残差块
        self.projection = nn.Linear(cnn_width, d_model)  # 把CNN输出通道投影为Transformer token宽度
        self.local_pool, self.final_pool = TokenPool(d_model, pooling), TokenPool(d_model, pooling)  # 分别建立进入Transformer前、后的token汇聚层
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))  # 建立 CLS 汇总 token
        self.position = nn.Parameter(torch.empty(1, 17, d_model))  # 为1个CLS和16个阵元token创建可学习位置编码
        self.layers = nn.ModuleList([  # 开始构建当前模型的编码层列表，各层独立注册参数
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * ffn_ratio,  # 设置注意力头数与前馈宽度
                                       dropout=dropout, activation="gelu",  # 设置丢弃比例和激活函数
                                       batch_first=True, norm_first=True)  # 设置批量优先与前置归一化
            for _ in range(num_layers)])  # 按配置的编码层数完成模块列表，具体层类型见前面定义
        self.output_norm = nn.LayerNorm(d_model)  # 归一化编码器输出
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model),  # 为head建立本行指定维度的线性映射
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3))  # 建立本行指定输入和输出维度的线性层及相邻子层
        self.local_head = nn.Linear(d_model, 3) if shortcut else None  # 根据开关建立早期汇聚表示到xyz的局部捷径
        self.aux_head = nn.Linear(d_model, 3)  # 为单流早期汇聚特征建立辅助坐标预测头
        self.register_buffer("target_center", torch.zeros(3))  # 保存目标坐标中心值
        self.register_buffer("target_scale", torch.ones(()))  # 保存目标坐标缩放值
        for value in (self.cls, self.position):  # 遍历本行指定的数据、模块或配置项
            nn.init.normal_(value, std=0.02)  # 用小方差正态分布初始化编码
        for layer in (self.head[-1], self.local_head, self.aux_head):  # 遍历本行指定的数据、模块或配置项
            if layer is not None:  # 跳过未启用的捷径层
                nn.init.normal_(layer.weight, std=0.01)  # 初始化预测层权重
                nn.init.zeros_(layer.bias)  # 初始化预测层偏置

    def set_target_transform(self, center, scale):  # 定义函数：写入坐标中心与缩放缓冲区
        with torch.no_grad():  # 评估或EMA更新时不构建梯度图，以节省内存且不反向传播
            center = torch.as_tensor(center, dtype=self.target_center.dtype, device=self.target_center.device)  # 把训练坐标中心转换到目标缓冲区相同的数据类型与设备
            scale = torch.as_tensor(scale, dtype=self.target_scale.dtype, device=self.target_scale.device)  # 把训练坐标尺度转换到目标缓冲区相同的数据类型与设备
            if center.shape != (3,) or scale.numel() != 1 or not torch.isfinite(center).all():  # 检查本行所列张量维数或形状是否满足当前模型约定
                raise ValueError("Invalid target center or scale shape")  # 校验失败时抛出异常并停止，错误信息指出对应输入或配置问题
            if not torch.isfinite(scale).all() or scale.item() <= 0:  # 检查参与本步计算的数据是否全部为有限值
                raise ValueError("Invalid target scale")  # 校验失败时抛出异常并停止，错误信息指出对应输入或配置问题
            self.target_center.copy_(center)  # 写入目标中心值
            self.target_scale.copy_(scale.reshape(()))  # 写入目标缩放值

    def fuse_input(self, statistical, physical, path_mask):  # 定义函数：将路径参数及有效标记广播到统计网格并在通道维提前融合
        b = statistical.shape[0]  # 读取批大小
        if statistical.shape != (b, 7, 4, 4) or physical.shape != (b, self.max_paths, self.physical_dim):  # 检查本行所列张量维数或形状是否满足当前模型约定
            raise ValueError("Unexpected statistical or physical shape")  # 校验失败时抛出异常并停止，错误信息指出对应输入或配置问题
        if path_mask.shape != physical.shape[:2] or path_mask.dtype != torch.bool:  # 核查路径掩码形状和类型
            raise ValueError("Expected boolean valid-path mask [B,K]")  # 校验失败时抛出异常并停止，错误信息指出对应输入或配置问题
        physical = physical.masked_fill(~path_mask.unsqueeze(-1), 0)  # 将无效路径的所有参数清零，防止填充值进入后续输入
        paths = torch.cat((physical.flatten(1), path_mask.to(physical.dtype)), dim=1)  # 展平所有路径参数并拼接浮点型路径有效标记
        paths = paths[:, :, None, None].expand(-1, -1, 4, 4)  # 把每个路径通道复制到4×4网格的所有阵元位置
        return torch.cat((statistical, paths), dim=1)  # 在通道轴拼接统计网格和广播后的路径数据

    def forward(self, statistical, physical, path_mask, return_aux=False):  # 定义输入提前融合的单流CNN和Transformer定位模型的前向计算
        # Three data tensors, but exactly one learned feature-extraction stream.
        image = self.fuse_input(statistical, physical, path_mask)  # 先融合两类原始指纹，得到唯一CNN流的输入网格
        features = self.cnn(self.stem(image)).flatten(2).transpose(1, 2)  # 融合网格经CNN后展开为16个阵元token
        features = self.projection(features)  # 将每个阵元的CNN特征投影到模型嵌入维度
        local = self.local_pool(features)  # 汇聚进入Transformer前的特征，供局部捷径和辅助头使用
        tokens = torch.cat((self.cls.expand(len(image), -1, -1), features), dim=1) + self.position  # 在16个阵元token前添加CLS，并叠加可学习位置编码
        for layer in self.layers:  # 按顺序执行当前模型实际定义的编码层
            tokens = layer(tokens)  # 将当前token序列送入本轮Transformer编码层
        tokens = self.output_norm(tokens)  # 归一化融合后的 token
        prediction = self.head(torch.cat((tokens[:, 0], self.final_pool(tokens[:, 1:])), dim=1))  # 拼接CLS与最终阵元池化表示，再经主回归头预测坐标
        if self.local_head is not None:  # 判断是否启用早期预测捷径
            prediction = prediction + self.local_head(local)  # 将早期特征的局部坐标预测加到主预测上
        prediction = prediction * self.target_scale + self.target_center  # 用训练目标尺度和中心还原到原始坐标单位
        if not return_aux:  # 判断是否仅需最终预测
            return prediction  # 只返回三维坐标
        return dict(prediction=prediction,  # 开始返回本次计算结果字典，后续续行补充字段
                    aux_prediction=self.aux_head(local) * self.target_scale + self.target_center)  # 返回经相同目标变换还原的单流辅助坐标预测
