# 文档说明：本文件的网络定义、输入形状与使用约定。
"""SDF-Net with only the selected statistical fingerprint branch active.

Only fingerprints enter the network. Predictions are xyz in source units.
Masks use True for a real path; PyTorch key_padding_mask uses the inverse.
"""
import torch  # 导入PyTorch张量计算、自动求导及CUDA接口
from torch import nn  # 导入神经网络模块接口


class SpatialNorm(nn.Module):  # 定义逐阵元执行通道LayerNorm
    """Channel LayerNorm at each array element (independent of batch size)."""  # 对每个天线空间位置的通道做LayerNorm，不依赖批大小的统计量
    def __init__(self, width):  # 初始化逐阵元执行通道LayerNorm的参数与子模块
        super().__init__()  # 初始化父类
        self.norm = nn.LayerNorm(width)  # 对每个空间位置的通道做归一化

    def forward(self, x):  # 定义逐阵元执行通道LayerNorm的前向计算
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()  # 调整通道轴、归一化并恢复原布局


class StatisticalBlock(nn.Module):  # 定义统计流二维残差卷积块
    def __init__(self, width, dropout):  # 初始化统计流二维残差卷积块的参数与子模块
        super().__init__()  # 初始化父类
        self.block = nn.Sequential(  # 构建顺序计算模块
            SpatialNorm(width), nn.Conv2d(width, width, 3, padding=1), nn.GELU(),  # 第一层归一化、卷积和激活
            nn.Dropout(dropout), SpatialNorm(width),  # 随机丢弃并再次归一化
            nn.Conv2d(width, width, 3, padding=1))  # 第二层三乘三卷积

    def forward(self, x):  # 定义统计流二维残差卷积块的前向计算
        return x + self.block(x)  # 残差相加并返回


class PhysicalBlock(nn.Module):  # 定义带路径掩码的一维残差卷积块
    """Per-token norms and masked convolutions prevent padded paths leaking in."""  # 逐路径归一化并在卷积各阶段屏蔽填充，防止填充值影响有效路径
    def __init__(self, width, dropout):  # 初始化带路径掩码的一维残差卷积块的参数与子模块
        super().__init__()  # 初始化父类
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)  # 建立两层路径 token 归一化
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)  # 建立第一层一维卷积
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)  # 建立第二层一维卷积
        self.activation, self.dropout = nn.GELU(), nn.Dropout(dropout)  # 建立激活与随机丢弃层

    def forward(self, x, valid):  # 定义带路径掩码的一维残差卷积块的前向计算
        m = valid.unsqueeze(-1)  # 扩展有效路径掩码的维度
        h = self.norm1(x).masked_fill(~m, 0)  # 归一化并清零无效路径
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)  # 调整轴顺序以卷积路径序列
        h = self.dropout(self.activation(h)).masked_fill(~m, 0)  # 激活、丢弃并再次清零无效路径
        h = self.norm2(h).masked_fill(~m, 0)  # 再次归一化和清零
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)  # 第二次一维卷积并恢复轴顺序
        return (x + h).masked_fill(~m, 0)  # 残差相加后清零无效路径


class MaskedPool(nn.Module):  # 定义屏蔽无效token并执行注意力或均值池化
    def __init__(self, width, mode="attention"):  # 初始化屏蔽无效token并执行注意力或均值池化的参数与子模块
        super().__init__()  # 初始化父类
        self.mode = mode  # 保存池化模式
        self.score = nn.Linear(width, 1) if mode == "attention" else None  # 注意力模式建立可学习打分层

    def forward(self, x, valid):  # 定义屏蔽无效token并执行注意力或均值池化的前向计算
        x = x.masked_fill(~valid.unsqueeze(-1), 0)  # 清零无效 token
        if self.score is None:  # 判断是否使用均值池化
            weights = valid.to(x.dtype)  # 均值池化给有效 token 权重一
        else:  # 当前面判断不成立时执行下面的另一分支
            logits = self.score(x).squeeze(-1).masked_fill(~valid, -1e4)  # 计算分数并屏蔽无效 token
            weights = torch.softmax(logits, dim=1) * valid.to(x.dtype)  # 归一化注意力分数并清零无效权重
        # All-missing physical branches return exactly zero, with no NaNs.
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)  # 再次归一化并防止全空分支除零
        return (x * weights.unsqueeze(-1)).sum(1)  # 加权求和得到流摘要


class SDFNet(nn.Module):  # 定义统计与物理双流定位网络
    def __init__(self, physical_dim=5, max_paths=15, cnn_width=32, cnn_blocks=2,  # 初始化统计与物理双流定位网络的参数与子模块
                 d_model=96, num_heads=4, num_layers=2, ffn_ratio=2,  # 配置 Transformer 宽度、头数、层数
                 dropout=0.1, fusion="gated", pooling="attention",  # 配置丢弃比例、融合和池化
                 branch="stat", shortcut=True):  # 配置branch、shortcut，具体取值见本行
        super().__init__()  # 初始化父类
        if min(physical_dim, max_paths, cnn_width, d_model, num_heads, ffn_ratio) < 1:  # 检查主要维度是否为正
            raise ValueError("Model dimensions must be positive")  # 维度错误则报错
        if max_paths > 15 or min(cnn_blocks, num_layers) < 0 or d_model % num_heads:  # 检查路径数、层数和头数整除关系
            raise ValueError("Invalid path count, block count or attention dimensions")  # 配置错误则报错
        if not 0 <= dropout < 1 or fusion not in ("gated", "concat"):  # 检查丢弃比例与融合方式
            raise ValueError("Invalid dropout or fusion")  # 配置错误则报错
        if pooling not in ("attention", "mean") or branch not in ("both", "stat", "physical"):  # 检查池化方式与分支选项
            raise ValueError("Invalid pooling or branch")  # 配置错误则报错
        self.physical_dim, self.max_paths = physical_dim, max_paths  # 保存物理维数和最大路径数
        self.fusion, self.branch, self.shortcut = fusion, branch, shortcut  # 保存融合、分支和捷径配置
        self.stat_stem = nn.Conv2d(7, cnn_width, 1)  # 用一点一卷积投影统计输入
        self.stat_blocks = nn.Sequential(*[  # 开始构建统计流卷积块列表
            StatisticalBlock(cnn_width, dropout) for _ in range(cnn_blocks)])  # 根据配置堆叠统计 CNN 残差块
        self.stat_projection = nn.Linear(cnn_width, d_model)  # 投影统计 token 到模型宽度
        self.phys_stem = nn.Linear(physical_dim, cnn_width)  # 将物理路径特征映射到卷积宽度
        self.phys_blocks = nn.ModuleList([  # 开始构建物理流卷积块列表
            PhysicalBlock(cnn_width, dropout) for _ in range(cnn_blocks)])  # 根据配置堆叠物理 CNN 残差块
        self.phys_projection = nn.Linear(cnn_width, d_model)  # 投影物理 token 到模型宽度
        self.stat_pool, self.phys_pool = MaskedPool(d_model, pooling), MaskedPool(d_model, pooling)  # 分别建立两流的早期池化层
        self.gate = (nn.Sequential(nn.Linear(2 * d_model + 1, d_model), nn.GELU(),  # 门控融合先读取两流摘要和路径比例
                                   nn.Linear(d_model, 2)) if fusion == "gated" else None)  # 输出两路门控分数；拼接模式不建门控层
        self.stat_position = nn.Parameter(torch.empty(1, 16, d_model))  # 建立十六个统计位置编码
        self.path_position = nn.Parameter(torch.empty(1, max_paths, d_model))  # 建立物理路径位置编码
        self.modality = nn.Parameter(torch.empty(2, d_model))  # 建立两种输入模态编码
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))  # 建立 CLS 汇总 token
        self.layers = nn.ModuleList([  # 开始构建当前模型的编码层列表，各层独立注册参数
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * ffn_ratio,  # 设置注意力头数与前馈宽度
                                       dropout=dropout, activation="gelu",  # 设置丢弃比例和激活函数
                                       batch_first=True, norm_first=True)  # 设置批量优先与前置归一化
            for _ in range(num_layers)])  # 按配置的编码层数完成模块列表，具体层类型见前面定义
        self.output_norm = nn.LayerNorm(d_model)  # 归一化编码器输出
        self.final_stat_pool = MaskedPool(d_model, pooling)  # 建立统计 token 最终池化层
        self.final_phys_pool = MaskedPool(d_model, pooling)  # 建立物理 token 最终池化层
        self.head = nn.Sequential(nn.LayerNorm(3 * d_model),  # 定位头归一化拼接的三路摘要
                                  nn.Linear(3 * d_model, d_model), nn.GELU(),  # 压缩摘要并激活
                                  nn.Dropout(dropout), nn.Linear(d_model, 3))  # 丢弃后输出三维坐标
        self.local_head = nn.Linear(2 * d_model, 3) if shortcut else None  # 可选的双流早期预测捷径
        self.stat_aux, self.phys_aux = nn.Linear(d_model, 3), nn.Linear(d_model, 3)  # 建立两流辅助预测头
        self.register_buffer("target_center", torch.zeros(3))  # 保存目标坐标中心值
        self.register_buffer("target_scale", torch.ones(()))  # 保存目标坐标缩放值
        for value in (self.stat_position, self.path_position, self.modality, self.cls):  # 遍历可学习位置和模态编码
            nn.init.normal_(value, std=0.02)  # 用小方差正态分布初始化编码
        for layer in (self.head[-1], self.local_head, self.stat_aux, self.phys_aux):  # 遍历各坐标预测层
            if layer is not None:  # 跳过未启用的捷径层
                nn.init.normal_(layer.weight, std=0.01)  # 初始化预测层权重
                nn.init.zeros_(layer.bias)  # 初始化预测层偏置

    def set_target_transform(self, center, scale):  # 定义函数：写入坐标中心与缩放缓冲区
        with torch.no_grad():  # 评估或EMA更新时不构建梯度图，以节省内存且不反向传播
            center = torch.as_tensor(center, dtype=self.target_center.dtype,  # 转换中心值的数据类型
                                     device=self.target_center.device)  # 将中心值放到目标设备
            scale = torch.as_tensor(scale, dtype=self.target_scale.dtype,  # 转换缩放值的数据类型
                                    device=self.target_scale.device)  # 将缩放值放到目标设备
            if center.shape != (3,) or scale.numel() != 1:  # 检查中心值与缩放值形状
                raise ValueError("Target center must have 3 values and scale must be scalar")  # 形状错误则报错
            if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or scale.item() <= 0:  # 检查数值有限且缩放值大于零
                raise ValueError("Invalid target transform")  # 非法目标变换则报错
            self.target_center.copy_(center)  # 写入目标中心值
            self.target_scale.copy_(scale.reshape(()))  # 写入目标缩放值

    def forward(self, statistical, physical, path_mask, return_aux=False):  # 定义统计与物理双流定位网络的前向计算
        b = statistical.shape[0]  # 读取批大小
        if statistical.shape != (b, 7, 4, 4):  # 核查统计输入形状
            raise ValueError("Expected statistical [B,7,4,4]")  # 统计输入错误则报错
        if physical.shape != (b, self.max_paths, self.physical_dim):  # 核查物理输入形状
            raise ValueError("Unexpected physical input shape")  # 物理输入错误则报错
        if path_mask.shape != physical.shape[:2] or path_mask.dtype != torch.bool:  # 核查路径掩码形状和类型
            raise ValueError("Expected boolean path_mask [B,K], True means valid")  # 掩码错误则报错
        stat_valid = torch.ones((b, 16), dtype=torch.bool, device=statistical.device)  # 默认十六个统计位置均有效
        valid = path_mask  # 将输入掩码作为有效物理路径
        if self.branch == "stat":  # 统计单流消融时关闭物理流
            valid = torch.zeros_like(valid)  # 将物理路径标记为无效
        elif self.branch == "physical":  # 物理单流消融时关闭统计流
            stat_valid = torch.zeros_like(stat_valid)  # 将统计位置标记为无效
            statistical = torch.zeros_like(statistical)  # 将统计输入清零
        # masked_fill (not multiplication) also neutralizes NaN in unused slots.
        physical = physical.masked_fill(~valid.unsqueeze(-1), 0)  # 清零填充路径避免参与卷积
        s = self.stat_blocks(self.stat_stem(statistical)).flatten(2).transpose(1, 2)  # 统计输入经二维卷积残差块后展开为16个token，此处没有CA计算
        s = self.stat_projection(s).masked_fill(~stat_valid.unsqueeze(-1), 0)  # 投影统计 token 并屏蔽无效位置
        p = self.phys_stem(physical).masked_fill(~valid.unsqueeze(-1), 0)  # 物理输入先经线性层并屏蔽填充路径
        for block in self.phys_blocks:  # 遍历物理流残差卷积块
            p = block(p, valid)  # 更新有效路径特征
        p = self.phys_projection(p).masked_fill(~valid.unsqueeze(-1), 0)  # 投影物理 token 并清零无效位置
        sg, pg = self.stat_pool(s, stat_valid), self.phys_pool(p, valid)  # 分别汇聚两路早期摘要
        availability = torch.stack((stat_valid.any(1), valid.any(1)), dim=1)  # 判断两路输入是否有效
        if self.gate is None:  # 非门控模式使用固定融合分支
            gate = availability.to(s.dtype)  # 把有效标记转为数值权重
        else:  # 当前面判断不成立时执行下面的另一分支
            count = valid.to(s.dtype).mean(1, keepdim=True)  # 计算有效路径占比
            logits = self.gate(torch.cat((sg, pg, count), dim=1))  # 由两路摘要和路径占比计算门控分数
            gate = torch.softmax(logits.masked_fill(~availability, -1e4), dim=1)  # 屏蔽缺失分支并做 softmax
            gate = gate * availability.to(gate.dtype)  # 强制无效分支权重为零
            gate = gate / gate.sum(1, keepdim=True).clamp_min(1e-8)  # 重新归一化两路权重
        s = (s + self.stat_position + self.modality[0]) * gate[:, 0, None, None]  # 给统计 token 加位置模态编码并乘门控
        p = (p + self.path_position + self.modality[1]) * gate[:, 1, None, None]  # 给物理 token 加位置模态编码并乘门控
        s = s.masked_fill(~stat_valid.unsqueeze(-1), 0)  # 再次清零无效统计 token
        p = p.masked_fill(~valid.unsqueeze(-1), 0)  # 再次清零无效物理 token
        tokens = torch.cat((self.cls.expand(b, -1, -1), s, p), dim=1)  # 合并 CLS、统计和物理 token
        token_valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=s.device),  # 建立始终有效的 CLS 掩码
                                 stat_valid, valid), dim=1)  # 拼接统计与物理 token 掩码
        for layer in self.layers:  # 按顺序执行当前模型实际定义的编码层
            tokens = layer(tokens, src_key_padding_mask=~token_valid)  # 执行带路径屏蔽的Transformer编码层，包括MHSA和前馈计算
            tokens = tokens.masked_fill(~token_valid.unsqueeze(-1), 0)  # 每层后清零无效 token
        tokens = self.output_norm(tokens)  # 归一化融合后的 token
        summary = torch.cat((tokens[:, 0], self.final_stat_pool(tokens[:, 1:17], stat_valid),  # 拼接 CLS 摘要与统计流最终池化
                             self.final_phys_pool(tokens[:, 17:], valid)), dim=1)  # 再拼接物理流最终池化
        prediction = self.head(summary)  # 经定位头预测归一化三维坐标
        if self.local_head is not None:  # 判断是否启用早期预测捷径
            prediction = prediction + self.local_head(torch.cat((sg, pg), dim=1))  # 将捷径预测加到主预测
        decode = lambda x: x * self.target_scale + self.target_center  # 定义坐标反归一化计算
        prediction = decode(prediction)  # 还原原单位的三维坐标
        if not return_aux:  # 判断是否仅需最终预测
            return prediction  # 只返回三维坐标
        return dict(prediction=prediction, stat_prediction=decode(self.stat_aux(sg)),  # 返回主预测与统计流辅助预测
                    physical_prediction=decode(self.phys_aux(pg)), gate=gate,  # 追加物理流辅助预测和门控权重
                    stat_available=stat_valid.any(1), physical_available=valid.any(1))  # 追加双流有效性标记
