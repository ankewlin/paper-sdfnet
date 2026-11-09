# 中文阅读路线：本文件只定义网络模块，数据读取和优化循环位于 run.py。
# 双流形状总览（以下以 K=15、cnn_width=32、d_model=96 为例）：
# 统计 [B,8,4,4] → 可选通道倍率 → Conv2d/残差块 → CA → [B,16,32] → [B,16,96]
# 物理 [B,15,5] → Linear/掩码 Conv1d 残差块 → [B,15,32] → [B,15,96]
# 两路各池化得 sg/pg [B,96] → 门控网络生成每个样本的两路权重。
# 两路 token 加位置/模态编码并加权，再与 CLS 拼接成 [B,32,96]。
# 共享 Transformer（内部含 MHSA）→ CLS/统计池化/物理池化拼接 [B,288]
# → 主定位头 [B,3] + local_head(sg,pg) [B,3] → 恢复坐标单位。
# CA 仅位于统计流 CNN 后，处理4×4阵列的行/列；两路都进入共享 Transformer。
# 门控权重、原始统计输入倍率、CA 权重、MHSA 权重是不同层面的加权。

"""SDF-Net: dual-stream CNNs, Coordinate Attention and Transformer MHSA.

Only fingerprints enter the network. Predictions are xyz in source units.
Masks use True for a real path; PyTorch key_padding_mask uses the inverse.

数据流向图（B=批大小，K=最多15条路径，D=d_model；以双流默认设置为例）：

  prepared.npz -> run.py:pack_inputs() -> model.forward()
                    |
                    +-- 统计流 statistical [B,8,4,4]（8个特征 x 4x4阵列）
                    |     -> 输入特征权重（none/learned/fixed）
                    |     -> stat_stem: 1x1 Conv2d
                    |     -> stat_blocks: 残差 3x3 Conv2d（CNN）
                    |     -> coordinate_attention: 行、列注意力（CA，可关闭）
                    |     -> 展平为16个阵列位置 token -> stat_projection [B,16,D]
                    |     -> stat_pool 得到统计流摘要 sg [B,D] ----+
                    |                                           |
                    +-- 物理流 physical [B,K,3或5] + path_mask  |
                          -> phys_stem: Linear                  |
                          -> phys_blocks: 掩码残差 Conv1d（CNN） |
                          -> phys_projection [B,K,D]           |
                          -> phys_pool 得到物理流摘要 pg [B,D] --+
                                                                  |
               [sg, pg, 有效路径比例] -> gate（门控融合权重） ------+
                                                                  |
  统计 token + 位置/模态编码 -> 乘统计门控权重 ----------------------+
  物理 token + 位置/模态编码 -> 乘物理门控权重 ----------------------+
                                                                  |
  [CLS token, 16个统计 token, K个物理 token] -> TransformerEncoderLayer x N
       （每层内部包含 MHSA 多头自注意力 + 前馈网络；mask 遮住填充路径）
       -> output_norm -> [CLS, 统计池化, 物理池化] 拼接 [B,3D]
       -> head -> 归一化坐标 [B,3]
       -> 若启用 shortcut，加上 local_head([sg,pg]) 的预测
       -> target_scale / target_center 反变换 -> (x,y,z) [B,3]

  return_aux=True 时，另返回两路辅助预测、门控权重及分支可用标记。
  branch=stat/physical 为消融设置；上图展示 branch=both 的完整双流。
"""
import torch  # 导入张量计算库
from torch import nn  # 导入神经网络模块接口

class SpatialNorm(nn.Module):  # 定义二维特征图的通道归一化模块
    """Channel LayerNorm at each array element (independent of batch size)."""  # 对每个天线空间位置的通道做LayerNorm，不依赖批大小的统计量
    def __init__(self, width):  # 接收输入通道宽度
        super().__init__()  # 初始化父类
        self.norm = nn.LayerNorm(width)  # 对每个空间位置的通道做归一化

    def forward(self, x):  # 定义前向计算
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()  # 调整通道轴、归一化并恢复原布局


class StatisticalBlock(nn.Module):  # 定义统计流残差卷积块 残差连接是为了保留已有特征，让卷积块学习还需要调整什么，同时帮助梯度传播。
    def __init__(self, width, dropout):  # 接收卷积宽度与丢弃比例
        super().__init__()  # 初始化父类
        self.block = nn.Sequential(  # 构建顺序计算模块
            SpatialNorm(width), nn.Conv2d(width, width, 3, padding=1), nn.GELU(),  # 第一层归一化、卷积和激活
            nn.Dropout(dropout), SpatialNorm(width),  # 随机丢弃并再次归一化
            nn.Conv2d(width, width, 3, padding=1))  # 第二层三乘三卷积

    def forward(self, x):  # 定义卷积块前向计算
        return x + self.block(x)  # 残差相加并返回


class CoordinateAttention(nn.Module):  # 定义统计流坐标注意力
    def __init__(self, channels, reduction=8):  # 接收通道数与压缩比
        super().__init__()  # 初始化父类
        hidden = max(8, channels // reduction)  # 确定中间通道数
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)  # 用一点一卷积压缩通道
        self.norm = nn.GroupNorm(1, hidden)  # 归一化中间特征
        self.activation = nn.GELU()  # 设置激活函数
        self.height_attention = nn.Conv2d(hidden, channels, 1)  # 建立行方向权重生成层
        self.width_attention = nn.Conv2d(hidden, channels, 1)  # 建立列方向权重生成层

    def forward(self, x, *args):  # 定义坐标注意力前向计算
        height, width, *_ = (*x.shape[2:], *args) #可变参数列表接受双流
        along_height = x.mean(dim=3, keepdim=True)  # 沿宽度求均值得到行上下文
        along_width = x.mean(dim=2, keepdim=True).transpose(2, 3)  # 沿高度求均值并转置得到列上下文
        context = self.activation(self.norm(self.reduce(torch.cat((along_height, along_width), dim=2))))  # 拼接行列上下文并做压缩、归一化、激活
        height_context, width_context = torch.split(context, (height, width), dim=2)  # 把上下文拆回行和列
        height_weights = torch.sigmoid(self.height_attention(height_context))  # 生成行注意力权重
        width_weights = torch.sigmoid(self.width_attention(width_context)).transpose(2, 3)  # 生成列注意力权重并恢复方向
        return x * height_weights * width_weights  # 用行列权重重新加权原特征

# 5 物理流用的是一维CNN
class PhysicalBlock(nn.Module):  # 定义物理流掩码残差卷积块 残差连接是为了保留已有特征，让卷积块学习还需要调整什么，同时帮助梯度传播。
    """Per-token norms and masked convolutions prevent padded paths leaking in."""  # 逐路径归一化并在卷积各阶段屏蔽填充，防止填充值影响有效路径
    def __init__(self, width, dropout):  # 接收特征宽度与丢弃比例
        super().__init__()  # 初始化父类
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)  # 建立两层路径 token 归一化
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)  # 建立第一层一维卷积
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)  # 建立第二层一维卷积
        self.activation, self.dropout = nn.GELU(), nn.Dropout(dropout)  # 建立激活与随机丢弃层

    def forward(self, x, valid):  # 定义物理卷积块前向计算
        m = valid.unsqueeze(-1)  # 扩展有效路径掩码的维度
        h = self.norm1(x).masked_fill(~m, 0)  # 归一化并清零无效路径
        h = self.conv1(h.transpose(1, 2)).transpose(1, 2)  # 调整轴顺序以卷积路径序列
        h = self.dropout(self.activation(h)).masked_fill(~m, 0)  # 激活、丢弃并再次清零无效路径
        h = self.norm2(h).masked_fill(~m, 0)  # 再次归一化和清零
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)  # 第二次一维卷积并恢复轴顺序
        return (x + h).masked_fill(~m, 0)  # 残差相加后清零无效路径

class MaskedPool(nn.Module):  # 定义支持路径掩码的池化层
    def __init__(self, width, mode="attention"):  # 接收宽度和池化模式
        super().__init__()  # 初始化父类
        self.mode = mode  # 保存池化模式
        self.score = nn.Linear(width, 1) if mode == "attention" else None  # 注意力模式建立可学习打分层

    def forward(self, x, valid):  # 定义池化前向计算
        x = x.masked_fill(~valid.unsqueeze(-1), 0)  # 清零无效 token
        if self.score is None:  # 判断是否使用均值池化
            weights = valid.to(x.dtype)  # 均值池化给有效 token 权重一
        else:  # 注意力池化进入打分分支
            logits = self.score(x).squeeze(-1).masked_fill(~valid, -1e4)  # 计算分数并屏蔽无效 token
            weights = torch.softmax(logits, dim=1) * valid.to(x.dtype)  # 归一化注意力分数并清零无效权重
        # All-missing physical branches return exactly zero, with no NaNs.
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)  # 再次归一化并防止全空分支除零
        return (x * weights.unsqueeze(-1)).sum(1)  # 加权求和得到流摘要


class StatisticalFeatureWeighting(nn.Module):  # 定义统计输入逐通道加权层
    """Multiply each statistical input channel once, before its first convolution."""  # 在第一层统计卷积之前，对每个原始统计通道施加一次倍率
    def __init__(self, channels, mode="none", fixed_weights=None, weight_source=None):  # 接收通道数、模式、权重和来源
        super().__init__()  # 初始化父类
        if mode not in ("none", "learned", "fixed"):  # 限定可用的加权模式
            raise ValueError("Unknown statistical feature weighting mode")  # 非法模式抛出异常
        self.channels, self.mode = channels, mode  # 保存通道数和模式
        self.register_buffer("unit_weights", torch.ones(channels), persistent=False)  # 注册全一参考权重
        self.register_parameter("logits", nn.Parameter(torch.zeros(channels)) if mode=="learned" else None)  # 学习模式创建可优化权重参数
        if mode=="fixed":  # 判断是否使用固定外部权重
            if fixed_weights is None or len(fixed_weights) != channels:  # 核对固定权重数量
                raise ValueError("Fixed weights must match the original eight selected features")  # 权重缺失或数量错误则报错
            weights = torch.tensor(fixed_weights, dtype=torch.float32)  # 把固定权重转换为浮点张量
            if weights.shape != (channels,) or not torch.isfinite(weights).all() or not torch.all(weights>0):  # 检查权重形状、有限性和正值
                raise ValueError("Invalid fixed statistical feature weights")  # 非法权重则报错
        else:  # 进入非固定模式
            weights = None  # 非固定模式无需固定权重张量
        # Buffer: saved in checkpoints and moved with the model; not optimized.
        self.register_buffer("fixed_weights", weights)  # 注册可随模型保存的固定权重
        self.fixed_weight_source = weight_source if mode=="fixed" else None  # 固定模式保存权重来源

    def get_weights(self):  # 定义获取当前权重的方法
        if self.mode=="fixed":  # 判断是否是固定模式
            return self.fixed_weights  # 返回固定权重
        if self.mode=="learned":  # 判断是否是学习模式
            return 1.0 + 0.5 * torch.tanh(self.logits)  # 将学习倍率约束在零点五到一点五
        return self.unit_weights  # 关闭加权时返回全一权重

    def forward(self, x):  # 定义输入特征加权前向计算
        if x.ndim!=4 or x.shape[1]!=self.channels:  # 核查输入维数与通道数
            raise ValueError("Unexpected statistical channels for feature weighting")  # 输入形状错误则报错
        if self.mode=="none":  # 判断是否关闭加权
            return x  # 关闭时原样返回
        return x * self.get_weights().view(1,self.channels,1,1)  # 广播各通道倍率并乘到输入


class SDFNet(nn.Module):  # 定义完整双流 SDF-Net 模型
    def __init__(self, physical_dim=5, max_paths=15, cnn_width=32, cnn_blocks=2,  # 配置物理维数、路径数和 CNN
                 d_model=96, num_heads=4, num_layers=2, ffn_ratio=2,  # 配置 Transformer 宽度、头数、层数
                 dropout=0.1, fusion="gated", pooling="attention",  # 配置丢弃比例、融合和池化
                 branch="both", shortcut=True, coordinate_attention=True, stat_weighting="none",  # 配置分支消融、捷径、CA 与特征加权
                 stat_feature_weights=None, stat_weight_source=None):  # 配置可选固定权重和来源
        super().__init__()  # 初始化父类
        if min(physical_dim, max_paths, cnn_width, d_model, num_heads, ffn_ratio) < 1:  # 检查主要维度是否为正
            raise ValueError("Model dimensions must be positive")  # 维度错误则报错
        if max_paths > 15 or min(cnn_blocks, num_layers) < 0 or d_model % num_heads:  # 检查路径数、层数和头数整除关系
            raise ValueError("Invalid path count, block count or attention dimensions")  # 配置错误则报错
        if not 0 <= dropout < 1 or fusion not in ("gated", "concat"):  # 检查丢弃比例与融合方式
            raise ValueError("Invalid dropout or fusion")  # 配置错误则报错
        if pooling not in ("attention", "mean") or branch not in ("both", "stat", "physical"):  # 检查池化方式与分支选项
            raise ValueError("Invalid pooling or branch")  # 配置错误则报错
        if not isinstance(coordinate_attention, bool):  # 检查 CA 开关类型
            raise ValueError("coordinate_attention must be boolean")  # 类型错误则报错
        self.physical_dim, self.max_paths = physical_dim, max_paths  # 保存物理维数和最大路径数
        self.fusion, self.branch, self.shortcut = fusion, branch, shortcut  # 保存融合、分支和捷径配置
        self.coordinate_attention_enabled = coordinate_attention  # 保存 CA 开关状态
        self.stat_feature_weighting = StatisticalFeatureWeighting(  # 构建统计输入加权层
            7, stat_weighting, fixed_weights=stat_feature_weights, weight_source=stat_weight_source)  # 传入八个通道和固定权重配置
        self.stat_stem = nn.Conv2d(7, cnn_width, 1)  # 用一点一卷积投影统计输入 5 # 统计流：二维网格
        self.stat_blocks = nn.Sequential(*[  # 开始构建统计流卷积块列表
            StatisticalBlock(cnn_width, dropout) for _ in range(cnn_blocks)])  # 根据配置堆叠统计 CNN 残差块
        self.coordinate_attention = CoordinateAttention(cnn_width) if coordinate_attention else nn.Identity()  # 按开关构建 CA 或恒等映射
        self.stat_projection = nn.Linear(cnn_width, d_model)  # 投影统计 token 到模型宽度
        self.phys_stem = nn.Linear(physical_dim, cnn_width)  # 将物理路径特征映射到卷积宽度
        self.phys_blocks = nn.ModuleList([  # 开始构建物理流卷积块列表 5 # 物理流：二维网格
            PhysicalBlock(cnn_width, dropout) for _ in range(cnn_blocks)])  # 根据配置堆叠物理 CNN 残差块
        self.phys_projection = nn.Linear(cnn_width, d_model)  # 投影物理 token 到模型宽度
        self.stat_pool, self.phys_pool = MaskedPool(d_model, pooling), MaskedPool(d_model, pooling)  # 分别建立两流的早期池化层
        self.gate = (nn.Sequential(nn.Linear(2 * d_model + 1, d_model), nn.GELU(),  # 门控融合先读取两流摘要和路径比例
                                   nn.Linear(d_model, 2)) if fusion == "gated" else None)  # 输出两路门控分数；拼接模式不建门控层
        self.stat_position = nn.Parameter(torch.empty(1, 16, d_model))  # 建立十六个统计位置编码
        self.path_position = nn.Parameter(torch.empty(1, max_paths, d_model))  # 建立物理路径位置编码
        self.modality = nn.Parameter(torch.empty(2, d_model))  # 建立两种输入模态编码
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))  # 建立 CLS 汇总 token
        self.layers = nn.ModuleList([  # 开始构建 Transformer 层列表
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * ffn_ratio,  # 设置注意力头数与前馈宽度
                                       dropout=dropout, activation="gelu",  # 设置丢弃比例和激活函数
                                       batch_first=True, norm_first=True)  # 设置批量优先与前置归一化
            for _ in range(num_layers)])  # 按指定层数堆叠编码层
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

    def set_target_transform(self, center, scale):  # 定义目标坐标变换设置方法
        with torch.no_grad():  # 设置缓冲区时关闭梯度记录
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

    def forward(self, statistical, physical, path_mask, return_aux=False):  # 定义完整双流前向计算
        b = statistical.shape[0]  # 读取批大小
        if statistical.shape != (b, 7, 4, 4):  # 核查统计输入形状
            raise ValueError("Expected statistical [B,7,4,4]")  # 统计输入错误则报错
        if physical.shape != (b, self.max_paths, self.physical_dim):  # 核查物理输入形状 [B,15,3or5]
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
        physical = physical.masked_fill(~valid.unsqueeze(-1), 0)  # 清零填充路径避免参与卷积
        statistical = self.stat_feature_weighting(statistical)  # 对八个统计输入通道加权
        stat_stem_out = self.stat_stem(statistical)         # 统计分支特征提取
        stat_block_out = self.stat_blocks(stat_stem_out)         # 统计分支特征提取
        phys_stem_ca = self.phys_stem(physical)         # 物理分支特征提取
        s = self.coordinate_attention(stat_block_out).flatten(2).transpose(1, 2)
        s = self.stat_projection(s).masked_fill(~stat_valid.unsqueeze(-1), 0)  # 投影统计 token 并屏蔽无效位置
        p = self.phys_stem(physical).masked_fill(~valid.unsqueeze(-1), 0)  # 物理输入先经线性层并屏蔽填充路径
        for block in self.phys_blocks:  # 遍历物理流残差卷积块
            p = block(p, valid)  # 更新有效路径特征
        p = self.phys_projection(p).masked_fill(~valid.unsqueeze(-1), 0)  # 投影物理 token 并清零无效位置
        sg, pg = self.stat_pool(s, stat_valid), self.phys_pool(p, valid)  # 分别汇聚两路早期摘要 #7 门控融合 两路特征分别池化成。门控网络根据两个摘要和有效路径比例，计算当前样本的两路权重。
        availability = torch.stack((stat_valid.any(1), valid.any(1)), dim=1)  # 判断两路输入是否有效
        if self.gate is None:  # 非门控模式使用固定融合分支
            gate = availability.to(s.dtype)  # 把有效标记转为数值权重 #7 门控融合
        else:  # 门控模式使用学习融合分支
            count = valid.to(s.dtype).mean(1, keepdim=True)  # 计算有效路径占比
            logits = self.gate(torch.cat((sg, pg, count), dim=1))  # 由两路摘要和路径占比计算门控分数
            gate = torch.softmax(logits.masked_fill(~availability, -1e4), dim=1)  # 屏蔽缺失分支并做 softmax
            gate = gate * availability.to(gate.dtype)  # 强制无效分支权重为零
            gate = gate / gate.sum(1, keepdim=True).clamp_min(1e-8)  # 重新归一化两路权重
        s = (s + self.stat_position + self.modality[0]) * gate[:, 0, None, None]  # 7给统计 token 加位置模态编码并乘门控
        p = (p + self.path_position + self.modality[1]) * gate[:, 1, None, None]  # 7给物理 token 加位置模态编码并乘门控
        s = s.masked_fill(~stat_valid.unsqueeze(-1), 0)  # 再次清零无效统计 token
        p = p.masked_fill(~valid.unsqueeze(-1), 0)  # 再次清零无效物理 token
        tokens = torch.cat((self.cls.expand(b, -1, -1), s, p), dim=1)  # 合并 CLS、统计和物理 token
        token_valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=s.device),  # 建立始终有效的 CLS 掩码
                                 stat_valid, valid), dim=1)  # 拼接统计与物理 token 掩码
        for layer in self.layers:  # 8 逐层运行 Transformer 编码器
            tokens = layer(tokens, src_key_padding_mask=~token_valid)  # 在编码层内运行 MHSA 与前馈层并屏蔽填充路径
            tokens = tokens.masked_fill(~token_valid.unsqueeze(-1), 0)  # 每层后清零无效 token
        tokens = self.output_norm(tokens)  # 归一化融合后的 token  1个CLS token + 16个统计token + 15个物理token                  = 32个token
        summary = torch.cat((tokens[:, 0], self.final_stat_pool(tokens[:, 1:17], stat_valid),  # 拼接 CLS 摘要与统计流最终池化
                             self.final_phys_pool(tokens[:, 17:], valid)), dim=1)  # 再拼接物理流最终池化
        prediction = self.head(summary)  #  8 经定位头预测归一化三维坐标
        if self.local_head is not None:  # 8 判断是否启用早期预测捷径
            prediction = prediction + self.local_head(torch.cat((sg, pg), dim=1))  # 将捷径预测加到主预测
        decode = lambda x: x * self.target_scale + self.target_center  # 定义坐标反归一化计算
        prediction = decode(prediction)  # 还原原单位的三维坐标
        if not return_aux:  # 判断是否仅需最终预测
            return prediction  # 只返回三维坐标
        return dict(prediction=prediction, stat_prediction=decode(self.stat_aux(sg)),  # 返回主预测与统计流辅助预测
                    physical_prediction=decode(self.phys_aux(pg)), gate=gate,  # 追加物理流辅助预测和门控权重
                    stat_available=stat_valid.any(1), physical_available=valid.any(1))  # 追加双流有效性标记
