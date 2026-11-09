# 中文阅读路线：这是可选的统一启动入口，本文件本身不实现神经网络训练。
# python train.py --device cuda --seeds 42
#   → 解析启动参数 → 检查数据与可选权重配置 → 拼接 run.py 命令
#   → 为每个种子启动独立 Python 进程 → run.py 读取数据并调用 model.py。
# 也可直接运行 run.py；两种入口的默认超参数存在差别，应以实际命令为准。
# --dry-run 只检查命令和输入配置，不创建训练输出，也不执行反向传播。

"""Unified SDF-Net training entry: python experiments/sdfnet_full/train.py --device cuda --seeds 42."""  # 统一训练入口示例，--seeds可传入一个或多个独立训练种子
import argparse  # 解析命令行参数及生成帮助信息
import importlib.util  # 按文件路径加载run.py以复用校验逻辑
import json  # 读取/保存配置和指标JSON
import os  # 读取并为训练子进程传递环境变量
import shutil  # 复制源码、启动器及预处理配置快照
import subprocess  # 为不同种子启动独立的训练进程
import sys  # 访问当前Python解释器、模块搜索路径和命令行环境
from datetime import datetime  # 生成运行目录使用的时间戳
from pathlib import Path  # 使用Path管理数据、源码和输出路径

HERE=Path(__file__).resolve().parent  # 当前启动脚本所在目录，用于寻找同目录的run.py
PROJECT=next((p for p in HERE.parents if (p/'common/data.py').is_file()),HERE.parents[1])  # 向上寻找项目标记；找不到时使用备用目录，数据也可用--data显式指定


def build_command(args, seed, output):  # 按启动参数和当前种子构造调用run.py的完整命令列表
    return [sys.executable,'-B',str(HERE/'run.py'),  # 使用当前Python环境启动run.py，-B禁止产生pyc缓存
            '--mode','train','--skip-test','--device',args.device,'--seed',str(seed),  # 明确运行训练模式、跳过自动测试，并传递设备与当前种子
            '--data',str(args.data.resolve()),'--output',str(output.resolve()),  # 将数据和本次输出目录转为绝对路径传给子进程
            '--epochs',str(args.epochs),'--patience',str(args.patience),  # 转发最大训练轮数与早停耐心
            '--batch-size',str(args.batch_size),'--lr',str(args.lr),'--weight-decay',str(args.weight_decay),  # 转发批大小、学习率与权重衰减，覆盖run.py自身默认值
            '--dropout',str(args.dropout),'--ema-decay',str(args.ema_decay),'--feature-noise',str(args.feature_noise),  # 转发dropout、EMA系数与仅训练使用的输入噪声系数
            '--aux-weight',str(args.aux_weight),'--distance-weight',str(args.distance_weight),  # 转发双流辅助损失及3D距离损失系数
            '--selection-metric',args.selection_metric,  # 明确最佳模型采用的验证集选择指标
            '--stat-weighting',args.stat_weighting,  # 转发统计输入的固定/关闭/可学习倍率模式
            '--stat-weight-file',str(args.stat_weight_file.resolve())]  # 转发固定倍率配置路径；非fixed模式下run.py不会读取该文件


def parse_args(argv=None):  # 声明统一训练入口的参数并做种子及轮数的初步检查
    p=argparse.ArgumentParser(description=__doc__)  # 创建启动器命令行解析器
    p.add_argument('--device',default='auto')  # 命令行参数 --device：训练/测试设备，auto优先CUDA，也可显式指定cuda或cpu；此入口默认 'auto'
    p.add_argument('--seeds',nargs='+',type=int,default=[42])  # 命令行参数 --seeds：依次从头训练的种子列表，每个种子使用独立输出目录；此入口默认 [42]
    p.add_argument('--epochs',type=int,default=200)  # 命令行参数 --epochs：最大训练轮数，早停可能提前结束；此入口默认 200
    p.add_argument('--patience',type=int,default=40)  # 命令行参数 --patience：连续多少轮验证指标未改善后早停；0关闭早停；此入口默认 40
    p.add_argument('--batch-size',type=int,default=64)  # 命令行参数 --batch-size：每批包含的阵列位置数，不是天线数或路径数；此入口默认 64
    p.add_argument('--lr',type=float,default=0.00096194)  # 命令行参数 --lr：优化器初始学习率，会受调度器影响；此入口默认 0.00096194
    p.add_argument('--weight-decay',type=float,default=0.00017546)  # 命令行参数 --weight-decay：优化器的权重衰减系数；此入口默认 0.00017546
    p.add_argument('--dropout',type=float,default=0.1)  # 命令行参数 --dropout：训练时随机丢弃概率；评估模式关闭丢弃；此入口默认 0.1
    p.add_argument('--ema-decay',type=float,default=0)  # 命令行参数 --ema-decay：EMA参数滑动平均衰减系数，0表示完全禁用EMA；此入口默认 0
    p.add_argument('--feature-noise',type=float,default=0.05)  # 命令行参数 --feature-noise：训练时对标准化统计特征和有效路径时延加入的高斯噪声标准差；0关闭；此入口默认 0.05
    p.add_argument('--aux-weight',type=float,default=0.1)  # 命令行参数 --aux-weight：两路辅助坐标预测损失的系数，0关闭辅助监督；此入口默认 0.1
    p.add_argument('--distance-weight',type=float,default=0.1)  # 命令行参数 --distance-weight：主预测额外3D欧氏距离损失的系数，0关闭；此入口默认 0.1
    p.add_argument('--selection-metric',choices=['coordinate_mse','mean_error_3d'],default='coordinate_mse')  # 命令行参数 --selection-metric：验证集最佳模型选择指标，越小越好；不使用测试误差选轮；此入口默认 'coordinate_mse'
    p.add_argument('--data',type=Path,default=PROJECT/'data/prepared.npz')  # 命令行参数 --data：输入prepared.npz路径，沿用其中固定的数据划分；此入口默认 PROJECT/'data/prepared.npz'
    p.add_argument('--run-root',type=Path,default=HERE/'runs')  # 命令行参数 --run-root：统一启动器为每个种子创建新运行目录的父目录；此入口默认 HERE/'runs'
    p.add_argument('--stat-weighting',choices=['fixed','none','learned'],default='fixed')  # 命令行参数 --stat-weighting：none不加权；learned从1学习倍率；fixed从JSON加载固定倍率；此入口默认 'fixed'
    p.add_argument('--stat-weight-file',type=Path,default=HERE/'feature_weights/paper_selected8.json')  # 命令行参数 --stat-weight-file：fixed模式的8个统计输入倍率JSON文件路径；此入口默认 HERE/'feature_weights/paper_selected8.json'
    p.add_argument('--dry-run',action='store_true',help='Validate and show plan; no training or output creation')  # 命令行参数 --dry-run：只校验参数与输入配置并显示计划，不启动训练
    a=p.parse_args(argv)  # 获得用户参数；未提供的选项使用此启动器自己的默认值
    if a.epochs<1 or a.patience<0:  # 最大轮数至少为1，早停耐心可以为0但不能为负
        p.error('epochs must be positive; patience must be nonnegative')  # 不合法轮数或耐心时输出命令行错误
    if len(set(a.seeds))!=len(a.seeds) or any(s<0 or s>=2**32 for s in a.seeds):  # 种子列表不能重复，且种子必须处于NumPy支持的无符号32位范围
        p.error('seeds must be unique integers in [0, 2**32)')  # 拒绝重复或越界种子
    return a  # 返回启动参数对象


def main(argv=None):  # 统一启动器主流程：先校验，再按种子依次训练
    a=parse_args(argv)  # 解析启动参数，不直接创建神经网络
    spec=importlib.util.spec_from_file_location('sdfnet_training_runner',HERE/'run.py')  # 按文件路径定位run.py模块，以便复用它的参数校验与倍率读取函数
    runner=importlib.util.module_from_spec(spec)  # 根据模块规格创建模块对象
    spec.loader.exec_module(runner)  # 加载run.py定义；因为不是作为主脚本执行，此处不会触发训练
    if not a.data.is_file():  # 启动训练前确认指定NPZ真实存在
        raise FileNotFoundError(a.data)  # 数据文件缺失时立即停止
    if a.stat_weighting=='fixed':  # 固定输入倍率模式还需要预先检查外部JSON
        runner.load_stat_weight_file(a.stat_weight_file)  # 校验8个特征名称、顺序、倍率合法性及配置来源
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # 为这一批种子生成时间戳，避免混入旧运行目录
    for seed in a.seeds:  # 按用户传入的种子顺序分别从头训练
        output=a.run_root/('train_{}_seed_{}'.format(stamp,seed))  # 每个种子使用独立的运行输出目录
        command=build_command(a,seed,output)  # 构造该种子实际执行的run.py命令
        # Validate every forwarded option using the actual runner before training.
        runner.parse_args(command[3:])  # 跳过Python/-B/脚本路径，把所有转发选项交给run.py再次完整校验
        print('SDF-Net | seed={} | device={} | input weighting={} | epochs={} | patience={}'.format(  # 构造本次训练的概要日志
            seed,a.device,a.stat_weighting,a.epochs,a.patience),flush=True)  # 打印种子、设备、输入加权、最大轮数及耐心
        print('Output: '+str(output),flush=True)  # 显示本次运行结果的目标目录
        if a.dry_run:  # 仅检查模式下不启动训练、不创建运行输出
            continue  # 跳过此种子的执行，继续验证下一个种子计划
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')  # 继承当前环境，并禁止写pyc、要求日志即时输出
        subprocess.run(command,check=True,cwd=str(HERE),env=env)  # 同步启动并等待run.py完成；失败会抛出异常并停止后续种子
        shutil.copy2(HERE/'train.py',output/'train.py')  # 训练成功后在输出目录保存此次使用的启动器快照
        (output/'training_launcher.json').write_text(json.dumps(dict(  # 保存本次启动器路径、种子与实际命令，便于追溯参数
            entry=str(HERE/'train.py'),seed=seed,command=command,  # 记录真实入口、当前种子及包括覆盖参数的完整命令列表
            interpretation='Train network from scratch; fixed input multipliers loaded from named JSON'),  # 原有说明字符串描述fixed配置；实际是否使用倍率须同时看--stat-weighting参数
            ensure_ascii=False,indent=2),encoding='utf-8')  # 把启动记录以保留中文的UTF-8 JSON格式写出
    if a.dry_run:  # 仅检查模式下不启动训练、不创建运行输出
        print('Dry run passed. No network training or test evaluation performed.',flush=True)  # 全部种子计划校验成功后打印提示，本次没有训练或测试


if __name__=='__main__':  # 仅直接执行train.py时启动流程，导入时不自动训练
    main()  # 进入统一训练启动器
