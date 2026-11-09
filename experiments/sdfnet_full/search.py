import argparse  # 解析广范围搜索的命令行参数。
import ast  # 核查训练入口实际使用的特征列表。
import csv  # 保存便于查看的排名表。
import hashlib  # 锁定数据与源码版本。
import json  # 保存搜索计划和运行结果。
import math  # 检查指标和生成对数均匀采样。
import os  # 为子进程设置中文输出编码。
import random  # 用固定种子生成可重复的候选参数。
import shutil  # 保存源码快照。
import statistics  # 汇总不同训练种子的验证误差。
import subprocess  # 顺序调用现有训练入口。
import sys  # 使用当前激活环境的解释器。
from datetime import datetime  # 生成互不覆盖的目录名称。
from pathlib import Path  # 管理数据及实验目录。
import numpy as np  # 只检查固定划分索引，不读取测试特征或标签。

HERE = Path(__file__).resolve().parent  # 最终应放在sdfnet_full目录。
SPACE = {  # 搜索范围；保留双流、CA和MHSA，主要覆盖容量与过拟合相关参数。
    'batch_size': [16, 32, 64, 128],  # 每批阵列样本数。
    'cnn_width': [16, 24, 32, 48],  # 两路CNN的通道数。
    'cnn_blocks': [1, 2],  # 每路CNN残差块数。
    'd_model': [48, 64, 96, 128],  # Transformer嵌入维数。
    'num_heads': [4, 8],  # 注意力头数，所有候选维数均可整除。
    'num_layers': [1, 2, 3],  # Transformer层数，始终保留MHSA。
    'ffn_ratio': [2, 4],  # 前馈网络隐藏维数倍率。
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],  # 丢弃率。
    'ema_decay': [0.0, 0.98, 0.995],  # EMA关闭或两种平滑系数。
    'aux_weight': [0.0, 0.05, 0.1, 0.2],  # 双流辅助损失权重。
    'distance_weight': [0.0, 0.1, 0.2],  # 三维欧氏距离损失权重。
    'feature_noise': [0.0, 0.01, 0.03],  # 仅训练时施加的特征噪声。
}  # 学习率与权重衰减单独在对数尺度采样。
BASE = dict(lr=0.001, weight_decay=0.0001, batch_size=64, cnn_width=32, cnn_blocks=2, d_model=96, num_heads=4, num_layers=2, ffn_ratio=2, dropout=0.1, ema_decay=0.98, aux_weight=0.1, distance_weight=0.1, feature_noise=0.0)  # 纳入当前默认配置作为参照。
FIXED = dict(max_paths=15, angle_encoding='circular', fusion='gated', pooling='attention', branch='both', stat_weighting='none', optimizer='adamw', scheduler='cosine', warmup_epochs=10, min_lr_ratio=0.05, grad_clip=1.0, selection_metric='mean_error_3d')  # 不带入旧数据学得的固定输入权重。

def digest(path):  # 分块计算文件指纹。
    h = hashlib.sha256()  # 创建哈希对象。
    with Path(path).open('rb') as stream:  # 只读打开文件。
        for block in iter(lambda: stream.read(1024 * 1024), b''):  # 避免一次读取大文件。
            h.update(block)  # 更新指纹。
    return h.hexdigest()  # 返回可写入计划的文本。

def save(path, value):  # 通过临时文件替换，降低中断造成JSON损坏的风险。
    path = Path(path)  # 统一路径类型。
    temporary = path.with_name(path.name + '.tmp')  # 临时文件与目标在同一目录。
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')  # 拒绝保存非有限指标。
    temporary.replace(path)  # 替换上次汇总。

def load(path):  # 读取已有计划或单次结果。
    return json.loads(Path(path).read_text(encoding='utf-8'))  # 不执行文件中的任何代码。

def make_candidates(count, seed):  # 固定采样种子，不根据测试结果调整候选。
    rng = random.Random(seed)  # 独立于训练种子的搜索随机源。
    result = [dict(BASE)]  # 第一组始终为当前模型默认参数。
    result.append(dict(BASE, cnn_width=16, cnn_blocks=1, d_model=64, num_layers=1, dropout=0.2, batch_size=32))  # 显式覆盖小样本下的较小网络。
    result.append(dict(BASE, cnn_width=24, cnn_blocks=1, d_model=48, num_layers=1, dropout=0.3, lr=0.0003))  # 再加入较强正则化的小网络。
    seen = {json.dumps(c, sort_keys=True) for c in result}  # 避免重复配置。
    while len(result) < count:  # 生成指定数量的不重复候选。
        c = {name: rng.choice(values) for name, values in SPACE.items()}  # 独立采样离散参数。
        c['lr'] = float('{:.8g}'.format(math.exp(rng.uniform(math.log(5e-5), math.log(3e-3)))))  # 学习率范围5e-5到3e-3。
        c['weight_decay'] = 0.0 if rng.random() < 0.15 else float('{:.8g}'.format(math.exp(rng.uniform(math.log(1e-6), math.log(1e-2)))))  # 包含零衰减及1e-6到1e-2。
        key = json.dumps(c, sort_keys=True)  # 构造稳定去重键。
        if key not in seen:  # 仅接受新配置。
            seen.add(key)  # 登记已采样配置。
            result.append(c)  # 加入固定执行计划。
    return result[:count]  # 小预算时仍可运行前一至三组。

def inspect_data(data, runner):  # 在搜索开始前核查数据索引和实际输入特征。
    with np.load(data, allow_pickle=False) as archive:  # 只解压数组ID和三个索引数组。
        n = len(archive['array_id'])  # 得到总阵列样本数。
        indices = {name: archive[name + '_indices'] for name in ('train', 'val', 'test')}  # 测试索引仅用于检查互斥性。
        for name, values in indices.items():  # 检查每个划分合法。
            if values.ndim != 1 or values.dtype.kind not in 'iu' or len(values) == 0 or np.any(values < 0) or np.any(values >= n):  # 拒绝空集、越界及非整数索引。
                raise ValueError('无效划分索引：' + name)  # 停止错误数据上的搜索。
        joined = np.concatenate(list(indices.values()))  # 合并索引用于完整性核查。
        if len(joined) != n or len(np.unique(joined)) != n:  # 同时要求无交叉、无遗漏。
            raise ValueError('训练、验证、测试索引必须互斥且覆盖全部阵列')  # 防止错误划分。
        if len(np.unique(archive['array_id'])) != n:  # 阵列ID必须唯一。
            raise ValueError('array_id存在重复')  # 不能用重复阵列作为独立样本。
        counts = {name: len(v) for name, v in indices.items()}  # 记录划分数量。
    tree = ast.parse(runner.read_text(encoding='utf-8-sig'))  # 核查源码而不导入训练模块。
    columns = None  # 等待找到统计特征定义。
    for node in tree.body:  # 仅检查顶层定义。
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STAT_COLUMNS' for t in node.targets):  # 读取当前SDF训练入口的列定义。
            columns = ast.literal_eval(node.value)  # 仅接受静态列表。
    expected = {'tau_dom_RMS_ns', 'K_factor', 'PL_dB', 'Rx_power_dBm', 'Tau_RMS_ns', 'DPR', 'R_max_dB'}  # 当前新数据选出的七项统计特征。
    if columns is None or len(columns) != 7 or set(columns) != expected:  # 防止搜索时误用旧输入集合。
        raise ValueError('run.py的STAT_COLUMNS与新七项特征不一致')  # 指出需要同步输入配置。
    return dict(arrays=n, counts=counts, statistical_features=columns)  # 保存输入审计结果。

def command_for(plan, candidate, seed, output):  # 为一次训练生成独立命令。
    command = [plan['python'], '-B', '-u', plan['runner'], '--mode', 'train', '--skip-test', '--data', plan['data'], '--output', str(output), '--device', plan['device'], '--epochs', str(plan['epochs']), '--patience', str(plan['patience']), '--seed', str(seed)]  # 明确训练阶段不执行测试。
    for name, value in dict(FIXED, **candidate).items():  # 所有固定项和候选项均显式传递。
        command.extend(['--' + name.replace('_', '-'), str(value)])  # 转换为训练入口参数。
    return command  # 便于保存和复核。

def run_trial(plan, number, seed, root, retry_failed):  # 执行或恢复一个配置的一次训练。
    for name in ('run.py', 'model.py'):  # 长时间搜索期间也禁止训练源码被悄悄修改。
        if digest(HERE / name) != plan['hashes'][name]:  # 开始每次训练前确认仍为计划中的版本。
            raise RuntimeError('搜索期间源码发生变化，请恢复源码或新建搜索：' + name)  # 防止不同实现混入同一排名。
    trial = root / 'trials' / ('config_{:03d}'.format(number)) / ('seed_' + str(seed))  # 配置与种子分别建目录。
    completed = trial / 'completed.json'  # 只有成功检查指标后才写完成标记。
    if completed.is_file():  # 中断恢复时跳过已完成训练。
        record = load(completed)  # 读取已保存结果。
        if not Path(record['checkpoint']).is_file():  # 防止结果存在但模型文件被删除。
            raise FileNotFoundError(record['checkpoint'])  # 明确提示缺失文件。
        print('跳过已完成：配置{}，seed={}'.format(number, seed), flush=True)  # 显示恢复进度。
        return record  # 直接复用该种子的验证指标。
    if (trial / 'failed.json').exists() and not retry_failed:  # 失败默认停止，避免静默丢弃难训练配置。
        raise RuntimeError('存在失败记录，请检查日志；确认重试后使用 --retry-failed：' + str(trial))  # 保留失败原因供排查。
    attempt = trial / ('attempt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))  # 重试从头训练，保留上次日志。
    attempt.mkdir(parents=True)  # 不覆盖任何历史训练目录。
    command = command_for(plan, plan['candidates'][number - 1], seed, attempt / 'training')  # 固定候选与数据配置。
    save(attempt / 'command.json', command)  # 保存实际启动命令。
    print('配置{}/{}，seed={}，参数={}'.format(number, len(plan['candidates']), seed, plan['candidates'][number - 1]), flush=True)  # 每次训练前显示参数。
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')  # 保证中文日志编码一致。
    try:  # 训练和结果核验失败均保存错误记录。
        with (attempt / 'console.log').open('w', encoding='utf-8') as log:  # 控制台与文件同时保存逐轮进度。
            process = subprocess.Popen(command, cwd=str(HERE), env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')  # 顺序占用单张GPU。
            try:  # 用户中断时也停止训练子进程。
                for line in process.stdout:  # 实时转发训练输出。
                    print(line, end='', flush=True)  # 避免长时间看不到训练状态。
                    log.write(line)  # 写入当前尝试日志。
                    log.flush()  # 即使中断也保留最近输出。
                code = process.wait()  # 等待训练结束。
            except BaseException:  # 包括Ctrl+C。
                process.terminate()  # 停止后台训练。
                process.wait()  # 等待资源释放。
                raise  # 保留原始异常。
            finally:  # 总是关闭输出管道。
                process.stdout.close()  # 释放管道句柄。
        if code != 0:  # 任何训练错误都不能当作完成。
            raise RuntimeError('训练失败，返回码{}；日志：{}'.format(code, attempt / 'console.log'))  # 引导检查具体日志。
        training = attempt / 'training'  # 训练目录由现有run.py创建。
        report = load(training / 'selection_metrics.json')  # 仅读取验证选择报告。
        score = float(report['validation']['mean_error_3d'])  # 主指标与训练checkpoint选择指标一致。
        if not math.isfinite(score) or score < 0 or report['validation']['n'] != plan['audit']['counts']['val']:  # 拒绝异常或样本数不匹配的指标。
            raise ValueError('验证指标无效或验证样本数不一致')  # 不将错误结果纳入排名。
        if report['data_sha256'] != plan['data_sha256'] or report['selection_metric'] != 'mean_error_3d':  # 验证数据版本及选模准则。
            raise ValueError('训练报告的数据版本或选模指标不一致')  # 防止运行时配置偏离计划。
        checkpoint = training / 'best.pt'  # 记录对应的实际模型权重。
        if not checkpoint.is_file():  # 确保最终可用于正式评估。
            raise FileNotFoundError(checkpoint)  # 不导出无模型的成功记录。
        record = dict(number=number, seed=seed, val_mean_error_3d=score, best_epoch=report['best_epoch'], checkpoint=str(checkpoint), parameters=report['parameters'])  # 保存最少必要指标。
        save(completed, record)  # 最后写成功标记以支持可靠恢复。
        return record  # 供汇总排名使用。
    except Exception as error:  # 训练失败需要主动检查，不自动忽略。
        save(trial / 'failed.json', dict(error=str(error), attempt=str(attempt)))  # 记录失败原因与日志目录。
        raise  # 停止当前搜索，修复后可恢复。

def summarize(root, plan, records, selected):  # 分别汇总单种子筛选和多种子复核。
    screen_seed = plan['seeds'][0]  # 第一种子用于所有候选的初筛。
    screening = sorted([r for (number, seed), r in records.items() if seed == screen_seed], key=lambda r: r['val_mean_error_3d'])  # 按验证误差升序。
    save(root / 'screening_ranking.json', screening)  # 初筛结果不能冒充多种子最终排名。
    confirmed = []  # 仅收集完成所有指定种子的候选。
    for number in selected:  # 复核对象由完整初筛结果确定。
        results = [records.get((number, seed)) for seed in plan['seeds']]  # 包括初筛种子，避免重复训练。
        if all(r is not None for r in results):  # 种子不齐时不进入最终排名。
            values = [r['val_mean_error_3d'] for r in results]  # 保留各个种子的验证均值。
            confirmed.append(dict(number=number, val_mean_error_3d=statistics.mean(values), val_seed_std=statistics.pstdev(values), seed_values=values, checkpoints=[r['checkpoint'] for r in results], configuration=plan['candidates'][number - 1]))  # 不挑选表现最好的单一种子。
    confirmed.sort(key=lambda r: r['val_mean_error_3d'])  # 最终按跨种子平均误差排序。
    save(root / 'confirmed_ranking.json', confirmed)  # 保存可复核的完整复核排名。
    for filename, rows in [('screening_ranking.csv', screening), ('confirmed_ranking.csv', confirmed)]:  # 导出两个阶段的表格。
        with (root / filename).open('w', encoding='utf-8-sig', newline='') as stream:  # 方便Excel直接读取中文。
            writer = csv.writer(stream)  # 各行参数作为JSON文本保存。
            writer.writerow(['number', 'val_mean_error_3d', 'val_seed_std', 'configuration'])  # 单位与原始坐标一致，当前数据为米。
            for row in rows:  # 逐个写入已排序配置。
                writer.writerow([row['number'], row['val_mean_error_3d'], row.get('val_seed_std', ''), json.dumps(plan['candidates'][row['number'] - 1], sort_keys=True)])  # 初筛标准差留空。
    if confirmed:  # 至少一个候选完成多种子复核时才导出最佳配置。
        save(root / 'best_config.json', dict(provisional=len(confirmed) != len(selected), selection='mean validation 3D error across specified seeds; best within searched candidates', seeds=plan['seeds'], **confirmed[0]))  # 完整复核前明确标记暂定。
    return screening  # 用于选出待复核的前几名。

def main(argv=None):  # 支持直接命令行运行和检查。
    parser = argparse.ArgumentParser(description='SDF-Net新数据广范围随机搜索：验证集选模、断点恢复、多种子复核。')  # 帮助文本说明执行目标。
    parser.add_argument('--data', '--source', dest='data', type=Path, default=HERE.parents[1] / 'data' / 'prepared.npz')  # 使用实验项目的新数据，允许显式覆盖。
    parser.add_argument('--trials', type=int, default=96)  # 默认初筛96组，而非枚举庞大笛卡尔积。
    parser.add_argument('--top-k', type=int, default=5)  # 默认复核前五组。
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])  # 第一种子初筛，全部种子复核。
    parser.add_argument('--search-seed', type=int, default=20261104)  # 固定候选采样顺序，和数据划分种子独立。
    parser.add_argument('--epochs', type=int, default=250)  # 所有候选共享最大训练轮数。
    parser.add_argument('--patience', type=int, default=50)  # 验证平均误差无改善时提前结束。
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')  # 使用当前环境可用设备。
    parser.add_argument('--output', type=Path)  # 可指定新的搜索结果目录。
    parser.add_argument('--resume', type=Path)  # 指向原搜索目录，复用完成的配置。
    parser.add_argument('--retry-failed', action='store_true')  # 明确允许从头重试失败的训练。
    parser.add_argument('--dry-run', action='store_true')  # 检查输入并显示计划，不训练或创建目录。
    args = parser.parse_args(argv)  # 读取用户参数。
    if min(args.trials, args.top_k, args.epochs, args.patience) < 1 or args.top_k > args.trials:  # 防止不合理的搜索预算。
        parser.error('预算必须为正，top-k不得大于trials')  # 直接显示参数错误。
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):  # 至少两个不重复种子用于复核。
        parser.error('至少提供两个不同且合法的训练种子')  # 禁止把单次结果称为多种子复核。
    if args.output and args.resume:  # 新建和恢复不能混用。
        parser.error('--output与--resume不能同时使用')  # 明确当前执行模式。
    data, runner = args.data.resolve(), HERE / 'run.py'  # 固定绝对路径。
    paths = {'run.py': runner, 'model.py': HERE / 'model.py', 'search.py': Path(__file__).resolve()}  # 锁定直接影响训练和采样的源码。
    for path in [data, data.with_suffix('.json')] + list(paths.values()):  # 配套元数据也是数据版本的一部分。
        if not path.is_file():  # 开始训练前确认文件齐全。
            parser.error('文件不存在：' + str(path))  # 避免批量失败后才发现路径错误。
    audit = inspect_data(data, runner)  # 只核查划分，不生成新划分。
    plan = dict(version=1, data=str(data), data_sha256=digest(data), metadata_sha256=digest(data.with_suffix('.json')), runner=str(runner), python=sys.executable, hashes={k: digest(v) for k, v in paths.items()}, audit=audit, trials=args.trials, top_k=args.top_k, seeds=args.seeds, search_seed=args.search_seed, epochs=args.epochs, patience=args.patience, device=args.device, fixed=FIXED, space=SPACE, candidates=make_candidates(args.trials, args.search_seed))  # 完整保存预先确定的计划。
    print('数据={}；划分={}；初筛{}次，复核新增{}次；总计最多{}次训练。'.format(data, audit['counts'], args.trials, args.top_k * (len(args.seeds) - 1), args.trials + args.top_k * (len(args.seeds) - 1)), flush=True)  # 显示预算，不估计未经测量的耗时。
    if args.dry_run:  # 预览模式完全不启动训练。
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)  # 便于检查各组参数与数据版本。
        return  # 不创建输出目录。
    root = (args.resume or args.output or HERE / 'searches' / ('broad_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()  # 自动生成独立搜索目录。
    if args.resume:  # 恢复时参数和源码必须完全一致。
        if load(root / 'search_plan.json') != plan:  # 不允许旧结果混入新数据或新源码。
            parser.error('恢复参数、数据或源码已变化；请使用原参数恢复，或新建搜索')  # 保护实验可复核性。
    else:  # 新搜索先保存计划与源码快照。
        root.mkdir(parents=True, exist_ok=False)  # 拒绝覆盖旧目录。
        save(root / 'search_plan.json', plan)  # 在首次训练前固定候选。
        snapshot = root / 'source_snapshot'  # 保存可追溯的代码版本。
        snapshot.mkdir()  # 创建源码快照目录。
        for name, path in paths.items():  # 复制所有已锁定源码。
            shutil.copy2(path, snapshot / name)  # 不修改原训练入口。
    print('搜索目录：' + str(root), flush=True)  # 中断恢复需要这个目录。
    if (root / 'best_config.json').is_file():  # 恢复过程中旧汇总先标记为暂定。
        previous_best = load(root / 'best_config.json')  # 保留旧配置与检查点便于核对。
        previous_best['provisional'] = True  # 只有本次完整复核结束才重新标记完成。
        save(root / 'best_config.json', previous_best)  # 避免恢复被中断时误认为已重新完成。
    records = {}  # 运行中保存已完成结果。
    for number in range(1, args.trials + 1):  # 先完整初筛，再确定复核名单。
        record = run_trial(plan, number, args.seeds[0], root, args.retry_failed)  # 统一初筛种子。
        records[(number, args.seeds[0])] = record  # 登记验证指标。
        screening = summarize(root, plan, records, [])  # 每次完成即更新可见排名。
    selected = [r['number'] for r in screening[:args.top_k]]  # 前K名依据验证误差确定。
    save(root / 'shortlist.json', selected)  # 保存复核名单。
    for number in selected:  # 只对初筛前几名追加训练种子。
        for seed in args.seeds[1:]:  # 第一种子的结果直接复用。
            records[(number, seed)] = run_trial(plan, number, seed, root, args.retry_failed)  # 多种子复核也只使用验证集。
            summarize(root, plan, records, selected)  # 动态更新完整种子配置排名。
    summarize(root, plan, records, selected)  # 最终将最佳配置标记为非暂定。
    print('完成。最佳配置：{}；请参数固定后再单独评估测试集。'.format(root / 'best_config.json'), flush=True)  # 搜索阶段不自动测试。

if __name__ == '__main__':  # 直接运行脚本时进入搜索入口。
    main()  # 调用预先定义的流程。
