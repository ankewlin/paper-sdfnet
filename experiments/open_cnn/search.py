import argparse  # 解析命令行中的搜索范围和训练设置。
import csv  # 将各次试验和最终排名保存为表格。
import hashlib  # 记录数据和代码指纹，防止恢复搜索时混用版本。
import itertools  # 枚举学习率和批大小的全部组合。
import json  # 读写搜索计划和训练生成的验证指标。
import math  # 检查学习率及误差是否为有限数值。
import os  # 设置子进程使用的文本编码。
import statistics  # 计算不同种子的平均验证误差及标准差。
import subprocess  # 调用现有训练入口，避免重复实现训练逻辑。
import sys  # 使用当前Conda环境中的Python解释器。
from datetime import datetime  # 为搜索和重试生成独立目录名。
from pathlib import Path  # 统一处理Windows文件路径。

ROOT = Path(__file__).resolve().parent  # 搜索脚本与model.py、run.py放在同一目录。

def read_json(path):  # 读取UTF-8格式的JSON记录。
    return json.loads(path.read_text(encoding="utf-8-sig"))  # 兼容带BOM和不带BOM的文件。

def write_json(path, value):  # 写入可直接阅读的JSON结果。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")  # 禁止写入NaN等无效结果。

def digest(path):  # 计算文件SHA256而不一次性加载整个数据文件。
    result = hashlib.sha256()  # 初始化文件哈希。
    with path.open("rb") as stream:  # 以二进制读取文件。
        for block in iter(lambda: stream.read(1024 * 1024), b""):  # 每次读取一兆字节。
            result.update(block)  # 将当前数据块加入哈希。
    return result.hexdigest()  # 返回可保存到JSON的十六进制指纹。

def export_csv(path, rows, columns):  # 将试验进度或配置排名写入CSV。
    with path.open("w", encoding="utf-8-sig", newline="") as stream:  # 使用BOM方便Excel显示中文。
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")  # 固定列顺序并忽略内部辅助字段。
        writer.writeheader()  # 写入表头。
        writer.writerows(rows)  # 写入当前全部记录。

def refresh(output, plan):  # 根据已完成试验重建汇总，支持中断后恢复。
    trials = []  # 保存已经正常完成的种子试验。
    ranking = []  # 保存全部指定种子都完成的超参数配置。
    for config in plan["configurations"]:  # 逐个检查学习率和批大小组合。
        scores = []  # 保存该配置在各个种子下的验证平均误差。
        checkpoints = []  # 保存该配置对应的各个最佳模型路径。
        for seed in plan["seeds"]:  # 每个配置使用完全相同的种子列表。
            state_path = output / config["id"] / ("seed_" + str(seed)) / "completed.json"  # 仅认可明确完成的试验。
            if not state_path.exists():  # 中断或失败的试验不参与结果排名。
                continue  # 等待后续恢复运行补齐该种子。
            record = read_json(state_path)  # 读取本脚本保存的完成记录。
            report_path = Path(record["run_directory"]) / "validation_metrics.json"  # 找到训练入口产生的验证指标。
            checkpoint = Path(record["run_directory"]) / "best.pt"  # 找到只按验证表现保存的模型。
            report = read_json(report_path)  # 从训练结果重新读取分数。
            score = float(report["mean_error_3d"])  # 排名指标统一为验证集平均三维误差。
            if not math.isfinite(score) or not checkpoint.is_file():  # 防止无效分数或缺失检查点进入排名。
                raise ValueError("完成记录损坏：" + str(state_path))  # 直接报告问题而不是默默使用错误记录。
            row = dict(config_id=config["id"], lr=config["lr"], batch_size=config["batch_size"], seed=seed, val_mean_error_3d=score, best_epoch=report["best_epoch"], checkpoint=str(checkpoint))  # 汇总一次完整试验。
            trials.append(row)  # 加入种子级结果表。
            scores.append(score)  # 收集配置级平均值所需分数。
            checkpoints.append(str(checkpoint))  # 保留每个种子的模型，不选择表现最好的种子。
        if len(scores) == len(plan["seeds"]):  # 只有种子全部完成的配置才可以比较。
            ranking.append(dict(config_id=config["id"], lr=config["lr"], batch_size=config["batch_size"], seeds=plan["seeds"], val_mean_error_3d=statistics.mean(scores), val_seed_std=statistics.pstdev(scores), checkpoints=checkpoints))  # 使用跨种子均值排名，标准差采用总体定义。
    ranking.sort(key=lambda item: (item["val_mean_error_3d"], item["config_id"]))  # 分数相同时采用固定顺序，保证可复现。
    export_csv(output / "trials.csv", trials, ["config_id", "lr", "batch_size", "seed", "val_mean_error_3d", "best_epoch", "checkpoint"])  # 保存所有已完成的种子结果。
    export_csv(output / "ranking.csv", ranking, ["config_id", "lr", "batch_size", "val_mean_error_3d", "val_seed_std"])  # 保存配置级排名。
    if ranking:  # 至少有一个完整配置时提供当前最佳记录。
        write_json(output / "best_config.json", dict(provisional=len(ranking) != len(plan["configurations"]), selection_metric="mean validation 3D error across specified seeds", completed_configurations=len(ranking), total_configurations=len(plan["configurations"]), **ranking[0]))  # 搜索未完成时明确标注为暂定结果。
    return ranking  # 返回当前排名供终端显示。

def main():  # 执行搜索，训练入口和网络结构保持不变。
    if hasattr(sys.stdout, "reconfigure"):  # 兼容Windows控制台的默认编码。
        sys.stdout.reconfigure(encoding="utf-8")  # 统一输出中文和训练日志。
    parser = argparse.ArgumentParser(description="纯CNN训练超参数网格搜索，仅按验证集选择配置。")  # 描述本脚本的选择规则。
    parser.add_argument("--data", "--source", dest="data", type=Path, default=Path("/data/old_data_1340/prepared.npz"))  # 使用与主实验相同的数据。
    parser.add_argument("--metadata", type=Path)  # 可选指定配套JSON，默认与NPZ同名。
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.0003, 0.001, 0.003])  # 默认比较三个学习率，包含作者默认值。
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128])  # 默认比较三个批大小，包含作者默认值。
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])  # 默认固定种子42，也可一次指定42、43、44。
    parser.add_argument("--epochs", type=int, default=200)  # 每个试验采用相同的最大训练轮数。
    parser.add_argument("--patience", type=int, default=20)  # 每个试验采用相同的早停耐心值。
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")  # 自动选择GPU或由用户显式指定。
    parser.add_argument("--output", type=Path)  # 可指定一个尚不存在的新搜索目录。
    parser.add_argument("--resume", type=Path)  # 恢复已有搜索目录，自动跳过已完成试验。
    parser.add_argument("--dry-run", action="store_true")  # 只打印计划，不训练也不创建结果目录。
    args = parser.parse_args()  # 解析用户参数。
    if args.output and args.resume:  # 新建和恢复目录不能同时指定。
        parser.error("--output和--resume不能同时使用")  # 提示修正互斥参数。
    if min(args.epochs, args.patience, *args.batch_sizes) < 1:  # 训练轮数、耐心值和批大小必须为正。
        parser.error("epochs、patience和batch-sizes必须为正整数")  # 阻止无效训练任务。
    if any(not math.isfinite(lr) or lr <= 0 for lr in args.lrs):  # 学习率必须为正且有限。
        parser.error("所有学习率必须为正且有限")  # 阻止NaN或无穷大学习率。
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):  # NumPy种子必须在支持的范围内。
        parser.error("种子必须位于0到2**32-1之间")  # 提前指出非法种子。
    args.data = args.data.resolve()  # 子进程使用绝对数据路径。
    metadata = (args.metadata or args.data.with_suffix(".json")).resolve()  # 解析配套元数据位置。
    for path in [args.data, metadata, ROOT / "run.py", ROOT / "model.py"]:  # 搜索前检查所有必需文件。
        if not path.is_file():  # 缺失文件时不启动任何训练。
            parser.error("找不到文件：" + str(path))  # 显示具体缺失路径。
    combinations = list(itertools.product(dict.fromkeys(args.lrs), dict.fromkeys(args.batch_sizes)))  # 保留输入顺序并去除重复配置。
    configurations = [dict(id="config_{:03d}".format(i + 1), lr=lr, batch_size=batch) for i, (lr, batch) in enumerate(combinations)]  # 为每个组合生成稳定编号。
    plan = dict(data=str(args.data), metadata=str(metadata), data_sha256=digest(args.data), metadata_sha256=digest(metadata), run_sha256=digest(ROOT / "run.py"), model_sha256=digest(ROOT / "model.py"), search_sha256=digest(Path(__file__).resolve()), configurations=configurations, seeds=list(dict.fromkeys(args.seeds)), epochs=args.epochs, patience=args.patience, device=args.device, python=sys.executable, selection="validation only; no automatic test")  # 保存搜索范围及代码和数据版本。
    print("共{}组配置，{}个种子，合计{}次训练；不改网络结构。".format(len(configurations), len(plan["seeds"]), len(configurations) * len(plan["seeds"])), flush=True)  # 提前显示实际搜索预算。
    if args.dry_run:  # 只查看计划时立即退出。
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)  # 打印完整计划供用户核对。
        return  # 不启动训练或创建结果目录。
    output = (args.resume or args.output or ROOT / "search_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()  # 每次搜索放入独立目录。
    if args.resume:  # 恢复时严格校验设置和文件版本。
        if read_json(output / "search_plan.json") != plan:  # 禁止在同一搜索中混入不同设置或数据。
            parser.error("恢复参数或代码、数据版本不同；请使用原参数恢复，或新建搜索")  # 要求完整保留原搜索参数。
    else:  # 新建搜索时不覆盖已有目录。
        output.mkdir(parents=True, exist_ok=False)  # 防止误覆盖旧实验。
        write_json(output / "search_plan.json", plan)  # 训练开始前固定并记录搜索计划。
        (output / "search_snapshot.py").write_bytes(Path(__file__).read_bytes())  # 保存搜索脚本快照便于审查。
    refresh(output, plan)  # 恢复时先重建已完成试验的汇总。
    for config in configurations:  # 按固定顺序逐个执行配置，不并行争抢GPU。
        for seed in plan["seeds"]:  # 对每个配置采用相同种子列表。
            trial = output / config["id"] / ("seed_" + str(seed))  # 为每个配置和种子创建独立目录。
            if (trial / "completed.json").exists():  # 已完成试验不会重复训练。
                print("跳过已完成：{} seed={}".format(config["id"], seed), flush=True)  # 显示恢复进度。
                continue  # 转到下一个种子。
            trial.mkdir(parents=True, exist_ok=True)  # 为首次运行或重试准备目录。
            attempt = trial / ("attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))  # 重试保留失败记录，不删除或覆盖旧输出。
            attempt.mkdir()  # 为本次尝试建立日志目录。
            run_directory = attempt / "training"  # 由run.py创建实际训练目录。
            command = [sys.executable, "-B", "-u", str(ROOT / "run.py"), "--mode", "train", "--data", str(args.data), "--metadata", str(metadata), "--lr", str(config["lr"]), "--batch-size", str(config["batch_size"]), "--seed", str(seed), "--epochs", str(args.epochs), "--patience", str(args.patience), "--device", args.device, "--output", str(run_directory)]  # 只调用训练模式，绝不调用测试模式。
            write_json(attempt / "command.json", command)  # 保存实际执行参数。
            print("开始{}：lr={}，batch={}，seed={}".format(config["id"], config["lr"], config["batch_size"], seed), flush=True)  # 每次试验前显示配置。
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")  # 保证子进程中文日志编码一致。
            with (attempt / "console.log").open("w", encoding="utf-8") as logfile:  # 同时将训练进度保存到文件。
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=environment, cwd=str(ROOT))  # 使用当前环境启动独立训练。
                try:  # 捕捉用户中断并停止当前子进程。
                    for line in process.stdout:  # 实时读取每轮训练日志。
                        print(line, end="", flush=True)  # 在控制台持续显示，避免误以为没有运行。
                        logfile.write(line)  # 写入本次尝试的完整日志。
                        logfile.flush()  # 中途终止时也尽量保留最新日志。
                    returncode = process.wait()  # 等待训练完成并获取退出码。
                except BaseException:  # 用户中断或读取失败时清理训练进程。
                    process.terminate()  # 请求子进程结束。
                    process.wait()  # 等待释放资源后再返回异常。
                    raise  # 保留原始中断信息。
                finally:  # 不论成功失败都关闭输出管道。
                    process.stdout.close()  # 释放管道句柄。
            if returncode != 0:  # 失败试验不写入完成记录或排名。
                raise RuntimeError("训练失败，请查看日志后使用相同参数加--resume恢复：" + str(attempt / "console.log"))  # 停止搜索避免反复执行同类错误。
            report = read_json(run_directory / "validation_metrics.json")  # 核实训练确实生成了验证结果。
            if not math.isfinite(float(report["mean_error_3d"])) or not (run_directory / "best.pt").is_file():  # 验证结果和检查点都必须完整。
                raise ValueError("训练输出不完整：" + str(run_directory))  # 不把部分输出标记为完成。
            write_json(trial / "completed.json", dict(run_directory=str(run_directory)))  # 仅完整成功后记录完成状态。
            ranking = refresh(output, plan)  # 每次完成立即更新进度及最佳配置。
            if ranking:  # 有完整配置时显示当前最佳验证分数。
                print("当前最佳：{}，验证平均误差={:.6f} m".format(ranking[0]["config_id"], ranking[0]["val_mean_error_3d"]), flush=True)  # 此分数不是测试误差。
    print("搜索完成：" + str(output / "best_config.json"), flush=True)  # 告知最终配置文件位置。
    print("仅按验证集选择配置；未执行测试。模型路径见best_config.json中的checkpoints。", flush=True)  # 提醒后续显式测试选定模型。

if __name__ == "__main__":  # 只在直接运行脚本时启动搜索。
    main()  # 执行带中文逐行注释的网格搜索入口。
