#!/usr/bin/env python3
"""physics-models 统一测试入口。

自动发现 models/mechanics/MEC-* 目录并逐个运行测试。
不依赖手工维护的模型列表；新增 MEC 模型后自动被发现。

运行方式（在项目根目录执行）：
    python run_all_tests.py          # 运行全部测试
    python run_all_tests.py -v       # 详细输出
    python run_all_tests.py --pytest # 用 pytest 收集并运行

设计说明：
    各模型的 test 文件使用 `from model import ...` 裸模块名导入，
    在同一个 Python 进程中批量收集会导致 sys.modules['model'] 冲突。
    本脚本通过为每个模型目录启动子进程来隔离导入，避免冲突。
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def discover_models(base_dir: Path) -> list[tuple[str, Path, str]]:
    """自动发现所有 MEC 模型目录及其测试文件。

    返回 [(model_name, model_dir, test_filename), ...]
    """
    results = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith("MEC-"):
            continue
        # 查找 test_MEC*_consistency.py
        test_files = list(entry.glob("test_MEC*_consistency.py"))
        if not test_files:
            continue
        test_file = test_files[0].name
        results.append((entry.name, entry, test_file))
    return results


def run_model_test(model_name: str, model_dir: Path, test_file: str,
                   verbose: bool = False) -> tuple[bool, int, int, int, int, str]:
    """在子进程中运行单个模型的测试。

    返回 (passed, n_pass, n_fail, n_skip, n_warn, summary_line)
    """
    # 在模型目录中运行测试，确保 from model import 正确解析
    cmd = [sys.executable, test_file]
    result = subprocess.run(
        cmd,
        cwd=str(model_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = result.stdout + result.stderr

    # 解析输出
    # 格式: "✓ xxx" 行数 + "OK: MEC-xxx 所有一致性测试通过"
    ok_line = ""
    for line in output.split("\n"):
        if line.startswith("OK:"):
            ok_line = line
            break

    passed = result.returncode == 0

    # 统计测试数（✓ 行数 + OK 行）
    check_lines = [l for l in output.split("\n") if l.strip().startswith("✓")]
    ok_lines = [l for l in output.split("\n") if l.strip().startswith("OK:")]
    # 如果有 ✓ 行用 ✓ 行计数；否则用 OK 行计数为 1（至少 1 个测试通过）
    if check_lines:
        n_pass = len(check_lines)
    elif ok_lines:
        n_pass = 1  # 旧格式只有 OK 行，至少 1 个测试
    else:
        n_pass = 0
    n_fail = 0 if passed else 1
    n_skip = 0
    n_warn = 0

    if verbose:
        for line in output.split("\n"):
            if line.strip():
                print(f"  [{model_name}] {line}")

    return passed, n_pass, n_fail, n_skip, n_warn, ok_line


def main():
    parser = argparse.ArgumentParser(
        description="physics-models 统一测试入口"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="显示每个测试的详细输出"
    )
    parser.add_argument(
        "--pytest", action="store_true",
        help="使用 pytest 运行（而非直接 python）"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent
    mechanics_dir = project_root / "models" / "mechanics"

    if not mechanics_dir.exists():
        print(f"错误: 未找到 {mechanics_dir}")
        sys.exit(1)

    # 自动发现模型
    models = discover_models(mechanics_dir)

    if not models:
        print("错误: 未发现任何 MEC 模型目录")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"physics-models 全项目测试")
    print(f"{'=' * 70}")
    print(f"发现 {len(models)} 个模型\n")

    total_pass = 0
    total_fail = 0
    total_skip = 0
    total_warn = 0
    failed_models = []

    for model_name, model_dir, test_file in models:
        if args.pytest:
            # 使用 pytest 运行
            cmd = [sys.executable, "-m", "pytest", test_file,
                    "--tb=short", "-q"]
            result = subprocess.run(
                cmd, cwd=str(model_dir),
                capture_output=True, text=True, timeout=300
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # 解析 pytest 输出
            n_p = n_f = n_s = n_w = 0
            for line in output.split("\n"):
                m = re.search(r"(\d+) passed", line)
                if m: n_p = int(m.group(1))
                m = re.search(r"(\d+) failed", line)
                if m: n_f = int(m.group(1))
                m = re.search(r"(\d+) skipped", line)
                if m: n_s = int(m.group(1))
                m = re.search(r"(\d+) warnings", line)
                if m: n_w = int(m.group(1))

            total_pass += n_p
            total_fail += n_f
            total_skip += n_s
            total_warn += n_w

            status = "✅" if passed else "❌"
            detail = f"{n_p} passed"
            if n_f: detail += f", {n_f} failed"
            if n_w: detail += f", {n_w} warnings"
            print(f"  {status} {model_name}: {detail}")

            if not passed:
                failed_models.append(model_name)
                if args.verbose:
                    print(output)
        else:
            # 直接 python 运行
            passed, n_p, n_f, n_s, n_w, ok_line = run_model_test(
                model_name, model_dir, test_file, args.verbose
            )
            total_pass += n_p
            total_fail += n_f
            total_skip += n_s
            total_warn += n_w

            status = "✅" if passed else "❌"
            detail = f"{n_p} tests"
            if not passed:
                detail += f" ({n_f} failed)"
                failed_models.append(model_name)
            print(f"  {status} {model_name}: {detail}")

    print(f"\n{'=' * 70}")
    print(f"汇总")
    print(f"{'=' * 70}")
    print(f"  模型总数:    {len(models)}")
    print(f"  总测试数:    {total_pass + total_fail + total_skip}")
    print(f"  passed:      {total_pass}")
    print(f"  failed:      {total_fail}")
    print(f"  skipped:     {total_skip}")
    print(f"  warnings:    {total_warn}")

    if failed_models:
        print(f"\n  失败模型: {', '.join(failed_models)}")
        sys.exit(1)
    else:
        print(f"\n  ✅ 全部 {len(models)} 个模型测试通过")
        sys.exit(0)


if __name__ == "__main__":
    main()
