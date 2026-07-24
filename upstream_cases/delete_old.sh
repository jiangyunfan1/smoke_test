#!/bin/bash
# remove_matching.sh - 删除目标文件中包含指定模式的行及其下一行
# ./remove_matching.sh patterns.txt data.yml > new_data.yml
if [ $# -ne 2 ]; then
    echo "用法: $0 <模式文件> <目标文件>" >&2
    echo "示例: $0 patterns.txt data.yml > new_data.yml" >&2
    exit 1
fi

pattern_file="$1"
target_file="$2"

if [ ! -f "$pattern_file" ]; then
    echo "错误: 模式文件 '$pattern_file' 不存在" >&2
    exit 1
fi

if [ ! -f "$target_file" ]; then
    echo "错误: 目标文件 '$target_file' 不存在" >&2
    exit 1
fi

awk -v patfile="$pattern_file" '
BEGIN {
    # 读取所有模式（忽略空行）
    while ((getline line < patfile) > 0) {
        if (line != "") patterns[line] = 1
    }
    close(patfile)
}
{
    # 如果当前行因上一行匹配而被标记为跳过，则跳过并减少计数
    if (skip > 0) {
        skip--
        next
    }

    # 检查当前行是否包含任一模式（子串匹配）
    for (p in patterns) {
        if (index($0, p) > 0) {
            skip = 1          # 标记下一行也要跳过
            next              # 跳过当前行（不输出）
        }
    }

    # 没有匹配则正常输出
    print
}' "$target_file"
