#!/bin/sh
# 用法: ./script.sh input.txt [> output.txt]

if [ $# -eq 0 ]; then
    echo "用法: $0 <文件名>" >&2
    exit 1
fi

input_file="$1"

while IFS= read -r line; do
    printf -- '  - name: %s\n' "$line"
    printf '    estimate_time: 20\n'
done < "$input_file"
