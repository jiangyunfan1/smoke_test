#!/bin/bash

logdir="20251029qwen25vl7b1P1D_1024pstress"  # 修改1：保存日志的文件夹
mkdir -p $logdir 

# 获取主机名和起始时间，只获取一次
ip=$(ifconfig | grep -oE '90\.90\.[0-9]+\.[0-9]+' | head -n 1)  # 修改2：获取主机名，查找以90.90开头的IP，如果是141.61.开头，对应修改
datetime=$(date '+%Y%m%d_%H%M%S')
logfile="$logdir/${ip}_${datetime}.log"   # 日志命名，可自定义

# 无限循环，每分钟采样一次
while true; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"    # 获取当前时间
    {
        echo "$now"
        ps -ef | grep -v "\["             # 使用ps命令获取进程信息，grep 过滤掉内核进程
        echo "----------------------"
        free -m                           # 使用 free -m 查看整机内存
            echo "----------------------"
            /usr/local/bin/smem               # 使用 smem 查看进程内存
        echo "======================"
    } | tee -a "$logfile"
    sleep 60                              # 每隔60秒采集1次，可自定义
done