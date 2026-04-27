#!/bin/bash
unset ftp_proxy
unset https_proxy
unset http_proxy

nic_name="enp48s3u1u1"
local_ip="141.xx.xx.101"

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
# AIV
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export ASCEND_BUFFER_POOL=4:8
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export HCCL_BUFFSIZE=1800
export VLLM_ASCEND_ENABLE_MLAPO=1
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /mnt/share/kimi25_w4a8_static_m6 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --quantization ascend \
    --served-model-name kimi25 \
    --trust-remote-code \
    --max-num-seqs 48 \
    --max-model-len 32768 \
    --max-num-batched-tokens 256 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4,8,16,32,48,64,80,96,112,128,144,160]}' \
    --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": false}' \
    --speculative_config '{"method": "eagle3", "model":"/mnt/share/weight/lightseekorg_kimi-k2.5-eagle3", "num_speculative_tokens": 3}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30200",
    "engine_id": "2",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
    "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 2,
                        "tp_size": 8
                },
                "decode": {
                        "dp_size": 32,
                        "tp_size": 1
                }
        }
    }'\
    2>&1 | tee 1p1d_d1.log
