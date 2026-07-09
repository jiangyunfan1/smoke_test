input_file="new18.txt"

log_dir="/home/yf/analyse_log_v0180"

while IFS= read -r line; do
    logfile="${log_dir}/${line%.py}.log"
    dir_name=$(dirname $logfile)
    mkdir -p ${dir_name}
    if [ -f "$logfile" ]; then
        continue
    fi
    echo "testing $line"
    pytest -sv $line | tee $logfile
done < "$input_file"
