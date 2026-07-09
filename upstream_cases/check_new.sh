dirA=./tests
dirB=/home/yf/vllm/tests
save_file=new.txt
rm $save_file
touch $save_file

find "$dirA" -type f -name "test*.py" -print0 | while IFS= read -r -d '' file; do
    relpath="${file#$dirA/}"
    if [ ! -f "$dirB/$relpath" ]; then
        echo "tests/$relpath" >> $save_file
    fi
done

