#!/bin/bash

# Display help information
usage() {
    echo "Usage: $0 [Options] [Arguments...]"
    echo ""
    echo "Options:"
    echo "  -h,  --help                Show this help message"
    echo "  -m,  --models              Specify one or more models, separated by ','"
    echo "  -f,  --file FILE           Specify the input file(.txt),with model names separated by spaces or newlines(default: all models of the ollama library)"
    echo "  -s,  --maxsize             Specify the maximum size of the model that can be downloaded (default: 40GB)"
    echo "  -d, --down_model_path      Specify the model download directory (default: current directory)"
    echo "  -b, --llama-bench_path     Absolute path of llama-bench"
    exit 0
}

# default value
file_name=""
models=""
maxsize=40
model_path="."
llama_bench_path=""

declare -a model_names

# Parsing command line arguments
while getopts "hm:f:s:d:b:" opt; do
    case $opt in
        h)
            usage
            exit 0
            ;;
	m)
            models=$OPTARG
            ;;
        f)
            file_name=$OPTARG
	    if [[ ! -f "$file_name" ]]; then
		echo "Error: File '$file_name' not found!"
                exit 1
	    fi
	    file_name=$OPTARG
            ;;
        s)
            maxsize=$OPTARG
            ;;
	d)
            model_path=$OPTARG
            ;;
	b)
            llama_bench_path=$OPTARG
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

# Check if omdd exists in the current directory
if [ ! -f "./omdd" ];then
    wget -O omdd https://github.com/amirrezaDev1378/ollama-model-direct-download/releases/download/v2.1.1/omdd-linux-arm64
    chmod +x omdd
fi

# acquire model name
if [ ! -z "$models" ]; then
    IFS=',' read -r -a model_names <<< "$models"

elif [ ! -z "$file_name" ];then
    echo "Reading contents of $file_name"
    model_names=($(cat $file_name))

else
    url="https://ollama.com/search"
    model_names=($(curl -s "$url" | grep "x-test-search-response-title" | awk -F '>' '{print $2}' | awk -F '<' '{print $1}'))

fi


if [ -f "./model_urls.txt" ]; then
    rm "./model_urls.txt"
fi

# Filter tags with the smallest number of parameters according to the model name.
for model_name in "${model_names[@]}";do
	url="https://ollama.com/library/$model_name/tags"
   	echo "Processing: $url, model_name = $model_name"
	model_tag=($(curl -s "$url" | grep "library/$model_name:"  | grep -oP '(?<=/library/)[^"]+' | grep -E 'q4_0|q8_0|fp16' | sort -t ':' -k2,2 -n | perl -ne 'print if /q4_0/ && !$q4++; print if /q8_0/ && !$q8++; print if /fp16/ && !$fp16++; exit if $q4 && $q8 && $fp16'))
	for tag in "${model_tag[@]}";do
		echo "$tag `./omdd get $tag | grep  '1 -' | awk '{print $3}'`" >> model_urls.txt
	done
done


all_model_args=""

# Limiting the size of downloaded files
max_size=$(($maxsize * 1024 * 1024 * 1024)) 
while read -r name url; do
    echo "Checking size for $name..."

    size=$(curl -sI "$url" | grep -i "Content-Length" | cut -d ' ' -f2 | tr -d '\r')

    if [[ -z "$size" ]]; then
        echo "Warning: Unable to determine size for $name. Skipping..."
        continue
    fi

    if ((size > max_size)); then
	    echo "Skipping $name, file size exceeds $maxsize GB ($((size / 1024 / 1024 / 1024)) GB)"
        continue
    fi
    
    all_model_args="$all_model_args -m $model_path/$name"
    echo "Downloading $name from $url (Size: $((size / 1024 / 1024)) MB)..."
    if [[ -f "$model_path/$name" ]]; then
        local_size=$(stat -c %s "$model_path/$name")
	echo $local_size
        if ((local_size == size)); then
            echo "$name is already fully downloaded. Skipping..."
	    continue
        fi
    fi
    
    wget -c --progress=bar -P "$model_path" -O "$model_path/$name" "$url"

done < model_urls.txt

#echo $llama_bench_path $all_model_args -p \"what is a car?\" -ngl 99 -o csv
llama_bench_res=$($llama_bench_path $all_model_args -p "what is a car?" -ngl 99 -o csv)

echo $llama_bench_res >> llama_bench_data.csv
