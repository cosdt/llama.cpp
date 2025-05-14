#!/bin/bash

# Display help information
usage() {
    echo "Usage: $0 [Options] [Arguments...]"
    echo ""
    echo "Options:"
    echo "  -h,  --help                Show this help message"
    echo "  -m,  --models              Specify one or more models, separated by ','"
    echo "  -f,  --file FILE           Specify the input file(.txt),with model names separated by spaces or newlines(default: all models of the ollama library)"
    echo "  -d, --down_model_path      Specify the model download directory (default: current directory)"
    echo "  -b, --llamaCpp_path        Absolute path of llama.cpp"
    exit 0
}

# default value
file_name=""
models=""
model_path="."
llamaCpp_path=""
# 重试次数
max_retries=3
# 重试间隔（秒）
retry_interval=6

declare -a model_names

run_inference_with_expect() {
    local model_path="$1"   # .gguf path
    local prompt="$2"       
    local timeout_val="$3"  
    local llama_cli_path="$4"  # llama-cli path

    expect <<EOF
        set timeout $timeout_val
        spawn $llama_cli_path -m $model_path -ngl 99 -p "$prompt"

        expect {
            "$prompt" {
                puts "Start inferencing..."
            }
        }

        expect {
            ">" {
                puts "Waiting for user input, Sending Ctrl+C"
                send "\x03"
            }
            timeout {
                puts "timeout, Send Ctrl+C"
                send "\x03"
            }
            eof {
                puts "progress exit!!!"
                exit 1
            }
        }

        expect eof
EOF
}

# Parsing command line arguments
while getopts "hm:f:d:b:" opt; do
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
	    d)
            model_path=$OPTARG
            ;;
	    b)
            llamaCpp_path=$OPTARG
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

pip install -r $llamaCpp_path/requirements/requirements-convert_hf_to_gguf.txt huggingface_hub --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple > /dev/null 2>&1
pip install huggingface_hub -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple > /dev/null 2>&1
export HF_ENDPOINT=https://hf-mirror.com
if ! command -v expect &> /dev/null; then
    echo "expect is not installed. Installing expect..."
    sudo apt-get update
    sudo apt-get install -y expect

    if [ $? -ne 0 ]; then
        echo "expect installation failed. Please check your network or APT source configuration."
        exit 1
    fi
fi

# acquire model name
if [ ! -z "$models" ]; then
    IFS=',' read -r -a model_names <<< "$models"
elif [ ! -z "$file_name" ];then
    echo "Reading contents of $file_name"
    model_names=($(cat $file_name))
fi

for model in "${model_names[@]}"; do
    if [ -n "$model" ]; then
        download_path=""
        attempt=0

        while [ $attempt -lt $max_retries ]; do
            if [[ "$model" == *.gguf ]]; then
                # === 情况1：具体 .gguf 文件 ===
                repo_id=$(echo "$model" | cut -d'/' -f1,2)
                gguf_file=$(basename "$model")
                model_name=$(echo "$repo_id" | cut -d'/' -f2)
                download_path="$model_path/gguf/"
                gguf_path="$download_path/$gguf_file"
                echo "============Downloading single .gguf file: $gguf_file from $repo_id============"
                # 判断文件是否已存在
                if [ -f "$gguf_path" ]; then
                    echo "-------------------$gguf_file already exists, Skip!!!"
                else
                    python - <<END > /dev/null 2>&1
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="$repo_id",
    filename="$gguf_file",
    local_dir="$download_path",
    local_dir_use_symlinks=False
)
END
                    status=$?
                    if [ $status -eq 0 ]; then
                        echo "$model -------------------Download Success!!! (.gguf)"
                    fi
                fi

                run_inference_with_expect "$gguf_path" "what is a car?" 60 "$llamaCpp_path/build/bin/llama-cli" > /dev/null 2>&1
                ret=$?
                if [ $ret -eq 0 ]; then
                    echo "$model Success inference"
                else
                    echo "$model Failed inference"
                fi
                break
            else
                # === 情况2：模型仓库 ===
                model_name=$(echo "$model" | cut -d'/' -f2)
                download_path="$model_path/hf-models/$model_name"

                echo "============Attempting to download model: $model (Attempt: $((attempt + 1)))============"
                huggingface-cli download "$model" --local-dir "$download_path"
                status=$?

                if [ $status -eq 0 ]; then
                    echo "$model -------------------Download Success!!! "

                    gguf_src="$model_path/$model_name.gguf"
                    if [ -f "$gguf_src" ]; then
                        echo "$model -------------------$gguf_src already exists, Skip!!!"
                    else
                        python "$llamaCpp_path/convert_hf_to_gguf.py" "$download_path" --outfile "$gguf_src" > /dev/null 2>&1
                        if [ $? -eq 0 ]; then
                            echo "$model -----------Convert GGUF Success!"
                        else
                            echo "$model -----------Convert GGUF Failed!"
                            break
                        fi
                    fi
                 
                    quant_tool="$llamaCpp_path/build/bin/llama-quantize"
                    for quant_type in F16 Q4_0 Q8_0; do
                        quant_out="$model_path/gguf/${model_name}_${quant_type}.gguf"

                        if [ -f "$quant_out" ]; then
                            echo "$model -------------------File already exists ($quant_type), Skip!!!"
                        else
                            echo "Quantizing to $quant_type ..."
                            "$quant_tool" "$gguf_src" "$quant_out" "$quant_type" > /dev/null 2>&1

                            if [ $? -eq 0 ]; then
                                echo "$model -----------Quantify Success ($quant_type)"
                            else
                                echo "$model -----------Quantify Failed ($quant_type)"
                            fi
                        fi

                        run_inference_with_expect "$quant_out" "what is a car?" 60 "$llamaCpp_path/build/bin/llama-cli" > /dev/null 2>&1
                        ret=$?
                        if [ $ret -eq 0 ]; then
                            echo "$model Success inference ($quant_type)"
                        else
                            echo "$model Failed inference ($quant_type)"
                        fi
                    done
                    break
                fi
            fi

            echo "+++++++++++++ Failed to download $model (Attempt: $((attempt + 1)))"
            attempt=$((attempt + 1))
            if [ $attempt -lt $max_retries ]; then
                echo "Retrying in $retry_interval seconds..."
                sleep $retry_interval
            else
                echo "============Failed to download $model after $max_retries attempts============"
            fi
        done
    fi
done