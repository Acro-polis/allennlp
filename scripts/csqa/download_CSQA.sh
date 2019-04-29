#!/usr/bin/env bash

# create data dir
#mkdir -p data

out_dir="/home/ubuntu/Desktop"
unzip_dir=$out_dir"/CSQA_v9"
fileid="1dgf-Qjvhfv-_EWoDjrTCAY5CwYCw-djt"
filename="CSQA_v9.zip"

# download file if not exists and save to folder
# NOTE!!! CSQA is in capitals, v9 is with a small v

if [ ! -d $unzip_dir ]; then
    echo "downloading, and unzipping to "$unzip_dir
    # some hacks to deal with large files on google drive
    cd $out_dir

    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

    unzip $filename
    rm $filename
    rm cookie
fi
