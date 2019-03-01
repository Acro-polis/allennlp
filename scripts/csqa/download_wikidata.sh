#!/bin/bash

wikidata_dir='~/Desktop/wikidata'

filenames=(
    'wikidata_short_1.json'
    'wikidata_short_2.json'
    'child_all_parents_till_5_levels.json'
    'par_child_dict.json'
    'comp_wikidata_rev.json'
    )
fileids=( 
    '1ST5lqRNlaJlDqZEWe0Nq2Bvl9MyN9vdC'
    '15ctxOZQ68y9cVnZaP-mW9MBpDhRNWG1k'
    '1h2NQSyGM-66JU9IYVDz2I-dAavM322Qy'
    '1pzlX_LJjwZFx-wTFzPsi5wIy59QgrQm4'
    '1YBGZgK6ultWwZveX3vRr5-MN18TIj39b' )

#filename="wikidata_short_1.json"
#fileid="1ST5lqRNlaJlDqZEWe0Nq2Bvl9MyN9vdC"

for index in ${!filenames[*]}; do 
    filename=${filenames[$index]}
    fileid=${fileids[$index]}

    if [ ! -f $filename ]; then
        echo "downloading $filename"
        # some hacks to deal with large files on google drive
        curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
    fi
done

rm cookie
