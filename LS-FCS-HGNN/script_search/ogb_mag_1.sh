#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
EXEC="../src/main.py"
output_dir="../../../output_dir/V5/run"

data_name='OGB_MAG'
epoch=200
search_homo=False


for patience in 15
do
    for edge_feats in 8   # 16 32
    do
        for hidden_dim in 64   # 128 V100考虑
        do
            for residual in 'False'
            do
                for use_batchnorm in 'True'
                do  
                    for max_depth in 1 2
                    do
                        python $EXEC \
                        --epoch $epoch \
                        --data_name $data_name \
                        --patience $patience \
                        --output_dir $output_dir \
                        --search_homo $search_homo \
                        --edge_feats $edge_feats \
                        --hidden_dim $hidden_dim \
                        --residual $residual \
                        --use_batchnorm $use_batchnorm \
                        --max_depth $max_depth
                    
                    done
                done
            done
        done
    done
done
