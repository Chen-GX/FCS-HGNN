#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
EXEC="../src/main.py"
output_dir="../../../output_dir/V3_ls/run"

data_name='ACM'
epoch=200
search_homo=False

for batch_size in 32
do
    for patience in 5
    do
        for edge_feats in 32
        do
            for hidden_dim in 128
            do
                for residual in 'False'
                do
                    for use_batchnorm in 'True'
                    do
                        for max_depth in 1 2 3 4 5 6 7 8 9 10
                        do
                            python $EXEC \
                            --epoch $epoch \
                            --data_name $data_name \
                            --patience $patience \
                            --batch_size $batch_size \
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
done
