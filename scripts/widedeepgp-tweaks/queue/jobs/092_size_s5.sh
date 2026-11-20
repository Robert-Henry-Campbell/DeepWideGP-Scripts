# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned size axis: S5 (d 256, eight layers, 8 heads; head 1024).
train $CODE3 size_s5 $CACHE_FLAGS --d_model 256 --e_layers 8 --n_heads 8 --head_hidden 1024 && score $CODE3 size_s5
echo "exit $? at $(stamp)"
