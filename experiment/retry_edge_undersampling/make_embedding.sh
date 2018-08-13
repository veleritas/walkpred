idx=$1

emb_size=16

echo "Running deepwalk for fold $idx"

pathb="/home/tongli/walkpred/experiment/retry_edge_undersampling/data/min_hetionet/test"

cd ~/dw_toby_debug
#cd ~/deepwalk

source venv/bin/activate

deepwalk --input "${pathb}/adjlist_${idx}.txt" --output "${pathb}/embedding_${emb_size}_${idx}.txt" --representation-size $emb_size --number-walks 40 --window-size 10 --workers 4


deactivate
