fbase=$1
idx=$2

echo "Running deepwalk for $fbase fold $idx"

pathb="/home/tongli/walkpred/experiment/edge_undersampling/tmp/data"

cd ~/deepwalk

source venv/bin/activate


deepwalk --input "${pathb}/adjlist/${fbase}_adj_${idx}.txt" --output "${pathb}/embeddings/${fbase}_embedding_${idx}.txt" --representation-size 16 --number-walks 40 --window-size 10 --workers 4


#deepwalk --input "${pathb}/adjlist/${fbase}_adj_${idx}.txt" --output "${pathb}/embeddings/${fbase}_embedding_${idx}.txt" --representation-size 128 --number-walks 40 --window-size 10 --workers 4

deactivate
