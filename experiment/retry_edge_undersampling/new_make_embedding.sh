emb_size=1024

echo "Running deepwalk"

pathb="/home/tongli/walkpred/experiment/retry_edge_undersampling/data"

cd ~/deepwalk

source venv/bin/activate

deepwalk --input "${pathb}/min_hetionet/minhet_adj.txt" --output "${pathb}/min_hetionet/minhet_emb_${emb_size}.txt" --representation-size $emb_size --number-walks 40 --window-size 10 --workers 4

deactivate
