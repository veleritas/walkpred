idx=$1

echo "Running deepwalk for fold $idx"

cd ~/deepwalk

source venv/bin/activate

deepwalk --input ~/walkpred/experiment/data/adjlist/adjlist_$idx.txt --output ~/walkpred/experiment/data/embeddings/embedding_$idx.txt --representation-size 128 --number-walks 50 --window-size 10 --workers 4

deactivate