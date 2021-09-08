export FASTREID_DATASETS=/export/home/DATA

python3 demo/visualize_result.py --config-file projects/bjzProject/configs/sbs.yml \
--parallel --vis-label --dataset-name "bjzExit" --output projects/bjzProject/logs/vis/r34_ibn_bjzExit_vis \
--opts MODEL.WEIGHTS projects/bjzProject/logs/bjz/r34-ibn_20200930/model_final.pth

