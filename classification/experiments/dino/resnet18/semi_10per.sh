cd sslight/classification && python3 main_cls.py --cfg config/exp_yamls/dino/dino_r18_semi.yaml DATA.PATH_TO_DATA_DIR $IN1K_PATH  OUTPUT_DIR $OUTPUT_PATH MODEL.PRE_TRAINED_PATH $CKPT_PATH TRAIN.SUBSET_FILE_PATH ./data/imagenet_subsets/semi_10percent.txt SOLVER.TOTAL_EPOCHS 30