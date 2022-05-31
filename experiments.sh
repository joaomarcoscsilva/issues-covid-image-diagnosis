# python3 run.py --cv 5 configs/crossdata/mendeley_curated.json --wandb --save --name from_scratch_mendeley_curated -f
# python3 run.py --cv 5 configs/crossdata/tawsifur_curated.json --wandb --save --name from_scratch_tawsifur_curated -f
# python3 run.py --cv 5 configs/crossdata/covidx_curated.json --wandb --save --name from_scratch_covidx_curated -f

python3 run.py --cv 5 configs/crossdata/mendeley_covidx_curated.json --wandb --save --name from_scratch_mendeley_covidx_curated -f

# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_scratch_0.15 --cv 5
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_scratch_0.3 --cv 5
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_scratch_0.45 --cv 5
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_scratch_0.6 --cv 5
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_scratch_0.75 --cv 5
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_scratch_0.9 --cv 5

# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_0.15 --load models/from_scratch_mendeley_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_0.15 --load models/from_scratch_mendeley_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_0.15 --load models/from_scratch_mendeley_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_0.15 --load models/from_scratch_mendeley_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_0.15 --load models/from_scratch_mendeley_curated_4.pickle --cv 5 --cv-id 4
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_covidx_0.15 --load models/from_scratch_covidx_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_covidx_0.15 --load models/from_scratch_covidx_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_covidx_0.15 --load models/from_scratch_covidx_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_covidx_0.15 --load models/from_scratch_covidx_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_covidx_0.15 --load models/from_scratch_covidx_curated_4.pickle --cv 5 --cv-id 4

# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_0.3 --load models/from_scratch_mendeley_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_0.3 --load models/from_scratch_mendeley_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_0.3 --load models/from_scratch_mendeley_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_0.3 --load models/from_scratch_mendeley_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_0.3 --load models/from_scratch_mendeley_curated_4.pickle --cv 5 --cv-id 4
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_covidx_0.3 --load models/from_scratch_covidx_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_covidx_0.3 --load models/from_scratch_covidx_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_covidx_0.3 --load models/from_scratch_covidx_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_covidx_0.3 --load models/from_scratch_covidx_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_covidx_0.3 --load models/from_scratch_covidx_curated_4.pickle --cv 5 --cv-id 4

# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_0.45 --load models/from_scratch_mendeley_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_0.45 --load models/from_scratch_mendeley_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_0.45 --load models/from_scratch_mendeley_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_0.45 --load models/from_scratch_mendeley_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_0.45 --load models/from_scratch_mendeley_curated_4.pickle --cv 5 --cv-id 4
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_covidx_0.45 --load models/from_scratch_covidx_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_covidx_0.45 --load models/from_scratch_covidx_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_covidx_0.45 --load models/from_scratch_covidx_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_covidx_0.45 --load models/from_scratch_covidx_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_covidx_0.45 --load models/from_scratch_covidx_curated_4.pickle --cv 5 --cv-id 4

# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_0.6 --load models/from_scratch_mendeley_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_0.6 --load models/from_scratch_mendeley_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_0.6 --load models/from_scratch_mendeley_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_0.6 --load models/from_scratch_mendeley_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_0.6 --load models/from_scratch_mendeley_curated_4.pickle --cv 5 --cv-id 4
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_covidx_0.6 --load models/from_scratch_covidx_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_covidx_0.6 --load models/from_scratch_covidx_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_covidx_0.6 --load models/from_scratch_covidx_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_covidx_0.6 --load models/from_scratch_covidx_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_covidx_0.6 --load models/from_scratch_covidx_curated_4.pickle --cv 5 --cv-id 4

# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_0.75 --load models/from_scratch_mendeley_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_0.75 --load models/from_scratch_mendeley_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_0.75 --load models/from_scratch_mendeley_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_0.75 --load models/from_scratch_mendeley_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_0.75 --load models/from_scratch_mendeley_curated_4.pickle --cv 5 --cv-id 4
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_covidx_0.75 --load models/from_scratch_covidx_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_covidx_0.75 --load models/from_scratch_covidx_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_covidx_0.75 --load models/from_scratch_covidx_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_covidx_0.75 --load models/from_scratch_covidx_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_covidx_0.75 --load models/from_scratch_covidx_curated_4.pickle --cv 5 --cv-id 4

# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_0.9 --load models/from_scratch_mendeley_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_0.9 --load models/from_scratch_mendeley_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_0.9 --load models/from_scratch_mendeley_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_0.9 --load models/from_scratch_mendeley_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_0.9 --load models/from_scratch_mendeley_curated_4.pickle --cv 5 --cv-id 4
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_covidx_0.9 --load models/from_scratch_covidx_curated_0.pickle --cv 5 --cv-id 0
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_covidx_0.9 --load models/from_scratch_covidx_curated_1.pickle --cv 5 --cv-id 1
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_covidx_0.9 --load models/from_scratch_covidx_curated_2.pickle --cv 5 --cv-id 2
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_covidx_0.9 --load models/from_scratch_covidx_curated_3.pickle --cv 5 --cv-id 3
# python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_covidx_0.9 --load models/from_scratch_covidx_curated_4.pickle --cv 5 --cv-id 4









python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_covidx_0.15 --load models/from_scratch_mendeley_covidx_curated_0.pickle --cv 5 --cv-id 0
python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_covidx_0.15 --load models/from_scratch_mendeley_covidx_curated_1.pickle --cv 5 --cv-id 1
python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_covidx_0.15 --load models/from_scratch_mendeley_covidx_curated_2.pickle --cv 5 --cv-id 2
python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_covidx_0.15 --load models/from_scratch_mendeley_covidx_curated_3.pickle --cv 5 --cv-id 3
python3 run.py configs/tawsifur_finetune/0.15.json --wandb --name from_mendeley_covidx_0.15 --load models/from_scratch_mendeley_covidx_curated_4.pickle --cv 5 --cv-id 4

python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_covidx_0.3 --load models/from_scratch_mendeley_covidx_curated_0.pickle --cv 5 --cv-id 0
python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_covidx_0.3 --load models/from_scratch_mendeley_covidx_curated_1.pickle --cv 5 --cv-id 1
python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_covidx_0.3 --load models/from_scratch_mendeley_covidx_curated_2.pickle --cv 5 --cv-id 2
python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_covidx_0.3 --load models/from_scratch_mendeley_covidx_curated_3.pickle --cv 5 --cv-id 3
python3 run.py configs/tawsifur_finetune/0.3.json --wandb --name from_mendeley_covidx_0.3 --load models/from_scratch_mendeley_covidx_curated_4.pickle --cv 5 --cv-id 4

python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_covidx_0.45 --load models/from_scratch_mendeley_covidx_curated_0.pickle --cv 5 --cv-id 0
python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_covidx_0.45 --load models/from_scratch_mendeley_covidx_curated_1.pickle --cv 5 --cv-id 1
python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_covidx_0.45 --load models/from_scratch_mendeley_covidx_curated_2.pickle --cv 5 --cv-id 2
python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_covidx_0.45 --load models/from_scratch_mendeley_covidx_curated_3.pickle --cv 5 --cv-id 3
python3 run.py configs/tawsifur_finetune/0.45.json --wandb --name from_mendeley_covidx_0.45 --load models/from_scratch_mendeley_covidx_curated_4.pickle --cv 5 --cv-id 4

python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_covidx_0.6 --load models/from_scratch_mendeley_covidx_curated_0.pickle --cv 5 --cv-id 0
python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_covidx_0.6 --load models/from_scratch_mendeley_covidx_curated_1.pickle --cv 5 --cv-id 1
python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_covidx_0.6 --load models/from_scratch_mendeley_covidx_curated_2.pickle --cv 5 --cv-id 2
python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_covidx_0.6 --load models/from_scratch_mendeley_covidx_curated_3.pickle --cv 5 --cv-id 3
python3 run.py configs/tawsifur_finetune/0.6.json --wandb --name from_mendeley_covidx_0.6 --load models/from_scratch_mendeley_covidx_curated_4.pickle --cv 5 --cv-id 4

python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_covidx_0.75 --load models/from_scratch_mendeley_covidx_curated_0.pickle --cv 5 --cv-id 0
python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_covidx_0.75 --load models/from_scratch_mendeley_covidx_curated_1.pickle --cv 5 --cv-id 1
python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_covidx_0.75 --load models/from_scratch_mendeley_covidx_curated_2.pickle --cv 5 --cv-id 2
python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_covidx_0.75 --load models/from_scratch_mendeley_covidx_curated_3.pickle --cv 5 --cv-id 3
python3 run.py configs/tawsifur_finetune/0.75.json --wandb --name from_mendeley_covidx_0.75 --load models/from_scratch_mendeley_covidx_curated_4.pickle --cv 5 --cv-id 4

python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_covidx_0.9 --load models/from_scratch_mendeley_covidx_curated_0.pickle --cv 5 --cv-id 0
python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_covidx_0.9 --load models/from_scratch_mendeley_covidx_curated_1.pickle --cv 5 --cv-id 1
python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_covidx_0.9 --load models/from_scratch_mendeley_covidx_curated_2.pickle --cv 5 --cv-id 2
python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_covidx_0.9 --load models/from_scratch_mendeley_covidx_curated_3.pickle --cv 5 --cv-id 3
python3 run.py configs/tawsifur_finetune/0.9.json --wandb --name from_mendeley_covidx_0.9 --load models/from_scratch_mendeley_covidx_curated_4.pickle --cv 5 --cv-id 4