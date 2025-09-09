python main.py --model_type masked_reconstruction --epochs 50

nohup python main.py --model_type masked_reconstruction --mask_ratio 0.3 --hidden_size 256 --epochs 200 --num_workers 16 --batch_size 128 &


nohup python main.py --model_type contrastive --sequence_length 2000 --overlap 0.75 --hidden_size 256 --epochs 200 --num_workers 16 --batch_size 128 &




nohup python main.py --model_type denoising --sequence_length 2000 --overlap 0.875 --hidden_size 256 --epochs 200 --num_workers 16 --batch_size 128 &



nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.875 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 &



nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 01 04 27 18 31 12 16 25 20 38 45 09 52 50 05 22 06 35 33 17 47 42 30 37 08 46 41 36 48 34 03 14 07 15  --val_subjects 11 51 10 39 19 29 21 13 --test_subjects 40 44 24 49 32 02 53 43 26 23 28 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 44 49 32 02 43 26 28 25 20 38 29 19 45 39 52 50 05 22 35 33 17 47 42 37 46 41 36 48 34 03 07 11 15  --val_subjects 24 09 23 30 06 53 14 08 --test_subjects 01 10 04 27 18 31 13 51 21 12 16 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 44 24 49 32 02 53 26 23 28 10 04 18 31 13 21 12 16 22 06 33 47 42 30 37 08 46 48 34 03 14 07 11 15  --val_subjects 17 41 36 43 01 35 27 51 --test_subjects 25 20 38 29 19 45 39 09 52 50 05 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 24 49 32 53 43 26 23 28 01 10 04 27 18 51 12 25 20 38 29 45 39 09 52 05 46 41 36 48 34 03 14 07 11 15  --val_subjects 13 19 50 31 02 21 16 44 --test_subjects 22 06 35 33 17 47 42 30 37 08 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 44 49 02 53 43 26 28 01 10 04 18 31 13 51 21 12 16 25 20 38 19 45 39 09 52 50 05 06 35 33 47 30 37 08  --val_subjects 32 29 24 17 42 27 22 23 --test_subjects 46 41 36 48 34 03 14 07 11 15 &




nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 01 04 27 18 31 12 16 25 20 38 45 09 52 50 05 22 06 35 33 17 47 42 30 37 08 46 41 36 48 34 03 14 07 15 1009 1015 1016 1018 1023 1028 1029 1030 1031 1032 1035 1038 1103 1104 1105 1115 1121 1122 1123 1125 1127 1128 1133 1134 1142 1147 1148 1149 1150 1309 1311 1312 1313 1322 1325 1328 1329 1330 1331 1332 1333 1370 --val_subjects 11 51 10 39 19 29 21 13 --test_subjects 40 44 24 49 32 02 53 43 26 23 28 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 44 49 32 02 43 26 28 25 20 38 29 19 45 39 52 50 05 22 35 33 17 47 42 37 46 41 36 48 34 03 07 11 15 1009 1015 1016 1018 1023 1028 1029 1030 1031 1032 1035 1038 1103 1104 1105 1115 1121 1122 1123 1125 1127 1128 1133 1134 1142 1147 1148 1149 1150 1309 1311 1312 1313 1322 1325 1328 1329 1330 1331 1332 1333 1370 --val_subjects 24 09 23 30 06 53 14 08 --test_subjects 01 10 04 27 18 31 13 51 21 12 16 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 44 24 49 32 02 53 26 23 28 10 04 18 31 13 21 12 16 22 06 33 47 42 30 37 08 46 48 34 03 14 07 11 15 1009 1015 1016 1018 1023 1028 1029 1030 1031 1032 1035 1038 1103 1104 1105 1115 1121 1122 1123 1125 1127 1128 1133 1134 1142 1147 1148 1149 1150 1309 1311 1312 1313 1322 1325 1328 1329 1330 1331 1332 1333 1370 --val_subjects 17 41 36 43 01 35 27 51 --test_subjects 25 20 38 29 19 45 39 09 52 50 05 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 24 49 32 53 43 26 23 28 01 10 04 27 18 51 12 25 20 38 29 45 39 09 52 05 46 41 36 48 34 03 14 07 11 15 1009 1015 1016 1018 1023 1028 1029 1030 1031 1032 1035 1038 1103 1104 1105 1115 1121 1122 1123 1125 1127 1128 1133 1134 1142 1147 1148 1149 1150 1309 1311 1312 1313 1322 1325 1328 1329 1330 1331 1332 1333 1370 --val_subjects 13 19 50 31 02 21 16 44 --test_subjects 22 06 35 33 17 47 42 30 37 08 &
nohup python main.py --model_type contrastive --sequence_length 4000 --overlap 0.75 --hidden_size 256 --epochs 400 --num_workers 16 --batch_size 128 --train_subjects 40 44 49 02 53 43 26 28 01 10 04 18 31 13 51 21 12 16 25 20 38 19 45 39 09 52 50 05 06 35 33 47 30 37 08 1009 1015 1016 1018 1023 1028 1029 1030 1031 1032 1035 1038 1103 1104 1105 1115 1121 1122 1123 1125 1127 1128 1133 1134 1142 1147 1148 1149 1150 1309 1311 1312 1313 1322 1325 1328 1329 1330 1331 1332 1333 1370 --val_subjects 32 29 24 17 42 27 22 23 --test_subjects 46 41 36 48 34 03 14 07 11 15 &