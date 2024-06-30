# train
python efficientad.py --dataset mvtec_ad  \
                      --mvtec_ad_path hefei_ad \
                      --subdataset backpack-box 

# evaluate
# python mvtec_ad_evaluation/evaluate_experiment.py --dataset_base_dir './mvtec_anomaly_detection/' --anomaly_maps_dir './output/1/anomaly_maps/mvtec_ad/' --output_dir './output/1/metrics/mvtec_ad/' --evaluated_objects bottlehe 