Dataset：For S3DIS, please refer to http://buildingparser.stanford.edu/dataset.html#Download
Move the uncompressed data folder to `data`.

Pre-processing:
1. cd data_conversions
2. python3 prepare_s3dis_label.py
3. Use `code_for_directions` as before to generate directions information, then name the folder `dir_all_d` and move it to `data/S3DIS/`
4. python3 prepare_s3dis_data.py
5. python3 prepare_s3dis_filelists.py


train：
python  train_val_seg.py -t ../data/S3DIS/prepare_label_rgb/train_files_for_val_on_Area_5.txt -v ../data/S3DIS/prepare_label_rgb/val_files_Area_5.txt -s log -m seg_d_conv_rgb -x s3dis_option

test：
python  eval_s3.py


