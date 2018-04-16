# rop_3cls_combine_1_2
python main.py --arch DualPathNet --resume /home/ubuntu/skin_demo/RoP/models/kaggle/DualPathNet107_downsamping_enhance_DualPathNet107enhance_epoch12_loss0.9639.pkl --data rop_3cls_combine_1_2 --batch_size 32 --lr 0.01 --epochs 100 --save save/rop_3cls_combine_1_2_DualPathNet_100_T2

nohup python -u main.py --arch DualPathNet --resume /home/ubuntu/skin_demo/RoP/models/kaggle/DualPathNet107_downsamping_enhance_DualPathNet107enhance_epoch12_loss0.9639.pkl --data rop_3cls_combine_1_2 --batch_size 32 --lr 0.01 --epochs 100 --save save/rop_3cls_combine_1_2_DualPathNet_100_T1 &


nohup python -u main.py --arch DualPathNet --resume /home/ubuntu/skin_demo/RoP/models/kaggle/DualPathNet107_downsamping_enhance_DualPathNet107enhance_epoch12_loss0.9639.pkl --data rop_3cls_combine_1_2 --batch_size 32 --lr 0.01 --epochs 100 --save save/rop_3cls_combine_1_2_DualPathNet_100_T2 &


# rop_2cls
python main.py --arch DualPathNet --data rop_2cls --batch_size 32 --epochs 100 --save save/rop_2cls_DualPathNet_100_T1

nohup python -u main.py --arch DualPathNet --data rop_2cls --batch_size 32 --epochs 100 --save save/rop_2cls_DualPathNet_100_T1 &

nohup python -u main.py --arch DualPathNet --pretrain /home/ubuntu/skin_demo/RoP/models/kaggle/DualPathNet107_downsamping_enhance_DualPathNet107enhance_epoch12_loss0.9639.pkl --data rop_2cls --batch_size 32 --epochs 100 --save save/rop_2cls_DualPathNet_100_T1 &



# rop_4cls
python main.py --arch DualPathNet --data rop_4cls --batch_size 32 --epochs 100 --save save/rop_4cls_DualPathNet_100_T1

nohup python -u main.py --arch DualPathNet --data rop_4cls --batch_size 32 --epochs 100 --save save/rop_4cls_DualPathNet_100_T2 &


# rop_2cls_balance

python main.py --arch DualPathNet --data rop_2cls_balance --batch_size 32 --epochs 100 --save save/rop_2cls_balance_DualPathNet_100_T1

nohup python -u main.py --arch DualPathNet --data rop_2cls_balance --batch_size 32 --epochs 100 --save save/rop_2cls_balance_DualPathNet_100_T1 &

nohup python -u main.py --arch DualPathNet --data rop_2cls_balance --batch_size 32 --epochs 100 --lr 0.01 --save save/rop_2cls_balance_DualPathNet_100_T2 &

nohup python -u main.py --arch DualPathNet --data rop_2cls_balance --batch_size 32 --epochs 100 --lr 0.01 --save save/rop_2cls_balance_DualPathNet_100_T3 &

# revised train data set
nohup python -u main.py --arch DualPathNet --data rop_2cls_balance --batch_size 32 --epochs 100 --lr 0.01 --save save/rop_2cls_balance_DualPathNet_100_T4 &

nohup python -u main.py --arch DualPathNet --data rop_2cls_balance --batch_size 32 --epochs 100 --lr 0.01 --save save/rop_2cls_balance_DualPathNet_100_T5 &



#single image inference: save output image with predict and label, 2cls_balanced
--arch DualPathNet --data val_2cls --resume ~/workspace/RoP/master/save/rop_2cls_balance_DualPathNet_100_T4/checkpoint.pth.tar --save save/val_rop_2cls_balance_T4_2

nohup python -u inference_hitmap.py --arch DualPathNet --data val_2cls --resume ~/workspace/RoP/master/save/rop_2cls_balance_DualPathNet_100_T4/checkpoint.pth.tar --save save/val_rop_2cls_balance_T4_2 &

nohup python -u inference_hitmap.py --arch DualPathNet --data val_2cls --resume ~/workspace/RoP/master/save/rop_2cls_balance_DualPathNet_100_T4/checkpoint.pth.tar --save save/val_rop_2cls_balance_T4_3 &

--arch DualPathNet --data val_3cls --resume ~/workspace/RoP/master/save/rop_3cls_combine_1_2_DualPathNet_100_T2/checkpoint.pth.tar --save save/val_rop_3cls_balance_T2



