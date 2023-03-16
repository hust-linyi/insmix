set -ex
python train.py --dataroot /home/ylindq/Data/kumar/np --name nuclei_pix2pix --model nuclei --netG deepfill --direction AtoB --lambda_L1 100 --dataset_mode nuclei --norm batch --pool_size 0 --gpu_ids 3
# triplet
# python train.py --dataroot /home/ylindq/Data/kumar/np --name nuclei_pix2pix --model nuclei --netG deepfill --direction AtoB --lambda_L1 100 --dataset_mode nuclei --norm batch --pool_size 0 --gpu_ids 3 --gan_mode triplet
