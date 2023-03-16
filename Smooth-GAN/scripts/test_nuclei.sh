set -ex
# python test.py --dataroot /home/ylindq/Data/kumar/np --name nuclei_pix2pix --model nuclei --netG deepfill --direction AtoB --dataset_mode nuclei --norm batch --gpu_ids 3
# generate images
python test.py --dataroot /home/ylindq/Data/kumar/np --name nuclei_pix2pix --model nuclei --netG deepfill --direction AtoB --dataset_mode nucleitest --norm batch --gpu_ids 3 --num_test 1000 
