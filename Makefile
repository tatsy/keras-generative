
test:
	python train.py --model=vae --dataset=mnist --zdims=128 --epoch=1 --testmode
	python train.py --model=dcgan --dataset=mnist --zdims=128 --epoch=1 --testmode
