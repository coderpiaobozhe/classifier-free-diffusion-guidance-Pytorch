.PHONY : train
train:
	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=gpu train.py
.PHONY : sample
samplepict:
	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=gpu sample.py
.PHONY : samplenpz
samplenpz:
	CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=gpu sample.py --fid True
.PHONY : clean
clean:
	rm -rf __pycache__
	rm -rf model/*
	rm -rf sample/*
