.PHONY : train
train:
	python train.py
.PHONY : sample
sample:
	python sample.py
.PHONE : clean
clean:
	rm -rf __pycache__
	rm -rf model/*
	rm -rf sample/*
	rm -rf cifar-10-batches-py