.PHONY: transform cluster predict
all: transform cluster predict

transform:
	python transform.py -c config.json
	
cluster:
	python cluster.py -c config.json

predict:
	python predict.py -c config.json

clean:
	rm -f *.png *.grid *.weights *.pca *.txt *.pamm

