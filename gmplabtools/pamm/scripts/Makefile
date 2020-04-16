.PHONY: transform cluster predict
all: transform cluster predict

transform:
	python transform.py -c config.toml
	
cluster:
	python cluster.py -c config.toml

predict:
	python predict.py -c config.toml

clean:
	rm -f *.png *.grid *.weights *.pca *.txt *.pamm

