.PHONY: setup test train-all clean

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest backprop/tests pretrain/tests \
	    finetune/tests visiontx/tests \
	    infer/tests evaluate/tests -v

train-all:
	python -m backprop.train
	python -m pretrain.train
	python -m finetune.sft
	python -m finetune.rlhf
	python -m visiontx.train
	python -m infer.benchmark
	python -m evaluate.evaluate

clean:
	rm -rf outputs/*/checkpoints outputs/*/experiment_log.jsonl
	find . -type d -name __pycache__ -exec rm -rf {} +
