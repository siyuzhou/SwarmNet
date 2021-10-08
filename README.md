# SwarmNet
A Graph Neural Network based model for swarm motion prediction and control

For training,
```python
python run_swarwmnet.py --data-dir path/to/training/data --log-dir path/to/log/dir --config path/to/config/file --pred-steps <prediction_horizon> --train --epochs <num_epochs>
```

For evaluation or test,
```python
python run_swarwmnet.py --data-dir path/to/training/data --log-dir path/to/log/dir --config path/to/config/file --pred-steps <prediction_horizon> --eval/--test
```

See `run_swarmnet.py` for all arguments.
