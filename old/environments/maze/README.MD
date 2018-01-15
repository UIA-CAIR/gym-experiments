# gym-experiments: Maze
This section covers experiments conducted on the maze environment.

1. generator.py: This file generates training data in each of the defined environments (line 8 ish: envs). These are stored as a array of numpy arrays in pikcle version 3
2. trainer.py: Trains on previously generated training data. Stores model for defined timesteps
3. tester.py: Tests the trained model and stores results in results.py
4. results/parser.py: Parses results and generates a chart of the performance for each of the experiments

Following scripts are related to the DQN model:
1. agent.py This defines the training function, act function and inputs LEARNING_RATE, DISCOUNT_RATE and keeps track of loss while training
2. models/*.py These are functions that defiens the RL model. (The neural network)

The RL proceedure itself is almost identical as in other environments, so for further assistance lookup these :)
