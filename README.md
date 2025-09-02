# TicTacTournament

This project trains a neural network to play tic-tac-toe using a simple tournament. Random policy networks play head-to-head. Winners are more likely to pass their weights to the next generation. Small random mutations are applied to explore the weight space. Over time this should move the population toward policies that avoid illegal moves and win more often.

The approach is intentionally minimal. It does not use backpropagation or gradients, so learning is slow and noisy. In short, it doesn't work very well, but was fun to make.

## How it works

### Model and move selection

* Input: a 9-element vector representing the board flattened to 1D.
* Network: a small Keras MLP with two hidden layers of size 9 and a 9-way softmax output.
* Output: a length-9 probability vector for the next move.
* Action choice: greedy argmax. If the argmax points to an occupied square, the move is illegal.

### Game rules

* Standard 3x3 tic-tac-toe, players 1 and 2.
* An illegal move raises `IncorrectMoveError`. The board does not change and a penalty is recorded.

### Fitness

* Win: 1.0
* Draw: 0.5
* Loss: 0.0
* Penalty: subtract 0.1 for each illegal move committed in the game.

### Tournament and reproduction

* Players are paired each round. Winners advance. Ties are resolved randomly.
* After a tournament, parents are sampled with probability proportional to their round wins, with a small base chance for everyone.
* Offspring copy parent weights and apply element-wise Gaussian noise according to:

  * `mutation_rate` for the mask
  * `mutation_scale` for the noise magnitude

### Outputs

* The script prints out the average fitness of each round and the current tournament being played.
* Optionally, a json dump of the weights can be output

## Requirements

* See requirements.txt

## Quick start

1. Install requirements, preferrably in a virtual environment, in python 3.12.6:
   ```bash
   pip install -r requirements.txt
   pip install notebook
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `TicTacTournament.ipynb`.
4. Run cells in order.

## Notes on performance and limitations

This method is a pure evolutionary search without gradients. Typical issues:

* Slow learning
* Illegal move penalties can dominate early, especially with greedy selection and no action masking.
* Convergence can stall in local optima or collapse due to low diversity.

## Reproducibility

For consistent runs, seed both `random` and `numpy.random` at the start of the program.
