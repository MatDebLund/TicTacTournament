import keras
from keras import layers
import random
import numpy as np
import json

class IncorrectMoveError(Exception):
    """Exception raised for an incorrect move in the game."""
    pass

def vector_to_position(vector):
    # Convert input to a NumPy array (if it's not already)
    vector = np.array(vector)

    # Check that the vector has exactly 9 elements
    if vector.shape[0] != 9:
        raise ValueError("Input vector must have exactly 9 elements.")

    # Ensure that exactly one element is 1 and the rest are 0s
    if np.count_nonzero(vector == 1) != 1 or np.count_nonzero(vector != 0) != 1:
        raise ValueError("Input vector must be one-hot: exactly one element is 1 and the rest are 0.")

    # Find the index where the 1 is located
    index = int(np.where(vector == 1)[0][0])

    # Convert the 1D index to 2D coordinates in a 3x3 array
    row = index // 3
    col = index % 3

    return (row, col)

class Player:
    def __init__(self, genetics, mutation_rate, mutation_scale):
        # Create a new model with the same architecture
        self.model = keras.Sequential([
            layers.Input(shape=(9,)),
            layers.Dense(9, activation="relu", name="layer1"),
            layers.Dense(9, activation="relu", name="layer2"),
            layers.Dense(9, activation='softmax')
        ])

        if genetics is not None:
            mutated_weights = []
            for weight_array in genetics:
                # Create a mask that determines which elements will be mutated
                mutation_mask = np.random.rand(*weight_array.shape) < mutation_rate

                # Generate random Gaussian noise for mutations
                noise = np.random.normal(loc=0.0, scale=mutation_scale, size=weight_array.shape)

                # Apply mutations: only modify weights where the mask is True
                new_weight_array = weight_array + mutation_mask * noise
                mutated_weights.append(new_weight_array)



            # Set the mutated weights to the new model
            self.model.set_weights(mutated_weights)

    def select_move(self, state):
        move = self.model(state.reshape(1, 9))

        # Find the index of the highest value in the output
        best_index = np.argmax(move)

        # Create a one-hot encoded vector of the same shape as the output
        selected_move = np.zeros_like(move.flatten())
        selected_move[best_index] = 1
        return selected_move


def single_game(player1, player2):
    no_turns = 0
    incorrect_counter = {0:0, 1:0}

    game = Game()
    while no_turns < 10:
        player1_move = player1.select_move(game.board)
        try:
            game.turn(1, player1_move)
        except IncorrectMoveError:
            incorrect_counter[0]+=1

        player2_move = player2.select_move(game.board)

        try:
            game.turn(2, player2_move)
        except IncorrectMoveError:
            incorrect_counter[1]+=1

        if game.evaluate() != 0:
            break

        no_turns += 1

    penalty_factor = 0.1

    outcome = game.evaluate()
    if outcome == 1:
        base_fitness1 = 1.0
        base_fitness2 = 0.0
    elif outcome == 2:
        base_fitness1 = 0.0
        base_fitness2 = 1.0
    else:
        base_fitness1 = 0.5
        base_fitness2 = 0.5

        # Subtract penalty for each illegal move:
    fitness1 = base_fitness1 - penalty_factor * incorrect_counter[0]
    fitness2 = base_fitness2 - penalty_factor * incorrect_counter[1]

    return fitness1, fitness2, incorrect_counter, no_turns, game.board



class Tournament:
    def __init__(self, no_players):
        self.no_players = no_players
        self.all_players = {}
        for i in range(no_players):
            self.all_players[i] = Player(None, None, None)

        self.remaining_ids = list(self.all_players.keys())

    def round(self):
        pairable_ids = self.remaining_ids

        player1_indices = [pairable_ids[i] for i in range(0, len(pairable_ids), 2)]
        player2_indices = [pairable_ids[i] for i in range(1, len(pairable_ids), 2)]

        winning_indices = []
        fitness_counter = 0

        assert len(player1_indices) == len(player2_indices)

        for match_no in range(len(player1_indices)):
            player1_index = player1_indices[match_no]
            player2_index = player2_indices[match_no]

            fitness1, fitness2, incorrect_counter, _, _ = single_game(
                self.all_players[player1_index], self.all_players[player2_index]
            )

            # Use the fitness scores to decide the winner.
            if fitness1 > fitness2:
                winning_indices.append(player1_index)
                fitness_counter += fitness1
                # print(f"Match {match_no}: Player 1 wins (fitness {fitness1:.2f} vs {fitness2:.2f}).")
            elif fitness2 > fitness1:
                winning_indices.append(player2_index)
                fitness_counter += fitness2
                # print(f"Match {match_no}: Player 2 wins (fitness {fitness2:.2f} vs {fitness1:.2f}).")
            else:
                winner = random.choice([player1_index, player2_index])
                winning_indices.append(winner)
                fitness_counter+=fitness1
                # print(f"Match {match_no}: Tie; randomly selected Player {1 if winner == player1_index else 2}.")

        print(f'Fitness: {fitness_counter/len(player1_indices)}')

        self.remaining_ids = winning_indices

        return winning_indices

    def run_tournament(self):
        round_number = 0
        winners = {}
        while len(self.remaining_ids) > 1:
            winners[round_number] = self.round()
            round_number += 1

        return winners

    def reproduce(self, winning_numbers, mutation_rate, mutation_scale):
        # Count wins for each player across all rounds.
        wins_count = {}
        for round_no, winners in winning_numbers.items():
            for player_index in winners:
                wins_count[player_index] = wins_count.get(player_index, 0) + 1

        # give even low performers a base chance (e.g., add 1 to every count).
        # This way, even players with 0 wins in the recorded rounds might get selected occasionally.
        for player_index in wins_count:
            wins_count[player_index] += 1

        # Build a reproduction pool where a player's index appears in proportion to its win count.
        reproduction_pool = []
        for player_index, count in wins_count.items():
            reproduction_pool.extend([player_index] * count)

        # Create a new generation of players.
        new_generation = {}
        for new_index in range(self.no_players):
            # Randomly select a parent from the reproduction pool.
            parent_index = random.choice(reproduction_pool)
            parent = self.all_players[parent_index]
            parent_weights = parent.model.get_weights()

            # Create a new Player with the parent's weights and apply mutation.
            # The Player __init__ handles the mutation when genetics are provided.
            new_generation[new_index] = Player(parent_weights, mutation_rate, mutation_scale)

        self.all_players = new_generation

        self.remaining_ids = list(self.all_players.keys())


class Game:
    def __init__(self):
        self.board = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])

    def row_win(self, player):
        for x in range(len(self.board)):
            win = True

            for y in range(len(self.board)):
                if self.board[x, y] != player:
                    win = False
                    continue

            if win == True:
                return (win)
        return (win)

    def col_win(self, player):
        for x in range(len(self.board)):
            win = True

            for y in range(len(self.board)):
                if self.board[y][x] != player:
                    win = False
                    continue

            if win == True:
                return (win)
        return (win)

    def diag_win(self, player):
        win = True
        y = 0
        for x in range(len(self.board)):
            if self.board[x, x] != player:
                win = False
        if win:
            return win
        win = True
        if win:
            for x in range(len(self.board)):
                y = len(self.board) - 1 - x
                if self.board[x, y] != player:
                    win = False
        return win

    def evaluate(self):
        winner = 0

        for player in [1, 2]:
            if (self.row_win(player) or
                    self.col_win(player) or
                    self.diag_win(player)):
                winner = player

        if np.all(self.board != 0) and winner == 0:
            winner = -1
        return winner

    def turn(self, player, location_vec):
        # Convert the vector to a (row, col) position
        position = vector_to_position(location_vec)
        row, col = position

        # Check if the move is valid (i.e., the cell is empty)
        if self.board[row][col] == 0:
            self.board[row][col] = player
        else:
            raise IncorrectMoveError(f"Invalid move! The cell at {position} is already taken.")


if __name__ == "__main__":
    mutation_rate = 0.01
    mutation_scale = 0.05
    test_tournament = Tournament(64)
    winning_numbers = test_tournament.run_tournament()

    for i in range(500):
        print(i)
        test_tournament.reproduce(winning_numbers, mutation_rate, mutation_scale)
        winning_numbers = test_tournament.run_tournament()


    def save_model_weights_to_json(model, file_path="weights.json"):
        # Get the weights from the model (a list of NumPy arrays)
        weights = model.get_weights()
        # Convert each NumPy array to a Python list so that it's JSON serializable
        weights_serializable = [w.tolist() for w in weights]

        # Write the list of weights to a JSON file
        with open(file_path, "w") as f:
            json.dump(weights_serializable, f, indent=2)

    save_model_weights_to_json(test_tournament.all_players[winning_numbers[5][0]].model,'best')
    save_model_weights_to_json(test_tournament.all_players[winning_numbers[5][1]].model,'second')
