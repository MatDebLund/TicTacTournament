# play_from_json.py

import json
import numpy as np
from TicTacTournament_main import Player, Game, IncorrectMoveError


def load_model_weights_from_json(file_path):
    """
    Load model weights from a JSON file and convert them to a list of NumPy arrays.

    Args:
        file_path (str): The path to the JSON file containing the weights.

    Returns:
        List[np.ndarray]: The weights as a list of NumPy arrays.
    """
    with open(file_path, "r") as f:
        weights_list = json.load(f)
    # Convert each weight (which is a list) back into a NumPy array
    return [np.array(w) for w in weights_list]


def play_game_display_moves(player1, player2):
    """
    Play a game between two players and print the board after each move.

    Args:
        player1 (Player): The first player.
        player2 (Player): The second player.

    Returns:
        Game: The finished game instance.
    """
    game = Game()
    turn_count = 0
    print("Initial board:")
    print(game.board)

    # Continue until a win/draw is detected or a maximum of 20 turns is reached.
    while game.evaluate() == 0 and turn_count < 20:
        # --- Player 1's move ---
        p1_move = player1.select_move(game.board)
        print(f"\nTurn {turn_count * 2 + 1} - Player 1 selects move:")
        print(p1_move)
        try:
            game.turn(1, p1_move)
        except IncorrectMoveError:
            print("Player 1 made an invalid move. Skipping move.")
        print("Board after Player 1's move:")
        print(game.board)

        # Check if the game ended after Player 1's move.
        if game.evaluate() != 0:
            break

        # --- Player 2's move ---
        p2_move = player2.select_move(game.board)
        print(f"\nTurn {turn_count * 2 + 2} - Player 2 selects move:")
        print(p2_move)
        try:
            game.turn(2, p2_move)
        except IncorrectMoveError:
            print("Player 2 made an invalid move. Skipping move.")
        print("Board after Player 2's move:")
        print(game.board)

        turn_count += 1

    # Print the final board and game result.
    print("\nFinal board:")
    print(game.board)
    result = game.evaluate()
    if result == 1:
        print("Player 1 wins!")
    elif result == 2:
        print("Player 2 wins!")
    elif result == -1:
        print("The game ended in a draw!")
    else:
        print("The game ended with no winner!")

    return game


if __name__ == "__main__":
    # Load the weights from the two JSON files.
    # (Make sure these files exist in the same directory or adjust the paths accordingly.)
    weights_best = load_model_weights_from_json("best")
    weights_second = load_model_weights_from_json("second")

    # Create two players using the loaded weights.
    # Set mutation_rate and mutation_scale to 0 so that the weights remain unchanged.
    player_best = Player(weights_best, mutation_rate=0, mutation_scale=0)
    player_second = Player(weights_second, mutation_rate=0, mutation_scale=0)

    # Let the two players play against each other and display the moves.
    play_game_display_moves(player_best, player_second)
