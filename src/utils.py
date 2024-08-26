from bisect import bisect_left
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np
from chess import Board, Move
from chess import pgn


def board_to_matrix(board: Board) -> np.array:
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board from which square a pice can be moved
    # 14th board for legal target squares
    matrix = np.zeros((14, 8, 8), dtype=np.uint8)
    piece_map = board.piece_map()

    # Populate first 12 8x8 board with the board state
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves (13th and 14th dim)
    legal_moves = board.legal_moves
    for move in legal_moves:
        from_square = move.from_square
        row_from, col_from = divmod(from_square, 8)
        matrix[12, row_from, col_from] = 1

        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[13, row_to, col_to] = 1

    return matrix


def move_to_vector(move: Move) -> np.array:
    vector = np.zeros(128, dtype=np.uint8)
    vector[move.from_square] = 1
    vector[(64 + move.to_square)] = 1

    return vector


def get_num_matches_from_pgn(file_path: Path) -> int:
    num = 0
    with open(file_path, "r") as pgn_file:
        while pgn.read_headers(pgn_file):
            num += 1
    return num


def load_games_from_pgn(file_path: Path) -> Iterator[Tuple[pgn.Game, int]]:
    with open(file_path, "r") as pgn_file:
        offset = pgn_file.tell()
        while game := pgn.read_game(pgn_file):
            yield game, offset
            offset = pgn_file.tell()


def load_headers_from_pgn(file_path: Path) -> Iterator[Tuple[pgn.Headers, int]]:
    with open(file_path, "r") as pgn_file:
        offset = pgn_file.tell()
        while header := pgn.read_headers(pgn_file):
            yield header, offset
            offset = pgn_file.tell()


def take_closest_lower(myList, myNumber):
    """
    Assumes myList is sorted. Returns lower closest value to myNumber.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after == myNumber:
        return after
    return before


def moves_from_game(game: pgn.Game) -> Iterator[Tuple[Move, Board, np.array, np.array]]:
    board = game.board()
    for move in game.mainline_moves():
        yield move, board, board_to_matrix(board), move_to_vector(move)
        board.push(move)


def get_move_from_game(
    game: pgn.Game, move_number: int
) -> Tuple[Move, Board, np.array, np.array]:
    board = game.board()
    for move_id, move in enumerate(game.mainline_moves()):
        if move_id < move_number:
            board.push(move)
        else:
            return move, board, board_to_matrix(board), move_to_vector(move)
