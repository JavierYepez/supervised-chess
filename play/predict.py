from pathlib import Path

import chess
import torch
from chess import Board, Move


def load_model(model_path: Path) -> torch.nn.Module:
    model = torch.jit.load(model_path)
    model.to("cpu")
    model.eval()
    return model


def board_to_matrix(board: Board) -> torch.Tensor:
    matrix = torch.zeros(14, 8, 8, dtype=torch.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    legal_moves = board.legal_moves
    for move in legal_moves:
        from_square = move.from_square
        row_from, col_from = divmod(from_square, 8)
        matrix[12, row_from, col_from] = 1

        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[13, row_to, col_to] = 1

    matrix.to("cpu")
    return matrix.unsqueeze(0)


def decode_output(output_tensor: torch.Tensor, board: Board) -> Move:
    tensor = output_tensor.squeeze(0)

    from_square_list = torch.argsort(torch.softmax(tensor[:64], dim=0), descending=True)
    to_square_list = torch.argsort(torch.softmax(tensor[64:], dim=0), descending=True)

    for from_square in from_square_list:
        piece = board.piece_at(from_square)
        if piece is None or piece.color != board.turn:
            continue
        for to_square in to_square_list:
            try:
                return board.find_move(from_square, to_square)
            except chess.IllegalMoveError as ex:
                pass

    return None


def check_one_move_checkmate(board: Board):
    checkmate_board = Board(board.fen())
    for move in board.legal_moves:
        checkmate_board.push(move)

        if checkmate_board.is_checkmate():
            return checkmate_board
        checkmate_board.pop()


def predict_next_move(fen_board: str, model: torch.nn.Module) -> str:

    board = Board(fen=fen_board)
    if check_mate_board := check_one_move_checkmate(board):
        return check_mate_board.fen()

    input_tensor = board_to_matrix(board)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    move = decode_output(output_tensor, board)
    board.push(move)

    return board.fen()
