from pathlib import Path
from chess import WHITE, Board, pgn, Move

from .processed_dataset import ProcessedChessDataset


def filter_game(game: pgn.Game) -> bool:
    header = game.headers
    
    result = header.get("Result", None)
    if result != "1-0":
        return False

    # event = header.get("Event", None)
    # if "Blitz" in event:
    #     return False

    white_elo = int(header.get("WhiteElo", 0))
    if white_elo < 2700:
        return False
    return True

def filter_move(move: Move, board: Board) -> bool:
    if board.turn == WHITE:
        return True
    return False


class WhiteFullChessDataset(ProcessedChessDataset):
    def __init__(self, 
                 folder_path: Path) -> None:
        super().__init__(folder_path=folder_path,
                         meta_filename="white_full_dataset.json",
                         moves_filename="white_full_dataset.bin",
                         filter_game=filter_game,
                         filter_move=filter_move)