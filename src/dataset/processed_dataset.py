import json
from pathlib import Path
from typing import Callable, List, Tuple
from chess import Board, Move, pgn
import psutil
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import operator
import functools

from src.utils import (
    get_num_matches_from_pgn,
    load_games_from_pgn,
    moves_from_game,
)


def no_filter_game(game: pgn.Game) -> bool:
    return True

def no_filter_move(move: Move, board: Board) -> bool:
    return True


class ProcessedChessDataset(Dataset):

    def __init__(self,
                 folder_path: Path,
                 meta_filename: str = "metadata.json",
                 moves_filename: str = "processed_moves.bin",
                 filter_game: Callable[[pgn.Game], bool] = no_filter_game,
                 filter_move: Callable[[Move, Board], bool] = no_filter_move) -> None:
        super().__init__()

        self.num_moves = 0
        self.loaded_data = None

        cache_path = folder_path / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        
        self.meta_info_path = cache_path / meta_filename
        self.moves_path = cache_path / moves_filename
        if self.meta_info_path.exists():
            with open(self.meta_info_path) as fp:
                metadata = json.load(fp)
            self.num_moves = metadata["num_moves"]
            self.X_shape = tuple(metadata["X_shape"])
            self.y_shape = tuple(metadata["y_shape"])
            self.X_nbytes = metadata["X_nbytes"]
            self.y_nbytes = metadata["y_nbytes"]
            self._X_size = functools.reduce(operator.mul, self.X_shape, 1)
            self._y_size = functools.reduce(operator.mul, self.y_shape, 1)
            self._sample_size = self._X_size + self._y_size

            self.loaded_data = self.load_in_memory()
            return

        with open(self.moves_path, 'wb') as moves_fp:
            for file in tqdm(
                folder_path.rglob("*.pgn"),
                desc="Files Processed",
                total=len(list(folder_path.rglob("*.pgn"))),
                position=0,
            ):
                for game, game_offset in tqdm(
                    load_games_from_pgn(file),
                    desc=str(file),
                    total=get_num_matches_from_pgn(file),
                    position=1,
                    leave=False,
                ):
                    if not filter_game(game):
                        continue
                    for move, board, X, y in moves_from_game(game):
                        if not filter_move(move, board):
                            continue
                        self.num_moves += 1
                        moves_fp.write(X.tobytes())
                        moves_fp.write(y.tobytes())

        self.X_shape = X.shape
        self.y_shape = y.shape
        self.X_nbytes = len(X.tobytes())
        self.y_nbytes = len(y.tobytes())
        metadata = {
            "num_moves": self.num_moves,
            "X_shape": self.X_shape,
            "y_shape": self.y_shape,
            "X_nbytes": self.X_nbytes,
            "y_nbytes": self.y_nbytes
        }

        with open(self.meta_info_path, 'w') as fp:
            json.dump(metadata, fp)

        self.loaded_data = self.load_in_memory()


    def __len__(self) -> int:
        return self.num_moves

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:

        if self.loaded_data is not None:
            return self.decode_tensor(idx)

        nbytes = self.X_nbytes + self.y_nbytes

        with open(self.moves_path, "rb") as moves_fp:
            moves_fp.seek(nbytes * idx)
            buffer = moves_fp.read(nbytes)

        X = torch.frombuffer(buffer[:self.X_nbytes], torch.int8).view(self.X_shape).float()
        y = torch.frombuffer(buffer[self.X_nbytes:], torch.int8).view(self.y_shape).float()
        y = (y[:64], y[64:])

        return X, y
    
    def decode_tensor(self, index):
        start = index * self._sample_size
        end = start + self._sample_size
        sample: torch.Tensor = self.loaded_data[start:end]

        X = sample[:-self._y_size].view(self.X_shape).float()
        y = sample[-self._y_size:].view(self.y_shape).float()

        return X, (y[:64], y[64:])
    
    def load_in_memory(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        available_memory = psutil.virtual_memory().total
        memory = available_memory - (available_memory // 8)

        if self.moves_path.stat().st_size >= memory:
            return None
    
        with open(self.moves_path, 'rb') as moves_fp:
            buffer = moves_fp.read()
            ds = torch.frombuffer(buffer, dtype=torch.int8)
        return ds