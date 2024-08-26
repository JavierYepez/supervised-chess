from pathlib import Path
import pickle
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from chess import pgn
from tqdm.auto import tqdm

from src.utils import (get_move_from_game, 
                   get_num_matches_from_pgn, 
                   load_games_from_pgn, 
                   take_closest_lower)


def save_meta(pickle_path: Path, data):
    with open(pickle_path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_meta(pickle_path: Path):
    if not pickle_path.exists():
        return None
    with open(pickle_path, 'rb') as fp:
        return pickle.load(fp)

class LazyChessDataset(Dataset):

    def __init__(self, folder_path: Path) -> None:
        super().__init__()

        self.num_matches = 0
        self.num_moves = 0
        self.moves_keys = []
        self.matches = {}

        matches_pkl = folder_path / 'cache' / 'matches.pickle'
        data = load_meta(matches_pkl)
        if data is not None:
            self.matches = data[0]
            self.num_moves = data[1]
            self.moves_keys = list(self.matches.keys())
            self.num_matches = len(self.moves_keys)
            return

        with tqdm(desc="Files Processed", 
                      total=len(list(folder_path.rglob('*.pgn'))),
                      position=0) as bar:
            for file in folder_path.rglob('*.pgn'):
                with tqdm(desc=str(file), 
                        total=get_num_matches_from_pgn(file),
                        position=1,
                        leave=False) as bar2:
                    for game, game_offset in load_games_from_pgn(file):
                        moves = len(list(game.mainline_moves()))

                        idx = self.num_moves - 1
                        if self.num_moves < 1:
                            idx = 0

                        self.matches[idx] = {
                            "path": file,
                            "offset": game_offset
                        }

                        self.num_moves += moves
                        bar2.update(1)
                bar.update(1)
                
        save_meta(matches_pkl, (self.matches, self.num_moves))
        self.moves_keys = self.matches.keys()
        self.num_matches = len(self.moves_keys)
                

    def __len__(self) -> int:
        return self.num_moves

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        
        match_id = take_closest_lower(self.moves_keys, idx)

        match_loc = self.matches[match_id]

        with open(match_loc["path"], 'r') as pgn_file:
            pgn_file.seek(match_loc["offset"])
            game = pgn.read_game(pgn_file)

        move_number = idx - match_id
        _, _, X, y = get_move_from_game(game, move_number)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

        return X, y