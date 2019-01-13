import numpy as np


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction




import torch
import torch.nn as nn
import numpy as np
from game2048.model import model
from game2048.grid_ohe import grid_ohe
from game2048.game import Game
import os
current_path = os.path.dirname(__file__)

class MyAgent(Agent):

    def __init__(self, game, display=None):
        if game.size !=4:
            raise ValueError(
                "`%s` can only work with game of 'size' 4." % self.__class__.__name__
            )
        super().__init__(game, display)

        self.game = game
        self.display = display



        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model().to(self.device)
        self.model.load_state_dict(torch.load(current_path + '/model.pkl', map_location=self.device))
        # self.model_256 = model().to(self.device)
        # self.model_256.load_state_dict(torch.load(current_path + '/model_256.pkl', map_location=self.device))
        # self.model_512 = model().to(self.device)
        # self.model_512.load_state_dict(torch.load(current_path + '/model_512.pkl', map_location=self.device))
        # self.model_1024 = model().to(self.device)
        # self.model_1024.load_state_dict(torch.load(current_path + '/model_1024.pkl', map_location=self.device))
        # self.model_t = model().to(self.device)
        # self.model_t.load_state_dict(torch.load(current_path + '/model_t.pkl', map_location=self.device))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)

        #self.obj_agent = ExpectiMaxAgent(self.game)
        from .expectimax import board_to_move
        self.obj_func = board_to_move


    def train(self):

        iter = 0
        while True:

            if self.game.end:
                print('final: %f'%self.game.score)
                self.new()

            ohe_board = grid_ohe(self.game.board)
            board_ohe = torch.Tensor(ohe_board.reshape(1, *ohe_board.shape)).to(self.device).float()
            labels = torch.Tensor([self.obj_func(self.game.board)]).to(self.device).long()

            #if self.game.score <= 512:


            self.optimizer.zero_grad()
            self.criterion(self.model.forward(board_ohe), labels).backward()
            self.optimizer.step()

            # elif self.game.score > 128 and self.game.score <= 256:
            #     param = list(self.model_256.parameters())
            #     optimizer = torch.optim.SGD(param, lr=0.01, momentum=0.9)
            #     optimizer.zero_grad()
            #     criterion(self.model_256(board), labels).backward()
            #     optimizer.step()
            #
            # elif self.game.score > 256 and self.game.score < 512:
            #     param = list(self.model_512.parameters())
            #     optimizer = torch.optim.SGD(param, lr=0.01, momentum=0.9)
            #     optimizer.zero_grad()
            #     criterion(self.model_512(board), labels).backward()
            #     optimizer.step()
            #
            # elif self.game.score > 512 and self.game.score <= 1024:
            #     param = list(self.model_1024.parameters())
            #     optimizer = torch.optim.SGD(param, lr=0.01, momentum=0.9)
            #     optimizer.zero_grad()
            #     criterion(self.model_1024(board), labels).backward()
            #     optimizer.step()
            #
            # else:
            #     param = list(self.model_t.parameters())
            #     optimizer = torch.optim.SGD(param, lr=0.01, momentum=0.9)
            #     optimizer.zero_grad()
            #     criterion(self.model_t(board), labels).backward()
            #     optimizer.step()
            #direction = self.step()

            # if np.random.rand() >= 0.5:
            #self.game.move(self.obj_func(self.game.board))
            # else:
            self.game.move(self.step())

            iter = iter + 1
            if iter == 10000:
                torch.save(self.model.state_dict(), 'model.pkl')
                # torch.save(self.model_256.state_dict(), 'model_256.pkl')
                # torch.save(self.model_512.state_dict(), 'model_512.pkl')
                # torch.save(self.model_1024.state_dict(), 'model_1024.pkl')
                # torch.save(self.model_t.state_dict(), 'model_t.pkl')

                iter = 0


    def step(self):
        # import time
        # t1 = time.clock()

        ohe_board = grid_ohe(self.game.board)
        board_ohe = torch.Tensor(ohe_board.reshape(1, *ohe_board.shape)).to(self.device).float()

        #if self.game.score <= 128:
        # out = self.model.forward(board_ohe)
        # print(out)
        direction = self.model.forward(board_ohe).max(1, keepdim=True)[1]

        # elif self.game.score > 128 and self.game.score <= 256:
        #     direction = self.model_256.forward(board_ohe).max(1, keepdim=True)[1]
        #
        # elif self.game.score > 256 and self.game.score < 512:
        #     direction = self.model_512.forward(board_ohe).max(1, keepdim=True)[1]
        #
        # elif self.game.score > 512 and self.game.score <= 1024:
        #     direction = self.model_1024.forward(board_ohe).max(1, keepdim=True)[1]
        #
        # else:
        #     direction = self.model_t.forward(board_ohe).max(1, keepdim=True)[1]
        #direction = int(direction)
        # t2 = time.clock()-t1
        # print(t2)

        return direction

    def new(self):
        self.game = Game()
        #self.obj_agent = ExpectiMaxAgent(self.game)
