import sys
sys.path.append('/cluster/home/it_stu83/2048-api')

from game2048.game import Game
from game2048.agents import MyAgent


game = Game()
agent = MyAgent(game)
agent.train()


