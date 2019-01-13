import sys
sys.path.append('/cluster/home/it_stu83/tr2048_test')

from game2048.game import Game
from game2048.agents import MyAgent


game = Game()
agent = MyAgent(game)
agent.train()


