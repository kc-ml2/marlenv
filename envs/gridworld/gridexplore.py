from envs.gridworld.gridworld import GridWorld 
from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from gym.envs.classic_control import rendering

import numpy as np
import random
import math
from gym import spaces
from PIL import ImageColor
import copy

CELL_SIZE = 30

PRE_IDS = {
    'agent': ['3','4','5','6'],
    'wall': '2',
    'empty': '0',
    'visited':'1'
}

WALL_COLOR = 'black'
VISITED_COLOR = 'grey'
AGENT_COLOR = ['green', 'blue', 'red', 'yellow']

class Cell:
    UNVISITED = 0
    VISITED = 1
    WALL = 2
    AGENTS = [3, 4, 5, 6]

class Move:
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3
	STAY = 3 
	STAY = 4 

class Rewards:
	TIMEPENALTY = -1
	EXPLORATION_BONUS = 2
	WIN = 50 

class Agent:

	def __init__(self, posx, posy, idx, sight=2):
		
		self.idx = int(idx)
		self.x = posx
		self.y = posy 
		self.position = (self.x, self.y)
		
		self.sight = sight

	def __str__(self):
		return "x: " + str(self.x) +  " y: "+ str(self.y)+ " agent idx: " + str(self.idx)

	def position_yx(self):
		return self.y, self.x

	def move(self, MOVE, grid):
		new_x, new_y = self.position[0], self.position[1]
		org_x, org_y = self.position[0], self.position[1]

		if MOVE==Move.UP and self.position[1] != 1:
			new_y -= 1 
		elif MOVE == Move.DOWN and self.position[1] != len(grid)-1:
			new_y += 1 	
		if MOVE==Move.LEFT and self.position[0] != 1:
			new_x -= 1 
		elif MOVE == Move.RIGHT and self.position[0] != len(grid)-1:
			new_x += 1 	

		#check if wall exist there
		if grid[new_y][new_x] != 2 and grid[new_y][new_x] not in Cell.AGENTS:
			self.makeMove(new_x, new_y)
			grid[org_y][org_x] = 1
			return new_x, new_y
		else:
			# print(self.idx, end=": 	")
			# print(grid[new_y][new_x], " ", grid[org_y][org_x], " ", org_x, org_y)
			# print(grid)
			return self.position[0], self.position[1]

	def makeMove(self, new_x, new_y):
		self.x = new_x
		self.y = new_y
		self.position = (self.x, self.y)


class GridExplore(GridWorld):

	def __init__(self, size, n_agents=1, full_observable=False, dist_penalty=5):

		self.size = size
		self.grid = [ [0 for _ in range(size)] for _ in range(size)]
		self._grid_shape=[size,size]

		self.agentList=[]
		self.time= 0
		self.n_agents=n_agents
		self.dist_penalty = dist_penalty 

		self.__setwall()
		
		self.init_agent_pos= {}
		self.viewer = None

		self.observation_space = MultiAgentObservationSpace([spaces.Box(low=0,high=6,shape=(4, self.size, self.size)) for _ in range(self.n_agents)])
	
		self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])

		# self.render()

	def reset(self):

		self.__init_full_obs()
		self.agentList=[]
		self.grid = [ [0 for _ in range(self.size)] for _ in range(self.size)]
		self._grid_shape=[self.size,self.size]
		self.__setwall()

		for i in range(self.n_agents):			
			x, y = self.__getEmptyCell()
			self.__putAgent(x,y, Cell.AGENTS[i])
			self.init_agent_pos[i] = (x, y)
			self.agent_pos[i] = self.agentList[i].position

			#initial surroundings
			self.searchArea(self.agentList[i])
		self.dones = np.zeros(self.n_agents, dtype=bool)

		return self.observation()



	def step(self, actions):
		self.time+=1
		actiondict={}
		for i in range(len(actions)):
			actiondict[i] = actions[i]

		nextMoveSet=set()
		nextPosSet=set()

		for i, v in actiondict.items():
			# self.agent_prev_pos[i] = self.agentList[i].position
			new_x, new_y = self.agentList[i].move(v, self.grid)

			pos = (new_x,new_y)
			if pos in nextPosSet:
				if v < 2 :
					new_x,new_y = self.agentList[i].move(v+2, self.grid)
					pos = (new_x,new_y)
				elif v == 2 or v == 3:	
					new_x,new_y = self.agentList[i].move(v-2, self.grid)
					pos = (new_x,new_y)
				else:
					print("action 4 is staying therefore there shouldn't be conflicts")	
			
			self.agent_pos[i] = self.agentList[i].position_yx()

			#insert to nextMoveSet to check conflict
			nextPosSet.add(pos)
			nextMoveSet.add((pos, self.agentList[i].idx))

		#New position set 
		for pos ,idx in nextMoveSet:
			self.grid[pos[1]][pos[0]]= idx 

		#Initialze reward and done list
		rewards = np.zeros(self.n_agents)
		dones = [False,False,False,False]
		
		#TIME PENALTY + EXPLORATION REWARD
		for i in range(self.n_agents):

			rewards[i] += (self.searchArea(self.agentList[i]) + Rewards.TIMEPENALTY)

		rewards = rewards + self.distancePenalty(self.agentList)
		# print(rewards)
		if not any(0  in i for i in self.grid):
			dones = [True,True,True,True]
			rewards = rewards + Rewards.WIN

		# print(self.time)
		# if self.time == 1:
		# 	dones = [True,True,True,True]

		return self.observation(), rewards, dones, self.grid 


	def render(self):
		for i in self.grid:
			print(i)
		print("")


	def __setwall(self):
		for i in range(self.size):
			self.grid[0][i] = 2 
			self.grid[self.size-1][i] = 2 
			self.grid[i][0] = 2
			self.grid[i][self.size-1]= 2

	def __getEmptyCell(self):
		
		x = random.choice(range(self.size-1))
		y = random.choice(range(self.size-1))

		if self.grid[y][x] == 0:
			return x, y 
		else:
			return self.__getEmptyCell()

	def __putAgent(self, x, y, agentnum):
		assert self.grid[y][x] == 0 

		self.grid[y][x] = agentnum
		self.agentList.append(Agent(x,y,agentnum))

	def __moveAgent(self,x,y,agentnum):
		assert self.grid[y][x] == 0 or self.grid[y][x] == 1
		self.grid[y][x] = agentnum

	def getObservation(self, agent, obs_size):
		(agentx , agenty) = agent.position
		h , w = obs_size, obs_size 
		x1 = int(agentx - w//2 )
		x2 = int(agentx + w//2 )
		y1 = int(agenty - h//2 )
		y2 = int(agenty + h//2 )
		arr = np.array(self.grid)
		arr = arr[max(y1,0):y2, max(0, x1):x2]

		if y1 < 0:
		    arr = np.pad(arr, ((-y1, 0), (0, 0),), mode="constant")
		elif y2 > self.size:
		    arr = np.pad(arr, ((0, y2 - self.size), (0, 0),),mode="constant")
		if x1 < 0:
		    arr = np.pad(arr, ((0, 0), (-x1, 0)),mode="constant")
		elif x2 > self.size:
		    arr = np.pad(arr, ((0, 0), (0,x2 - self.size)),mode="constant")

		return arr
			

	def searchArea(self, agent):
		reward =0
		sight = agent.sight 
		sight = sight // 2 
		x,y = agent.position
		
		x -= sight  
		y -= sight 

		#Give bonus points for exploration
		for i in range(y, y+2*sight+1):
			for j in range(x, x+2*sight+1):
				# print(self.grid[i][j], end="")
				if self.grid[i][j] == 0:
					self.grid[i][j] = 1 
					# print(i,j, end=" ")
					reward += Rewards.EXPLORATION_BONUS
			# print("")


		return float(reward) 

	def distance(self, pos1, pos2):
		return math.sqrt( (pos2.y - pos1.y)**2 + (pos2.x-pos1.x)**2 ) 


	def isNear(self, agent1, agent2, distance):
		if self.distance(agent2, agent1) < distance:
			return True
		else:
			return False

	def distancePenalty(self, agentlist, distance=2):
		rewards = np.zeros(len(agentlist))
		if len(agentlist) > 1 :

			for i in range(len(agentlist)):
				for j in range(i+1, len(agentlist)):
					if self.isNear(agentlist[i], agentlist[j], distance):
						rewards[i] -= 1
						rewards[j] -= 1

		return rewards

 
	def howNear(self, agent1, agent2, distance):
		return self.distance(agent2, agent1)


	def observation(self):

		statearray= []
		# print(self.observation_space) 
		for i in self.agentList:
			

			state = np.zeros(self.observation_space[0].shape)

			agents = np.isin(self.grid, Cell.AGENTS ).astype(np.float32)	
			agenti = np.isin(self.grid, i.idx ).astype(np.float32)			
			visited = np.isin(self.grid, Cell.VISITED).astype(np.float32) 

			#add agent's position as visited
			visited = visited + agents
			wall = np.isin(self.grid, Cell.WALL).astype(np.float32)

			#exclude self from agents pos list
			agents = agents - agenti

			for idx in range(self.n_agents):
				state[0] = agenti
				state[1] = agents
				state[2] = visited
				state[3] = wall

			statearray.append(state)

		return statearray
    

	def __init_full_obs(self):
	    self.agent_pos = copy.copy(self.init_agent_pos)
	    # self.agent_prev_pos = copy.copy(self.init_agent_pos)
	    self._full_obs = self.grid
	    self.__draw_base_img()


	def __draw_base_img(self):
		self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
		for row in range(self._grid_shape[0]):
			for col in range(self._grid_shape[1]):
				if 2 == self.grid[col][row]:
					fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.05)
				elif PRE_IDS['visited'] is self.grid[row][col]:
					fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=VISITED_COLOR, margin=0.05)
				elif PRE_IDS['agent'] is self.grid[row][col]:
					fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=AGENT_COLOR, margin=0.05)



	def render_graphic(self, mode='human'):
		img = copy.copy(self._base_img)

		#Draw Agents with Color and number 
		for agent_i in range(self.n_agents):
			draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR[agent_i])
			write_cell_text(img, text=str(agent_i + 3), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
							fill='white', margin=0.4)

		#Draw explored/visited cells 
		for row in range(self._grid_shape[0]):
			for col in range(self._grid_shape[1]):
				if self.grid[row][col] == 1:
					fill_cell(self._base_img, (row, col), cell_size=CELL_SIZE, fill=VISITED_COLOR, margin=0.05)

		# return img
		img = np.asarray(img)
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen

