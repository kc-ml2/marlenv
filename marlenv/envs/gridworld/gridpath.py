from envs.gridworld.gridworld import GridWorld 
from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

import random
import numpy as np
from gym import spaces
from PIL import ImageColor
import copy

CELL_SIZE = 30

PRE_IDS = {
    'agent': ['3','4','5','6'],
    'wall': '2',
    'empty': '0',
    'destinations':'1'
}

WALL_COLOR = 'black'
DESTINATION_COLOR = 'grey'
AGENT_COLOR = ['green', 'blue', 'red', 'yellow']

class Cell:
    UNVISITED = 0
    DESTINATIONS = 1
    WALL = 2
    AGENTS = [3, 4, 5, 6]

class Move:
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3
	STAY = 4 

class Rewards:
	TIMEPENALTY = -0.1
	WIN = 10 
	CONFLICTPENALTY = -0.5

class Agent:

	def __init__(self, posx, posy, idx, sight=2):
		
		self.idx = int(idx)
		self.x = posx
		self.y = posy 
		self.position = (self.x, self.y)
		
		self.sight = sight

	def __str__(self):
		return "x: " + str(self.x) +  " y: "+ str(self.y)+ " agent idx: " + str(self.idx)

			
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
		if grid[new_y][new_x] != 2 :
			# and grid[new_y][new_x] not in Cell.AGENTS:
			self.makeMove(new_x, new_y)
			grid[org_y][org_x] = 0
			return new_x, new_y
		else:
			return self.position[0], self.position[1]

	def makeMove(self, new_x, new_y):
		self.x = new_x
		self.y = new_y
		self.position = (self.x, self.y)


class GridPath(GridWorld):

	def __init__(self, size, n_agents=4, full_observable=False, dist_penalty=5):

		self.size = size
		#initialize with WALL
		self.grid = [ [Cell.WALL for _ in range(self.size)] for _ in range(self.size)]
		self._grid_shape=[self.size,self.size]

		self.agentList=[]
		self.time= 0
		self.n_agents=n_agents
		self.dist_penalty = dist_penalty 

		self.init_agent_pos= {}
		self.viewer = None

		self.observation_space = MultiAgentObservationSpace([spaces.Box(low=0,high=6,shape=(4, self.size, self.size)) for _ in range(self.n_agents)])
	
		self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])

	def render(self):
		for i in self.grid:
			print(i)
		print("")
	
	def __setpath(self):
		for i in range(1, self.size-1):
			self.grid[self.size-2][i] = 0
			self.grid[self.size-3][i] = 0
			self.grid[1][i] =0
			self.grid[2][i] =0

			#Path generation
			self.grid[i][self.size//2] = 0

	def __setDestination(self):
		self.grid[1][1] = self.grid[1][self.size-2] = 1
		self.grid[2][1] = self.grid[2][self.size-2] = 1
		self.destinations = [(1,1), (1, self.size-2), (2,1), (2,self.size-2)]

	#set starting position for 4 players
	def __setStartingPosition(self):
		assert self.n_agents == 4
		
		self.starting_positions = [(1, self.size-2), (1, self.size-3), (self.size-2, self.size-2), (self.size-2, self.size-3)]

		self.grid[self.size-2][1] = 3
		self.grid[self.size-3][1] = 4
		self.grid[self.size-2][self.size-2] = 5
		self.grid[self.size-3][self.size-2] = 6

		a = Agent(1, self.size-2, 3)
		b = Agent(1, self.size-3, 4)
		c = Agent(self.size-2, self.size-2, 5)
		d = Agent(self.size-2, self.size-3, 6)
		self.agentList.extend([a,b,c,d])

		for index, value in enumerate(Cell.AGENTS):
			self.init_agent_pos[index] =  self.starting_positions[index][1], self.starting_positions[index][0]



	def reset(self):

		self.grid = [ [Cell.WALL for _ in range(self.size)] for _ in range(self.size)]
		self.__setpath()
		self.__setDestination()
		self.__setStartingPosition()
		self.__init_full_obs()

		self.dones = np.zeros(self.n_agents, dtype=bool)
		
		return self.observation()
		
	def step(self, actions):
		assert len(actions) == self.n_agents

		#Get previous locations and initializations
		prevloc = [i.position for i in self.agentList]
		nextloc = []
		rewards=np.zeros(4)
		self.penalty = np.zeros(self.n_agents, dtype=float)

		# Populate action dictionary and candidate next locations
		actiondict = {}
		for i in range(len(actions)):
			if not self.dones[i]:
				actiondict[i] = actions[i]
				newx, newy = self.agentList[i].move(actions[i], self.grid)
				nextloc.append((newx,newy))
			else:
				nextloc.append(prevloc[i])

		# Resolve conflict if positions overlap
		conflict = True
		while conflict:
			conflict = self.resolveConflict(nextloc, prevloc)
		
		# Move agents to their final next locations 
		for index, value in enumerate(nextloc):
			self.grid[value[1]][value[0]]=Cell.AGENTS[index]
			self.agent_pos[index] = value[1], value[0]
		
			if (value[1], value[0]) in self.destinations:
				
				if not self.dones[index]:

					rewards[index]+= Rewards.WIN
					self.dones[index]= True

				# self.grid[value[1]][value[0]] = 2 
			
			elif not self.dones[index] :
				rewards[index] += Rewards.TIMEPENALTY

		rewards += self.penalty
		infos={}

		return self.observation(), rewards, self.dones, infos


	def resolveConflict(self, nextloc, prevloc):
        
		conflict = self.__checkMove(nextloc)

		if conflict:
			for i in conflict:
				for agent in i[1]:
					prevx, prevy = prevloc[agent]
					self.penalty[agent] +=Rewards.CONFLICTPENALTY
					self.agentList[agent].makeMove(prevx,prevy)
					nextloc[agent] = (prevx, prevy)
			return True
		
		return False 

	def __checkMove(self, poslist):
		
		counter = 0
		movedict = {}

		for i, v in enumerate(poslist):

			if v not in movedict:
				movedict[v] = [i]
			else:
				movedict[v].append(i)
		
		return [ (i,v) for i, v in movedict.items() if len(v) > 1] 

	def observation(self):
		statearray=[]
		for i in self.agentList:

			state = np.zeros(self.observation_space[0].shape)

			agents = np.isin(self.grid, Cell.AGENTS ).astype(np.float32)	
			agenti = np.isin(self.grid, i.idx ).astype(np.float32)			

			destinations = np.isin(self.grid, Cell.DESTINATIONS).astype(np.float32) 

			wall = np.isin(self.grid, Cell.WALL).astype(np.float32)

			#exclude self from agents pos list
			agents = agents - agenti

			for idx in range(self.n_agents):
				state[0] = agenti
				state[1] = agents
				state[2] = destinations
				state[3] = wall

			statearray.append(state)
		
		return statearray

	def __init_full_obs(self):
	    self.agent_pos = copy.copy(self.init_agent_pos)
	    self.agent_prev_pos = copy.copy(self.init_agent_pos)
	    self._full_obs = self.grid
	    self.__draw_base_img()



	def __draw_base_img(self):
		self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
		for row in range(self._grid_shape[0]):
			for col in range(self._grid_shape[1]):
				if 2 == self.grid[col][row]:
					fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.05)
				elif PRE_IDS['destinations'] is self.grid[col][row]:
					fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=DESTINATION_COLOR, margin=0.05)
				elif PRE_IDS['agent'] is self.grid[col][row]:
					fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=AGENT_COLOR[0], margin=0.05)


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
				if self.grid[col][row] == 1:
					fill_cell(self._base_img, (col, row), cell_size=CELL_SIZE, fill=DESTINATION_COLOR, margin=0.05)


		img = np.asarray(img)
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen

