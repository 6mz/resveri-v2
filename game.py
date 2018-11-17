import numpy as np
import logging

# 定义四子棋游戏
class Game:

    # 初始化一局游戏
	def __init__(self):		
        # 当前玩家为1
		self.currentPlayer = 1
        # 创建游戏状态:
        # 棋盘大小为8x8，第一回合是1号玩家（一号玩家：1，二号玩家：-1）
		board=np.zeros(64,dtype=np.int)     
        # 摆上初始棋子 28 35；27 36
		board[28]=1
		board[35]=1
		board[27]=-1
		board[36]=-1
		self.gameState = GameState(board, 1)
        
        # 创建行动空间，大小为65=8x8+1
		self.actionSpace = np.zeros(65,dtype=np.int)
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.grid_shape = (8,8)# 棋盘网格形式为8x8，用于创建神经网络的输入
		self.input_shape = (2,8,8)
		self.name = 'Reversi'# 游戏名
        # 计算游戏状态空间大小（128=2*64）
		self.state_size = len(self.gameState.binary)
        # 计算行动空间的大小（65）
		self.action_size = len(self.actionSpace)


    # 重置游戏
	def reset(self):
        # 创建游戏状态:
        # 棋盘大小为8x8，第一回合是1号玩家（一号玩家：1，二号玩家：-1）
		board=np.zeros(64,dtype=np.int)     
        # 28 35；27 36
		board[28]=1
		board[35]=1
		board[27]=-1
		board[36]=-1
		self.gameState = GameState(board, 1)
        # 当前玩家为1
		self.currentPlayer = 1
        # 返回游戏状态
		return self.gameState
    
    
    # 进行一步游戏
	def step(self, action):
        # 调用游戏状态类中的takeAction方法，传入action参数（即落子位置）
		next_state, value, done = self.gameState.takeAction(action)
        # 更新游戏状态
		self.gameState = next_state
        # 变换棋手（next_state 已经换了）
		self.currentPlayer = -self.currentPlayer
		info = None
		#self.gameState.prints()
		return ((next_state, value, done, info))


    # 对称性（棋盘是轴对称和旋转对称的）
    # 返回输入状态所有等效的其他状态及价值
	def identities(self, state, actionValues):
		identities = []
		currentBoard = state.board
		currentAV = actionValues[:64]

       # 旋转
		for n in range(5):
			currentBoard =np.rot90(currentBoard.reshape(8,8),3).reshape(64)
#          和下面等效            
#			currentBoard = np.array([
#						  currentBoard[56], currentBoard[48],currentBoard[40], currentBoard[32],currentBoard[24], currentBoard[16], currentBoard[ 8],currentBoard[0]
#						, currentBoard[57], currentBoard[49],currentBoard[41], currentBoard[33],currentBoard[25], currentBoard[17], currentBoard[ 9],currentBoard[1]
#						, currentBoard[59], currentBoard[50],currentBoard[42], currentBoard[34],currentBoard[26], currentBoard[18], currentBoard[10],currentBoard[2]
#						, currentBoard[60], currentBoard[51],currentBoard[43], currentBoard[35],currentBoard[27], currentBoard[19], currentBoard[11],currentBoard[3]
#						, currentBoard[60], currentBoard[52],currentBoard[44], currentBoard[36],currentBoard[28], currentBoard[20], currentBoard[12],currentBoard[4]
#						, currentBoard[61], currentBoard[53],currentBoard[45], currentBoard[37],currentBoard[29], currentBoard[21], currentBoard[13],currentBoard[5]
#						, currentBoard[62], currentBoard[54],currentBoard[46], currentBoard[38],currentBoard[30], currentBoard[22], currentBoard[14],currentBoard[6]
#						, currentBoard[63], currentBoard[55],currentBoard[47], currentBoard[39],currentBoard[31], currentBoard[23], currentBoard[15],currentBoard[7]
#						])
			currentAV =np.rot90(currentAV.reshape(8,8),3).reshape(64)
			currentAVs =np.append(currentAV,actionValues[64]) 
			identities.append((GameState(currentBoard, state.playerTurn), currentAVs))

       # 翻转
		currentBoard = np.flipud(currentBoard.reshape(8,8)).reshape(64)
		currentAV = np.flipud(currentAV.reshape(8,8)).reshape(64)

		for n in range(5):
			currentBoard =np.rot90(currentBoard.reshape(8,8),3).reshape(64)
			currentAV =np.rot90(currentAV.reshape(8,8),3).reshape(64)
			currentAVs =np.append(currentAV,actionValues[64])             
			identities.append((GameState(currentBoard, state.playerTurn), currentAVs))

		return identities


# 游戏状态类
class GameState():
    
    # 初始化类（根据参数创建状态）
	def __init__(self, board, playerTurn):
        # 获得棋盘（从参数中）
		self.board = board
        # 创建字典定义棋子外形
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
        # 定义胜利条件 （行，列，斜线）
		self.winners = []
		self.direction = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]] 
        # 获得本回合执棋的玩家（从参数中）
		self.playerTurn = playerTurn
        # 获得当前棋子信息数组（二进制），当前棋手在前，对方棋手在后（大小84=2*42）
		self.binary = self._binary()
        # 获得描述当前棋盘中棋子位置的字符串（id），1号棋手在前，-1号棋手在后（大小84=2*42字节）
		self.id = self._convertStateToId()
        # 获取这一回合允许落子的位置
		self.allowedActions = self._allowedActions()
        # 判断游戏是否结束，1为结束
		self.isEndGame = self._checkForEndGame()
        # 游戏结束时的价值（惩罚及奖励）和分数计算
		self.value = self._getValue()
		self.score = self._getScore()



    ################  以下函数在初始化时已执行  ####################
        
    # 获取这一回合允许落子的位置
	def _allowedActions(self,player=1):
	    allowed = []
        # 对于每一个棋盘点[0-63]
	    for ind in range(len(self.board)):
            # 为空可以落子
	        if self.board[ind]==0:
	            i0=ind//8
	            j0=ind%8                
	            for di,dj in self.direction:
	                other=0
	                i=i0 + di
	                j=j0 + dj 
	                while i>=0 and i<8 and j>=0 and j<8 and self.board[i*8+j]== -self.playerTurn * player:
	                     other=other+1
	                     i=i+di
	                     j=j+dj                            
	                if other>0 and i>=0 and i<8 and j>=0 and j<8  and self.board[i*8+j]==self.playerTurn * player:
	                     allowed.append(ind)
	                     break
       # 如果为空	                                                
	    if not allowed:
	          allowed.append(64)                           	
	    return allowed


    # 获得当前棋子信息数组（二进制），当前棋手在前，对方棋手在后（大小128=64x2）
	def _binary(self):
        
        # 获得当前棋手的棋子位置二进制数组：
        # 第一步，创建一个和棋盘一样大的列表
		currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        # 第二步，获得棋盘上编号和现在棋手一样的位置，并置为1（利用np的高级索引）
		currentplayer_position[self.board==self.playerTurn] = 1

        # 获得对手的棋子位置，思路同上一步。
		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-self.playerTurn] = 1

        # 根据当前棋手的棋子位置和对手的棋子位置两个数组创建二进制棋子信息数组，
        # 当前棋手在前（大小128=2*64）
        # 以后reshape成2x8x8
		position = np.append(currentplayer_position,other_position)

		return (position)


    # 获得描述当前棋盘中棋子位置的字符串，1号棋手在前，-1号棋手在后（大小128=64x2）
	def _convertStateToId(self):
        # 获得1棋手的棋子位置数组：
		player1_position = np.zeros(len(self.board), dtype=np.int)
		player1_position[self.board==1] = 1

        # 获得-1棋手的棋子位置数组：
		other_position = np.zeros(len(self.board), dtype=np.int)
		other_position[self.board==-1] = 1
        
        # 连接，1在前，-1在后
		position = np.append(player1_position,other_position)
        
        # 将数组转化为字符串（长度84）
        # map方法返回的是迭代器，所以还需要用join方法连接
        # str(a)
        # Out[9]: '[ 0  1  2  3  4  5  6  7  8  9 10 11]'
		id = ''.join(map(str,position))
		 
		if self.playerTurn == 1:
		    id = id + '1'
		else:
		    id = id + '0'            

		return id


    # 判断游戏是否结束，1为结束（本方法在本回合落子前执行）
	def _checkForEndGame(self):
        # 如果非零元素数量为64（即棋盘已满），则结束游戏
		if np.count_nonzero(self.board) == 64:
			return 1
        # 否则判断规则中的条件是否满足（双方都无棋可下）
		if self.allowedActions==[64] and self._allowedActions(player=-1)==[64]:
			return 1
		return 0


    # 获取价值，当前棋手在前，另一位在后
    # 不可能返回1的价值，因为在回合落子前判断的，只有继续、输了和平局集中可能
    #（或者准确的说，是选手落完子后，另一个创建选手的状态时返回的价值，此时选手已经改变）
	def _getValue(self):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose
		if self.isEndGame==1:
			if np.sum(self.board==self.playerTurn)<np.sum(self.board==-self.playerTurn):
				return (-1, -1, 1)
			elif np.sum(self.board==self.playerTurn)>np.sum(self.board==-self.playerTurn):
				return (1, 1, -1)            
		return (0, 0, 0)

    
    # 从价值中获得分数：把（-1,-1,1）拆成-1和（-1,1）；最后返回的value是-1（见takeAction函数）
    # 当前棋手在前，另一位在后
	def _getScore(self):
		tmp = self.value
		return (tmp[1], tmp[2])

    ################  以上函数在初始化时已执行  ######################


        
    # 根据动作（即落子位置编号）进行一步游戏，返回新的状态
    # 不改变原状态
	def takeAction(self, action):
        # 根据原有的棋盘复制创建一个新棋盘(np.array的赋值是复制)
		newBoard = np.array(self.board)
        # 新棋盘上对应位置落当前棋手的子
		if action<64:
		    newBoard[action]=self.playerTurn     
		    for di,dj in self.direction:
		        other=0
		        i=action//8 + di
		        j=action% 8 + dj                
		        changes=[] 
		        while i>=0 and i<8 and j>=0 and j<8 and self.board[i*8+j]== -self.playerTurn :
		            other=other+1
		            changes.append(i*8+j)  
		            i=i+di
		            j=j+dj                      
		        if other>0 and i>=0 and i<8 and j>=0 and j<8 and self.board[i*8+j]==self.playerTurn :
		            newBoard[changes]=self.playerTurn 
            
        # 落完子后立刻更换游戏棋手，同时更新游戏状态，并完成对游戏胜负的判断
		newState = GameState(newBoard, -self.playerTurn)

		value = 0
		done = 0
        # 如果游戏结束，done置为1
		if newState.isEndGame:
			value = newState.value[0]
			done = 1

        # 返回新状态，价值，及结束标志位
		return (newState, value, done) 



    
    # 输出棋盘到日志
	def render(self, logger):
        # 6 行
		for r in range(8):
            # 读取每一行的7个位置的棋子，用pieces字典中的样子代替（str把1变成'1'）
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            #----------------------------------------
			logger.info([self.pieces[str(x)] for x in self.board[8*r : (8*r + 8)]])
		logger.info('--------------')
        
    # 输出棋盘到日志
	def prints(self):
        # 6 行
		print('')
		for r in range(8):
            # 读取每一行的7个位置的棋子，用pieces字典中的样子代替（str把1变成'1'）
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            # ['-', '-', '-', '-', '-', '-', '-']
            #----------------------------------------
			print([self.pieces[str(x)] for x in self.board[8*r : (8*r + 8)]])
		print('--------------')        