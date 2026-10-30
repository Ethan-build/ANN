import numpy as np
import pandas as pd
import os

def CreateNN(nn):
	"""创建神经网络
	"""
	
	net = {} # 神经网络容器
	net['nn'] = nn # 网络结构列表，保存完整的网络结构 [7, 6, 6, 6, 1]
	net['M'] = len(nn)-1 # 计算层数（不包括输入层）
	net['layers'] = nn[1:] # 所有需要计算权重和偏置的层 [6, 6, 6, 1]
	net = w_Create(net) # 初始化随机权重向量
	net['w'] = net['w0'].copy() # w工作权重（训练中更新），w0初始权重（保持不变）
	net['N'] = len(net['w0']) # 权重总数
	return net
	
def w_Create(net):
	"""	创建神经网络的随机权重向量，并定义导数计算所需的集合
	"""
	
	M = net['M'] # 网络层数
	layers = net['layers'] # 所有需要计算权重和偏置的层 [6, 6, 6, 1]
	inputs = net['nn'][0] # 输入层节点数

	X = []	# 输入层集合
	U = [] 	# 输出层集合（用于代价函数计算）

	IW = {}	# 输入权重矩阵，IW[1, 1]是输入层到隐藏层1的权重矩阵
	LW = {} # LW[m,l] 层m到层l的连接权重矩阵

	L_b = {}# L_b[m]: 与层m有后向连接的层集合
	L_f = {}# L_f[m]: 与层m有前向连接的层集合

	CX_LW = {} # CX_LW[u]: 与u有连接的所有输入层集合
	CU_LW = {} # CU_LW[x]: 与x有连接的所有输出层集合
	b = {} # b[m]: 层m的偏置向量

	'''输入层'''
	I = {}	# 与层1有连接的输入集合
	I[1]=[1] 	# 输入只连接到层1
	
	# 输入权重矩阵
	# Xavier 初始化（适合 tanh 激活函数）
	IW[1, 1] = np.random.randn(layers[0], inputs) * np.sqrt(2.0 / (inputs + layers[0]))
	X.append(1) 	# 第一层是输入层
	
	'''内部连接权重矩阵'''
	for m in range(1,M+1):
		L_b[m] = [] # L_b[m]: 与层m有后向连接的层集合
		L_f[m] = [] # L_f[m]: 与层m有前向连接的层集合

		if m>1:
			l = m-1
			# Xavier 初始化（适合 tanh 激活函数）
			LW[m, l] = np.random.randn(layers[m-1], layers[l-1]) * np.sqrt(2.0 / (layers[l-1] + layers[m-1]))
			L_b[l].append(m) # 层m与层l有后向连接
			L_f[m].append(l) # 层l与层m有前向连接

		b[m] = np.random.randn(layers[m-1]) * 0.01 # 偏置用小随机值
	
	if M not in U:
		U.append(M) # 如果M还未加入，将M添加到输出层

	for u in U:
		CX_LW[u] = []
	for x in range(1,M+1):
		CU_LW[x] = []

	# 添加到神经网络
	net['U'] = U
	net['X'] = X
	net['L_b'] = L_b
	net['L_f'] = L_f
	net['I']=I
	net['CX_LW'] = CX_LW
	net['CU_LW'] = CU_LW
	net['w0'] = Wb2w(net,IW,LW,b)
	return net
	
def Wb2w(net,IW,LW,b):
	"""	将输入权重矩阵IW、连接权重矩阵LW和偏置向量b转换为权重向量w
	返回:
		w: 		权重向量
	"""

	I = net['I'] # 与层1有连接的输入集合
	L_f = net['L_f'] # L_f[m]: 与层m有前向连接的层集合
	M = net['M'] # 神经网络层数
	
	w = np.array([]) # 空权重向量

	for m in range(1,M+1):
		# 输入权重
		if m == 1:
			for i in I[m]:
				w = np.append(w,IW[m,i].flatten('F'))
		# 内部连接权重
		for l in L_f[m]:
				w = np.append(w,LW[m,l].flatten('F'))
		# 偏置权重
		w = np.append(w,b[m])
	
	return w
	
def w2Wb(net):
	"""	将权重向量w转换为输入权重矩阵IW、连接权重矩阵LW和偏置向量b

	参数:
		net: 	神经网络（包含权重向量w）		
	返回:
		IW		输入权重矩阵
		LW 		LW[m,l] 层m到层l的连接权重矩阵
		b		b[m]: 层m的偏置向量
	"""

	I = net['I'] # 与层1有连接的输入集合
	L_f = net['L_f'] # L_f[m]: 与层m有前向连接的层集合
	M = net['M'] # 神经网络层数
	layers = net['layers'] # 神经网络结构
	inputs = net['nn'][0] # 输入数量
	w_temp = net['w'].copy() # 权重向量
		
	IW = {}	# 输入权重矩阵
	LW = {} # LW[m,l,d] 层m到层l带延迟d的连接权重矩阵
	b = {} # b[m]: 层m的偏置向量
	
	for m in range(1,M+1):
	
		# 输入权重
		if m==1:
			for i in I[m]:
				w_i = inputs*layers[m-1]
				vec = w_temp[0:w_i]
				w_temp = w_temp[w_i:]
				IW[m,i] = np.reshape(vec,(layers[m-1],int(len(vec)/layers[m-1])),order='F')
		
		# 内部连接权重
		for l in L_f[m]:
			w_i = layers[l-1]*layers[m-1]
			vec = w_temp[0:w_i]
			w_temp = w_temp[w_i:]
			LW[m,l] = np.reshape(vec,(layers[m-1],int(len(vec)/layers[m-1])),order='F')
		
		# 偏置权重
		w_i = layers[m-1]
		b[m] =w_temp[0:w_i]
		w_temp = w_temp[w_i:]

	return IW,LW,b


def NNOut_(P,net,IW,LW,b,a={},q0=0):
	"""	计算给定输入P的神经网络输出
		仅供内部使用
	
	参数:
		P:		神经网络输入
		net: 	神经网络
		IW:		输入权重矩阵
		LW:		LW[m,l] 层m到层l的连接权重矩阵
		b:		b[m]: 层m的偏置向量
		a:		神经网络层输出，用于已知历史数据
		q0:		从数据点q0开始使用数据 P[q0:]
	返回:
		Y_NN: 	神经网络对输入P的输出
		a:		神经网络层输出
		n:		层的加权和输出
	"""

	I = net['I'] # 与层1有连接的输入集合
	L_f = net['L_f'] # L_f[m]: 与层m有前向连接的层集合
	M = net['M'] # 神经网络层数
	outputs = net['nn'][-1] # 输出数量
	
	n = {} # 层的加权和输出
	Q = P.shape[1] # 输入数据点数量

	Y_NN = np.zeros((outputs, Q)) # 神经网络输出
	
	for q in range(q0+1,Q+1):
	# 遍历所有数据点
		a[q, 1] = 0
		for m in range(1,M+1):
		# 遍历所有层m
			n[q, m] = 0 # 数据点q，层m的加权和输出

			if m==1:
				for i in I[m]:
					if q > 0:
						n[q, m] = n[q, m] + np.dot(IW[m, i], P[:, q-1])

			for l in L_f[m]:
					if q > 0:
						n[q, m] = n[q, m] + np.dot(LW[m, l], a[q, l])
			# 偏置
			n[q, m] = n[q, m] + b[m]
			
			# 计算层输出
			if m == M:
				a[q, M] = n[q, M] # 输出层使用线性激活
			else:
				a[q, m] = np.tanh(n[q, m]) # 隐藏层使用tanh激活
		Y_NN[:, q-1] = a[q, M]
	Y_NN = Y_NN[:, q0:]
	return Y_NN,n,a
	
def NNOut(P,net,P0=None,Y0=None):
	"""	计算给定输入P的神经网络输出
		用户函数

	参数:
		P:		神经网络输入
		net: 	神经网络
		P0:		之前的输入数据
		Y0:		之前的输出数据
	返回:
		Y_NN: 	神经网络对输入P的输出
	"""
	Y = np.zeros((net['layers'][-1], int(np.size(P)/net['nn'][0])))
	data, net = prepare_data(P,Y,net,P0=P0,Y0=Y0)
	IW,LW,b = w2Wb(net) # 输入权重矩阵、连接权重矩阵、偏置向量
	Y_NN = NNOut_(data['P'],net,IW,LW,b,a=data['a'],q0=data['q0'])[0]
	
	# 缩放归一化输出
	Y_NN_scaled = Y_NN.copy()
	for y in range(np.shape(Y_NN)[0]):
		Y_NN_scaled[y] = Y_NN[y]*net['normY'][y]
	
	if np.shape(Y_NN_scaled)[0]==1:
		Y_NN_scaled = Y_NN_scaled[0]
	return Y_NN_scaled
	
	
def RTRL(net,data):
	"""	基于以下论文实现实时递归学习算法（RTRL）：
		Williams, Ronald J.; Zipser, David: A Learning Algorithm for Continually Running
		Fully Recurrent Neural Networks. In: Neural Computation, Nummer 2, Vol. 1
		(1989), S. 270-280.
		
	参数:
		net:	神经网络
		data: 	训练数据
	返回:
		J: 		Jacobian矩阵，e关于权重向量w的导数
		E:		神经网络相对于训练数据的均方误差
		e:		误差向量：神经网络输出与目标数据['Y']的差值
	"""
	P = data['P']	# 训练数据输入
	Y = data['Y']	# 训练数据输出
	a = data['a']	# 层输出
	q0 = data['q0']	# 使用训练数据 [q0:]
	
	I = net['I'] # 与层1有连接的输入集合
	L_f = net['L_f'] # L_f[m]: 与层m有前向连接的层集合
	L_b = net['L_b'] # L_b[m]: 与层m有后向连接的层集合
	M = net['M'] # 神经网络层数
	inputs = net['nn'][0] # 输入数量
	outputs = net['nn'][-1] # 输出数量
	layers = net['layers'] # 神经网络结构

	U = net['U'] # 输入层集合
	X = net['X'] # 输出层集合（用于代价函数计算）

	CU_LW = net['CU_LW'] # CU_LW[x]: 与x有连接的所有输出层集合		
	IW,LW,b = w2Wb(net) # 输入权重矩阵、连接权重矩阵、偏置向量
	
	########################
	# 1. 计算神经网络输出
	Y_NN,n,a = NNOut_(P,net,IW,LW,b,a=a,q0=q0)
	
	########################
	# 2. 计算代价函数E
	Y_delta = Y - Y_NN # 误差矩阵
	e = np.reshape(Y_delta,(1,np.size(Y_delta)),order='F')[0] # 误差向量
	E = np.dot(e,e.transpose()) # 代价函数（均方误差）
	
	#########################
	# 3. RTRL反向传播
	
	Q = P.shape[1] # 输入数据点数量
	Q0 = Q-q0 # 不包含"旧数据"的数据点数量
	
	# 定义
	dAu_db = {}		# 层输出a(u)关于偏置向量b的导数
	dAu_dIW = {}	# 层输出a(u)关于输入权重IW的导数
	dAu_dLW = {}	# 层输出a(u)关于连接权重LW的导数
	dA_dw = {}		# 层输出a关于权重向量w的导数
	S = {}			# 灵敏度矩阵
	Cs = {}			# Cs[u]: 存在灵敏度矩阵S[q,u,m]的层m集合
	CsX = {}		# CsX[u]: 存在灵敏度矩阵S[q,u,x]的输入层x集合
					# Cs和CsX在反向传播过程中生成
			
	# 初始化
	J = np.zeros((Q0*layers[-1],net['N']))	# Jacobian矩阵
	for q in range(1,q0+1):
		for u in U:
			dAu_dLW[q,u] = np.zeros((layers[u-1],net['N']))
	
	###
	# 开始RTRL
	for q in range(q0+1,Q+1):
	
		# 初始化
		U_ = [] # 计算灵敏度所需的集合
		for u in U:
			Cs[u] = []
			CsX[u] = []
			dA_dw[q,u] = 0
			
		# 计算灵敏度矩阵
		for m in range(M,1-1,-1):
		# 按反向传播顺序递减m
		
			for u in U_:
				S[q,u,m] = 0 # 灵敏度矩阵 层u->m
				for l in L_b[m]:
					S[q,u,m] = S[q,u,m] \
						+ np.dot(np.dot(S[q,u,l],LW[l,m]),np.diag(1-(np.tanh(n[q,m]))**2))
				if m not in Cs[u]:
					Cs[u].append(m) # 将m添加到集合Cs[u]
					if m in X:
						CsX[u].append(m) # 如果m在X中，添加到CsX[u]
			
			if m in U:
				if m == M:
					# 输出层是线性的，无激活函数
					S[q,m,m] = np.diag(np.ones(outputs)) # 灵敏度矩阵 S[M,M]
				else:
					S[q,m,m] = np.diag(1-(np.tanh(n[q,m]))**2) # 灵敏度矩阵 S[m,m]
				
				U_.append(m) # 将m添加到U'
				Cs[m].append(m) # 将m添加到Cs
				if m in X:
					CsX[m].append(m) # 如果m在X中，添加到CsX[m]
		
		# 计算导数
		for u in sorted(U):
			# 静态导数计算
			dAe_dw = np.empty((layers[u-1],0)) # 静态导数向量：层输出关于权重向量的显式导数
			for m in range(1,M+1):
				# 输入权重
				if m==1:
					for i in I[m]:
						if ((q,u,m) not in S.keys()):
							# 如果灵敏度矩阵不存在：导数为零
							dAu_dIW[m,i] = \
								np.kron(P[:,q-1].transpose(),\
										np.zeros((layers[u-1],layers[m-1])))
						else:
							# 输出层u关于IW[m,i]的导数
							dAu_dIW[m,i] = \
								np.kron(P[:,q-1].transpose(),S[q,u,m])
						dAe_dw = np.append(dAe_dw,dAu_dIW[m,i],1) # 添加到静态导数向量
	
				# 连接权重
				for l in L_f[m]:
					if ((q,u,m) not in S.keys()):
						# 如果灵敏度矩阵不存在：导数为零
						dAu_dLW[m,l] = \
							np.kron(a[q,l].transpose(),\
									np.zeros((layers[u-1],layers[m-1])))
					else:
						dAu_dLW[m,l] = \
							np.kron(a[q,l].transpose(),S[q,u,m])
						# 输出层u关于LW[m,i]的导数
					dAe_dw = np.append(dAe_dw,dAu_dLW[m,l],1) # 添加到静态导数向量
				
				# 偏置权重
				if ((q,u,m) not in S.keys()):
					dAu_db[m] = np.zeros((layers[u-1],layers[m-1])) # 导数为零
				else:
					dAu_db[m] = S[q,u,m] # 输出层u关于b[m]的导数
				dAe_dw = np.append(dAe_dw,dAu_db[m],1) # 添加到静态导数向量
				
			# 动态导数计算
			# 注意：当前网络为纯前馈结构 [7,6,6,6,1]，无层内循环连接，
			# 因此动态导数 dAd_dw 应为0。以下代码保留用于将来扩展递归连接。
			dAd_dw = 0
			for x in CsX[u]:
				for u_ in CU_LW[x]:
					# sum_d = Σ ∂a_u'(q-d)/∂w × LW(x,u')（递归梯度累积）
					# 当前网络无递归连接，不需要时间延迟累积
					pass
				# 当前网络无递归连接，跳过动态导数累加
					
			# 静态 + 动态导数
			dA_dw[q,u] = dAe_dw + dAd_dw # 输出层u关于w的总导数
			
		# Jacobian矩阵
		J[range(((q-q0)-1)*outputs,(q-q0)*outputs),:] = -dA_dw[q,M]
		
		# 删除dA_dw中早于q-max_delay的条目
		if q > 0:
			new_dA_dw = dict(dA_dw)
			for key in dA_dw.keys():
				if key[0] == q:
					del new_dA_dw[key]
			dA_dw = new_dA_dw
		
		# 重置S
		S = {}
		
	return J,E,e

	
def train_LM(P,Y,net,k_max=100,E_stop=1e-10,dampfac=3.0,dampconst=10.0,\
			verbose = False,min_E_step=1e-09,val_tuple=None,val_patience=20):
	"""	基于以下论文实现Levenberg-Marquardt算法（LM）：
		Levenberg, K.: A Method for the Solution of Certain Problems in Least Squares.
		Quarterly of Applied Mathematics, 2:164-168, 1944.
		以及
		Marquardt, D.: An Algorithm for Least-Squares Estimation of Nonlinear Parameters.
		SIAM Journal, 11:431-441, 1963.
		
	参数:

		P:		神经网络输入
		Y:		神经网络目标值
		net: 	神经网络
		k_max:	最大迭代次数
		E_stop:	终止误差，当误差 <= E_stop时训练停止
		dampconst:	用于调整LM阻尼因子的常数
		dampfac:	LM阻尼因子
		min_E_step: 误差的最小步长。达到5次后训练终止。
		val_tuple: 	验证集 (P_val, Y_val) 元组（可选）
		val_patience: 验证误差不降的最大容忍轮数（默认20）
	返回:
		net: 	训练好的神经网络（验证误差最低时的权重）
	"""
	# 创建数据字典
	data,net = prepare_data(P,Y,net)

	# 准备验证集数据（使用相同的归一化因子）
	val_data = None
	if val_tuple is not None:
		P_val, Y_val = val_tuple
		val_data, _ = prepare_data(P_val, Y_val, net)
	
	# 计算第一次迭代的Jacobian、误差和误差向量
	J,E,e = RTRL(net,data)
	k = 0
	ErrorHistory=np.zeros(k_max+1) # 误差历史记录向量
	ErrorHistory[k]=E

	# 验证集跟踪
	val_history = []
	best_val_E = np.inf
	best_w = net['w'].copy()
	best_k = 0
	val_no_improve = 0

	if verbose:
		val_str = ''
		if val_data is not None:
			E_val = calc_error(net, val_data)
			val_history.append(E_val)
			best_val_E = E_val
			best_w = net['w'].copy()
			val_str = f'		Val Error: {E_val:.6f}'
		print('Iteration: ',k,'		Error: ',E,'	scale factor: ',dampfac, val_str)
	
	early=0

	while True:
	# 循环直到达到k_max或E_stop

		JJ = np.dot(J.transpose(),J) # J的转置 * J
		w = net['w'] # 权重向量
		while True:
		# 重复直到优化步骤成功
			# 梯度
			g = np.dot(J.transpose(),e)
			
			# 计算缩放的逆Hessian矩阵
			try:
				G = np.linalg.inv(JJ+dampfac*np.eye(net['N'])) # 缩放的逆Hessian矩阵
			except np.linalg.LinAlgError:
				# 不可逆。沿梯度方向小步前进
				w_delta = 1.0/1e10 * g
			else:
				# 计算权重修改量
				w_delta = np.dot(-G,g)
			
			net['w'] = w + w_delta # 新权重向量
			
			Enew = calc_error(net,data) # 计算新误差E			
			if Enew<E and abs(E-Enew)>=min_E_step:
			# 优化步骤成功！
				dampfac= dampfac/dampconst # 调整缩放因子
				early=0 # 重置早停条件
				break # 进入下一次迭代
			else:
			# 优化步骤不成功！
				dampfac = dampfac*dampconst # 调整缩放因子
				if abs(E-Enew)<=min_E_step:
					early=early+1
					
					if verbose:
						print('E-Enew<=min_E_step Encountered!!')
						if early>=5.0:
							print('5 Times * E-Enew<=min_E_step Encountered!!')
					break                    
		
		# 计算下一次迭代的Jacobian、误差和误差向量
		J,E,e = RTRL(net,data)
		k = k+1
		ErrorHistory[k] = E

		# 计算验证集误差（如有）
		val_str = ''
		if val_data is not None:
			E_val = calc_error(net, val_data)
			val_history.append(E_val)
			if E_val < best_val_E:
				best_val_E = E_val
				best_w = net['w'].copy()
				best_k = k
				val_no_improve = 0
				val_str = f'		Val Error: {E_val:.6f} *BEST*'
			else:
				val_no_improve += 1
				val_str = f'		Val Error: {E_val:.6f}'

		if verbose:
			print('Iteration: ',k,'		Error: ',E,'	scale factor: ',dampfac, val_str)
	
		# 检查是否满足终止条件
		if k>=k_max:
			print('Maximum number of iterations reached')
			break
		elif E<=E_stop:
			print('Termination Error reached')
			break
		elif early>=5.0:
			print('Error decreased 5 times by minimum step. Force training exit.')
			break
		elif val_data is not None and val_no_improve >= val_patience:
			print(f'Validation error did not improve for {val_patience} epochs. Early stopping.')
			break
        
	net['ErrorHistory'] = ErrorHistory[:k]

	# 恢复到验证误差最低时的权重
	if val_data is not None and best_val_E != np.inf:
		net['w'] = best_w
		net['best_val_E'] = best_val_E
		net['best_k'] = best_k
		net['val_history'] = val_history
		if verbose:
			print(f'Restored best weights from epoch {best_k} (Val Error: {best_val_E:.6f})')
	
	return net
	
	
def calc_error(net,data):
	"""	基于数据计算神经网络的误差
		
	参数:
		net:	神经网络
		data: 	训练数据
	返回:
		E:		神经网络相对于训练数据的均方误差
	"""
	P = data['P']	# 训练数据输入
	Y = data['Y']	# 训练数据输出
	a = data['a']	# 层输出
	q0 = data['q0']	# 使用训练数据 [q0:]
	
	IW,LW,b = w2Wb(net) # 输入权重矩阵、连接权重矩阵、偏置向量
	
	########################
	# 1. 计算神经网络输出
	Y_NN,n,a = NNOut_(P,net,IW,LW,b,a=a,q0=q0)
	
	########################
	# 2. 计算代价函数E
	Y_delta = Y - Y_NN # 误差矩阵
	e = np.reshape(Y_delta,(1,np.size(Y_delta)),order='F')[0] # 误差向量
	E = np.dot(e,e.transpose()) # 代价函数（均方误差）
	
	return E
	
def prepare_data(P,Y,net,P0=None,Y0=None):
	"""	准备神经网络训练数据并检查错误
		
	参数:
		P:		神经网络输入
		Y: 		神经网络目标值
		net: 	神经网络
		P0:		之前的输入数据
		Y0:		之前的输出数据
	返回:
		data:	包含训练或计算输出所需数据的字典
	"""	
	
	# 如果给定1D数组，将P和Y转换为2D数组
	if P.ndim==1:
		P = np.array([P])
	if Y.ndim==1:
		Y = np.array([Y])
		
	# 检查输入和输出数据是否匹配神经网络结构	
	if np.shape(P)[0] != net['nn'][0]:
		raise ValueError("Dimension of Input Data does not match number of inputs of the NN")
	if np.shape(Y)[0] != net['nn'][-1]:
		raise ValueError("Dimension of Output Data does not match number of outputs of the NN")
	if np.shape(P)[1] != np.shape(Y)[1]:
		raise ValueError("Input and output data must have same number of datapoints Q")
	
	# 检查是否给定了之前的数据
	if (P0 is not None) and (Y0 is not None):

		# 如果给定1D数组，将P0和Y0转换为2D数组
		if P0.ndim==1:
			P0 = np.array([P0])
		if Y0.ndim==1:
			Y0 = np.array([Y0])
			
		# 检查输入和输出数据是否匹配神经网络结构
		if np.shape(P0)[0] != net['nn'][0]:
			raise ValueError("Dimension of previous Input Data P0 does not match number of inputs of the NN")
		if np.shape(Y0)[0] != net['nn'][-1]:
			raise ValueError("Dimension of previous Output Data Y0 does not match number of outputs of the NN")
		if np.shape(P0)[1] != np.shape(Y0)[1]:
			raise ValueError("Previous Input and output data P0 and Y0 must have same number of datapoints Q0")

		q0 = np.shape(P0)[1] # 给定的之前数据点数量
		a = {} # 初始化层输出
		for i in range(1,q0+1):
			for j in range(1,net['M']):
				a[i,j]=np.zeros(net['nn'][j]) # 隐藏层的层输出未知 -> 设为零
			a[i,net['M']]=Y0[:,i-1]/net['normY'] # 设置输出层的层输出

		# 将之前的输入和输出添加到输入/输出矩阵
		P_ = np.concatenate([P0,P],axis=1)
		Y_ = np.concatenate([Y0,Y],axis=1)
	else:
		# 保持输入和输出不变，将q0和a设为默认值
		P_ = P.copy()
		Y_ = Y.copy()
		q0=0
		a={}
	# 归一化
	P_norm = P_.copy()
	Y_norm = Y_.copy()
	if 'normP' not in net.keys():
		normP = np.ones(np.shape(P_)[0])
		for p in range(np.shape(P_)[0]):
			normP[p] = np.max([np.max(np.abs(P_[p])),1.0])
			P_norm[p] = P_[p]/normP[p]
		normY = np.ones(np.shape(Y_)[0])
		for y in range(np.shape(Y_)[0]):
			normY[y] = np.max([np.max(np.abs(Y_[y])),1.0])
			Y_norm[y] = Y_[y]/normY[y]
		net['normP'] = normP
		net['normY'] = normY
	else:
		for p in range(np.shape(P_)[0]):
			P_norm[p] = P_[p]/net['normP'][p]
		normY = np.ones(np.shape(Y)[0])
		for y in range(np.shape(Y_)[0]):
			Y_norm[y] = Y_[y]/net['normY'][y]
		
	# 创建数据字典
	data = {}		
	data['P'] = P_norm
	data['Y'] = Y_norm
	data['a'] = a
	data['q0'] = q0
	
	return data,net

def saveNN(net,filename):
	"""	将神经网络对象保存到文件
		
	参数:
		net: 	神经网络对象
		filename:	保存神经网络的csv文件路径
	
	"""	
	import csv

	# 写入元数据（网络结构、归一化因子、权重标签）
	with open(filename, "w", newline='') as file:
		writer = csv.writer(file, lineterminator='\n')
		writer.writerow(['nn'])
		writer.writerow(net['nn'])
		writer.writerow(['normP'])
		writer.writerow(net['normP'])
		writer.writerow(['normY'])
		writer.writerow(net['normY'])
		writer.writerow(['w'])

	# 追加权重向量（np.savetxt格式）
	with open(filename, "a", newline='') as file:
		np.savetxt(file, net['w'], delimiter=',', fmt='%.55f')
	
	return
	
def loadNN(filename):
	"""	从文件加载神经网络对象
		
	参数:
		filename:	保存神经网络的csv文件路径
	返回:
		net: 	神经网络对象
	"""	
	import csv

	# 读取csv
	data= list(csv.reader(open(filename,"r")))

	# 读取网络结构nn
	nn = list(np.array(data[1],dtype=int))

	# 读取输入数据归一化因子normP
	normP = np.array(data[3],dtype=float)
	
	# 读取输出数据归一化因子normY
	normY = np.array(data[5],dtype=float)
	
	# 读取权重向量w
	w = pd.read_csv(filename,sep=',',skiprows=range(6))['w'].values
	
	# 创建神经网络并分配加载的权重和因子
	net = CreateNN(nn)
	net['normP'] = normP
	net['normY'] = normY
	net['w'] = w
	
	return net

def inputdata(path):#读取输入数据的函数
	"""准备输入数据
	"""
	df = pd.read_csv(path, sep=',')
	P = np.array([df['BDFE_cat'].values, df['BDFE_sub'].values, df['x_cat'].values, df['x_sub'].values, df['d'].values, df['Vbur_cat'].values, df['Vbur_sub'].values])
	Y = np.array(df['gibbs'].values)

	return P, Y

def split_data(P, Y, val_ratio=0.2, shuffle=True, seed=None):
	"""按催化剂 (x_cat, P的第3行) 分层划分训练集和验证集

	参数:
		P:          输入数据 (features × samples)，P[2] 为 x_cat
		Y:          目标数据 (outputs × samples)
		val_ratio:  验证集比例 (默认 0.2)
		shuffle:    是否打乱数据 (默认 True)
		seed:       随机种子
	返回:
		P_train, Y_train: 训练集
		P_val, Y_val:     验证集
	"""
	# 确保 Y 是二维数组 (1, samples)
	if Y.ndim == 1:
		Y = Y.reshape(1, -1)

	rng = np.random.RandomState(seed)
	x_cat = P[2, :]  # x_cat 在 P 的第3行 (index 2)
	unique_cats = np.unique(x_cat)
	train_idx_list = []
	val_idx_list = []
	for cat in unique_cats:
		cat_indices = np.where(x_cat == cat)[0]
		if shuffle:
			rng.shuffle(cat_indices)
		n_val = max(1, int(len(cat_indices) * val_ratio))  # 每种催化剂至少保留1条
		val_idx_list.append(cat_indices[:n_val])
		train_idx_list.append(cat_indices[n_val:])
	train_idx = np.concatenate(train_idx_list)
	val_idx = np.concatenate(val_idx_list)
	if shuffle:
		rng.shuffle(train_idx)
		rng.shuffle(val_idx)
	return P[:, train_idx], Y[:, train_idx], P[:, val_idx], Y[:, val_idx]

def trainNN(nn, P, Y, seed=64, val_ratio=0.2):
	"""训练神经网络

	参数:
		nn:        网络结构列表，如 [7,6,6,6,1]
		P:         输入数据
		Y:         目标数据
		seed:      随机种子（默认64），设为None则使用系统随机状态
		val_ratio: 验证集比例（默认0.2），设为0则不使用验证集
	"""
	if seed is not None:
		np.random.seed(seed)
	net = CreateNN(nn)

	# 划分训练集和验证集
	if val_ratio > 0:
		P_train, Y_train, P_val, Y_val = split_data(P, Y, val_ratio=val_ratio, seed=seed)
		val_tuple = (P_val, Y_val)
	else:
		P_train, Y_train = P, Y
		val_tuple = None

	net = train_LM(P_train, Y_train, net, verbose=True, k_max=500, E_stop=0.25,
	               val_tuple=val_tuple)

	return net

def evaluate_model(name, Y, y_pred):
	"""评估神经网络性能并可视化

	参数:
		name:   数据集名称（用于打印）
		Y:      真实目标值
		y_pred: 神经网络预测值
	"""

	mse = np.average((Y - y_pred) ** 2)
	rmse = np.sqrt(mse)
	mae = np.sum(np.absolute(Y - y_pred)) / len(Y)
	R2 = 1 - mse / np.var(Y)
	print(name, 'RMSE:', rmse, 'MAE:', mae, 'R^2:', R2, '\n', sep='\n')

	import matplotlib.pyplot as plt
	plt.scatter(x=Y, y=y_pred, color='blue')
	plt.plot([0, 25], [0, 25], color='black', linewidth=0.25)
	plt.show()
	return


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

P, Y = inputdata(os.path.join(SCRIPT_DIR, 'train.csv'))
Ptest, Ytest = inputdata(os.path.join(SCRIPT_DIR, 'test.csv'))
Ptest2, Ytest2 = inputdata(os.path.join(SCRIPT_DIR, 'selectivity.csv'))
Ptest3, Ytest3 = inputdata(os.path.join(SCRIPT_DIR, 'cumoexp.csv'))

net = trainNN([7,6,6,6,1], P, Y)
# net = loadNN(os.path.join(SCRIPT_DIR, 'mynetwork.csv'))

saveNN(net, os.path.join(SCRIPT_DIR, 'mynetwork.csv'))

# 计算训练好的神经网络对训练数据和测试数据的输出
y = NNOut(P, net)
evaluate_model('training', Y, y)

ytest = NNOut(Ptest, net)
evaluate_model('test', Ytest, ytest)

ytest2 = NNOut(Ptest2,net)
evaluate_model('CH3O_selectivity', Ytest2, ytest2)

ytest2 = ytest2.reshape(-1,1)
print('CH3O_selectivity:', ytest2)

ytest3 = NNOut(Ptest3,net)
evaluate_model('CumO_exp', Ytest3, ytest3)