import kaggle

# Assuming you are running run_me.py from the Submission/Code
# directory, otherwise the path variable will be different for you
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.cross_validation import KFold, ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import copy



'''
Below loads the robot data
'''

def loadRobotData():
	path = '../../Data/RobotArm/'
	data = np.load(path + 'Data.npz')
	features_train = data['X_train']
	labels_train = data['y_train']
	features_test = data['X_test']
	labels_test = data['y_test']
	print "RobotArm:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 
	print(features_train[0])
	print(labels_train[0])
	print(features_test[0])
	print(labels_test[0])
	return features_train,labels_train,features_test,labels_test
'''
I wrote the function below to look at the interaction term i was interested 
in analyzing for the robot data
'''
def addInteractionTerm(features_train,features_test,col1,col2):
	jointInteract=features_train[:,col1]*features_train[:,col2]
	jointInteractTest=features_test[:,col1]*features_test[:col2]
	features_train=np.hstack((features_train,np.transpose(np.array([jointInteract]))))
	features_test=np.hstack((features_test,np.transpose(np.array([jointInteractTest]))))
	return features_train, features_test


'''
Below is my Ridge and Lasso experiment on the Robot Data
'''

def robotRidgeLassoExp():
	features_train,labels_train,features_test,labels_test=loadRobotData()
	pipeFeat=[('addFeatures',PolynomialFeatures(2)),
				('feat_select',SelectKBest(f_regression,k=10)),
				('clf',Ridge(alpha=1.0))]
	gridParam=dict(feat_select__k=range(1,40,5),
				clf=[Ridge(),Lasso()],
				clf__alpha=np.arange(3,8,0.2))
	pipe=Pipeline(pipeFeat)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=5)
	gridCV.fit(features_train,labels_train)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	predict=gridCV.predict(features_test)
	kaggle.kaggleize(predict, "../Predictions/RobotArm/GridCVRidgeLassoTest.csv")
	return gridCV.grid_scores_

'''
Below is my tree regression Expirament with the robot data
'''

def robotTreeRegExp():
	print "Starting Robot Tree Reg Expirament"
	features_train,labels_train,features_test,labels_test=loadRobotData()
	pipeFeat=[('addFeatures',PolynomialFeatures(2,interaction_only=False)),
			('feat_select',SelectKBest(f_regression,k=10)),
			('clf',DecisionTreeRegressor())]
	gridParam=dict(
				feat_select__k=range(1,10),
				clf__max_depth=range(1,10)
				)
	pipe=Pipeline(pipeFeat)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=3)
	gridCV.fit(features_train,labels_train)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(features_test)
	kaggle.kaggleize(predict, "../Predictions/RobotArm/GridCVTreeReg1.csv")
	return gridCV.grid_scores_

'''
Below is my KNN regression experiment on the robot data. This ended up being the
most accurate:
Best Params: {'clf__weights': 'uniform', 'clf__metric': 'euclidean', 'feat_select__k': 3, 'clf__n_neighbors': 42}
'''

def robotKNNRegExp():
	print "Starting Robot KNN Reg Expirament"
	features_train,labels_train,features_test,labels_test=loadRobotData()
	pipeFeat=[('addFeatures',PolynomialFeatures(2,interaction_only=False)),
			('feat_select',SelectKBest(f_regression,k=10)),
			('clf',KNeighborsRegressor())]
	gridParam=dict(
				feat_select__k=range(1,10),
				clf__weights=['uniform','distance'],
				clf__n_neighbors=range(1,100),
				clf__metric=['euclidean','minkowski','manhattan','chebyshev'])
	pipe=Pipeline(pipeFeat)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=3)
	gridCV.fit(features_train,labels_train)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(features_test)
	kaggle.kaggleize(predict, "../Predictions/RobotArm/GridCVKNNRegTest9.csv")
	return gridCV.grid_scores_
	'''
	Best Params: {'clf__weights': 'uniform', 'clf__metric': 'euclidean', 'feat_select__k': 3, 'clf__n_neighbors': 42}
	'''
'''
Below is my SVR Expirament on the robot data
'''

def robotSVRExp():
	print "Starting Robot SVR Reg Expirament"
	features_train,labels_train,features_test,labels_test=loadRobotData()
	pipeFeat=[('addFeatures',PolynomialFeatures(2,interaction_only=False)),
			('feat_select',SelectKBest(f_regression,k=10)),
			('clf',SVR())]
	gridParam=dict(
				clf__C=[.001,.01,.1,1,10,100,1000,10000],
				#clf__C=[1,2,3,4,5,6,7,8,9,10]
				clf__epsilon=[0,0.05,0.1,.2,.25],
				clf__kernel=['poly'],
				feat_select__k=range(1,40))
	pipe=Pipeline(pipeFeat)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=3)
	gridCV.fit(features_train,labels_train)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(features_test)
	kaggle.kaggleize(predict, "../Predictions/RobotArm/GridCVSVR1.csv")
	return gridCV.grid_scores_

'''
Below is my LinearSVR Expirament on the robot data
'''

def robotLinearSVRExp():
	print "Starting Robot LinearSVR Reg Expirament"
	features_train,labels_train,features_test,labels_test=loadRobotData()
	pipeFeat=[('addFeatures',PolynomialFeatures(2,interaction_only=False)),
			('feat_select',SelectKBest(f_regression,k=10)),
			('clf',LinearSVR())]
	gridParam=dict(
				clf__C=[.001,.01,.1,1,10,100,1000,10000],
				#clf__C=[1,2,3,4,5,6,7,8,9,10]
				clf__loss=['epsilon_insensitive','squared_epsilon_insensitive'],
				clf__epsilon=[0,0.05,0.1,.2,.25],
				feat_select__k=range(35,45))
	pipe=Pipeline(pipeFeat)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=3)
	gridCV.fit(features_train,labels_train)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(features_test)
	kaggle.kaggleize(predict, "../Predictions/RobotArm/GridCVLinearSVR4.csv")
	return gridCV.grid_scores_

'''
Below is my Gradient Boosting Regression Expirament on the robot data
'''
def robotGradienBoostRegxp():
	print "Starting Robot Gradient Boost Reg Expirament"
	features_train,labels_train,features_test,labels_test=loadRobotData()
	pipeFeat=[('addFeatures',PolynomialFeatures(3,interaction_only=False)),
			('feat_select',SelectKBest(f_regression,k=10)),
			('clf',GradientBoostingRegressor())]
	gridParam=dict(
				clf__loss=['ls'],
				clf__learning_rate=[.04,.05],
				clf__n_estimators=[150,185,186],
				feat_select__k=[8,10,11])
	pipe=Pipeline(pipeFeat)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=3)
	gridCV.fit(features_train,labels_train)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(features_test)
	kaggle.kaggleize(predict, "../Predictions/RobotArm/GridCVGradientBoost.csv")
	return gridCV.grid_scores_




'''
Below is the code to make the 
Robot Scatter matrix
'''



def makeRobotScatterMatrix():
	print "Making Robot Scatter Matrix"
	features_train,labels_train,features_test,labels_test=loadRobotData()
	varNames=["joint1Pos","joint2Pos","joint3Pos","joint1AngVel","joint2AngVel","joint3AngVel","joint1torque","joint2torque"]
	df=pd.DataFrame(features_train,columns=varNames)
	df["joint3angaccel"]=labels_train
	label=pd.DataFrame(labels_train)
	sm=pd.tools.plotting.scatter_matrix(df, alpha=0.05, figsize=(15, 15), diagonal='kde')
	plt.plot()
	#plt.show()
	plt.savefig("../Figures/robotCorrelationMatrix4")
	print "Complete"
	return


'''
Below is the code to preform ols on the robot data
I used pandas because I found the output to be the most
sensible 
'''
def makeRobotOls():
	print "Making Robot Ols"
	features_train,labels_train,features_test,labels_test=loadRobotData()
	varNames=["joint1Pos","joint2Pos","joint3Pos","joint1AngVel","joint2AngVel","joint3AngVel","joint1torque","joint2torque","j1j2int"]
	jointInteract=features_train[:,1]*features_train[:,2]
	jointInteractTest=features_test[:,1]*features_test[:,2]
	features_train=np.hstack((features_train,np.transpose(np.array([jointInteract]))))
	features_test=np.hstack((features_test,np.transpose(np.array([jointInteractTest]))))
	df=pd.DataFrame(features_train,columns=varNames)
	df["joint3angaccel"]=labels_train
	df=pd.DataFrame(features_train,columns=varNames)
	label=pd.DataFrame(labels_train)
	#sm=pd.tools.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')
	print df.corr()
	res=pd.stats.api.ols(y=label[:][0],x=df[:])
	print res
	#plt.plot()
	#plt.show()
	#plt.savefig("../Figures/robotCorrelationMatrix4")
	print "Complete"
	return
'''
Below is the code to make the bank ols report
'''

def makeBankOls():
	print "Making Bank Ols"
	features_train,labels_train,features_test,labels_test=loadBankData()
	df=pd.DataFrame(features_train)
	df["waitTime"]=labels_train
	df=pd.DataFrame(features_train)
	label=pd.DataFrame(labels_train)
	#print df.corr()
	res=pd.stats.api.ols(y=label[:][0],x=df[:])
	print res
	print "Complete"
	return

'''
Below makes the KNN regression plots
I looked at CV accuracy over a k of 1 to 100 for each distance metric

'''
def plotKnnRobot(rbtKnnRegGrid):
	nneighbors=range(1,100)
	metric=['euclidean','minkowski','manhattan','chebyshev']
	scores=[feat[1] for feat in rbtKnnRegGrid]
	#scores=np.array(scores).reshape(len(nneighbors),len(metric))
	newMatrix=np.zeros((len(rbtKnnRegGrid),5),dtype=object)
	for i in range(len(rbtKnnRegGrid)):
		newMatrix[i,0]=rbtKnnRegGrid[i][1]
		newMatrix[i,1]=rbtKnnRegGrid[i][0]["feat_select__k"]
		newMatrix[i,2]=rbtKnnRegGrid[i][0]["clf__weights"]
		newMatrix[i,3]=rbtKnnRegGrid[i][0]["clf__metric"]
		newMatrix[i,4]=rbtKnnRegGrid[i][0]["clf__n_neighbors"]
	for met in metric:
		scoreArray=[]
		kneighbors=[]
		for i in range(len(rbtKnnRegGrid)):
			if newMatrix[i,1]==3 and newMatrix[i,2]=='uniform' and newMatrix[i,3]==met:
				scoreArray.append(newMatrix[i,0])
				kneighbors.append(newMatrix[i,4])
		plt.figure(1,figsize=(6,4))
		plt.plot(kneighbors,scoreArray,'sb-', linewidth=3)
		#plt.plot(kList,testErrList,'sb-', linewidth=3)

		plt.grid(True) #Turn the grid on
		plt.ylabel("Accuracy") #Y-axis label
		plt.xlabel("k Value") #X-axis label
		plt.title(met+" distance weights Error vs k Value") #Plot title
		plt.xlim(35,50+.1) #set x axis range
		plt.ylim(.946,.95) #Set yaxis range
		plt.legend(["3 Fold CV Accuracy"],loc="best")

		#Make sure labels and titles are inside plot area
		plt.tight_layout()

		#Save the chart
		plt.savefig("../Figures/"+"rbtKnn_"+met+"_line_plot.pdf")
		plt.clf()




def saveToPickle(dict,name):
	pickle.dump(dict,open(name,"wb"))
	return

def loadFromPickle(filename):
	dictLoad=pickle.load(open(filename,"rb"))
	return copy.deepcopy(dictLoad)
'''
The code below Loads the bank data
'''

def loadBankData():
	path = '../../Data/BankQueues/'
	data = np.load(path + 'Data.npz')
	features_train = data['X_train']
	labels_train = data['y_train']
	features_test = data['X_test']
	labels_test = data['y_test']
	print "BankQueues:", features_train.shape, labels_train.shape, features_test.shape, labels_test.shape 
	print(features_train[0])
	print(labels_train[0])
	print(features_test[0])
	print(labels_test[0])
	return features_train, labels_train, features_test, labels_test

'''
The code below is for my bank ridge lasso Expirament
'''


def bankRidgeLassoExp():
	print "Starting bank Ridge Lasso Expirament"
	features_train,labels_train,features_test,labels_test=loadBankData()
	pipeFeat=[('addFeatures',PolynomialFeatures(2)),
				('feat_select',SelectKBest(f_regression,k=10)),
				('clf',Ridge(alpha=1.0))]
	gridParam=dict(feat_select__k=['all'],
				clf=[Ridge(),Lasso()],
				clf__alpha=range(1,100))
	pipe=Pipeline(pipeFeat)
	gridCV=GridSearchCV(pipe,param_grid=gridParam,cv=5)
	gridCV.fit(features_train,labels_train)
	print "Best Estimator: "+str(gridCV.best_estimator_)
	print "Best Params: "+ str(gridCV.best_params_)
	print "Score: "+str(gridCV.best_score_)
	predict=gridCV.predict(features_test)
	kaggle.kaggleize(predict, "../Predictions/BankQueues/GridCVRidgeLassoTest12.csv")
	return gridCV.grid_scores_

'''
Below is the code to make the bank scatter_matrix.
Note that this takes a very long time, so I will not include a run of this in the final
code submission

'''

def makeBankScatterMatrix():
	print "Making bank scatter matrix"
	features_train,labels_train,features_test,labels_test=loadBankData()
	df=pd.DataFrame(features_train)
	df["RejRate"]=labels_train
	sm=pd.tools.plotting.scatter_matrix(df, alpha=0.01, figsize=(20, 20), diagonal='kde')
	plt.plot()
	#plt.show()
	plt.savefig("../Figures/BankCorrelationMatrix2")
	print "Complete"
	return
'''
below is the code to make my bank plots

'''
def plotBankRidge(bankRidgeGrid):
	clf=[type(Ridge()),type(Lasso())]
	alpha=range(1,100)
	feat_select__k=[200,300,400,'all']
	#metric=['euclidean','minkowski','manhattan','chebyshev']
	scores=[feat[1] for feat in bankRidgeGrid]
	#scores=np.array(scores).reshape(len(nneighbors),len(metric))
	newMatrix=np.zeros((len(bankRidgeGrid),4),dtype=object)
	typeToString=lambda x: "Ridge" if type(x)==type(Ridge()) else "Lasso"
	for i in range(len(bankRidgeGrid)):
		newMatrix[i,0]=bankRidgeGrid[i][1]
		newMatrix[i,1]=bankRidgeGrid[i][0]["feat_select__k"]
		newMatrix[i,2]=typeToString(bankRidgeGrid[i][0]["clf"])
		newMatrix[i,3]=bankRidgeGrid[i][0]["clf__alpha"]
		#newMatrix[i,4]=rbtKnnRegGrid[i][0]["clf__n_neighbors"]
	ridgeScoreArray=[]
	ridgeAlpha=[]
	lassoScoreArray=[]
	lassoAlpha=[]
	for i in range(len(bankRidgeGrid)):
		if newMatrix[i,1]=='all' and newMatrix[i,2]=='Ridge':
				ridgeScoreArray.append(newMatrix[i,0])
				ridgeAlpha.append(newMatrix[i,3])
		if newMatrix[i,1]=='all' and newMatrix[i,2]=='Lasso':
				lassoScoreArray.append(newMatrix[i,0])
				lassoAlpha.append(newMatrix[i,3])
	plt.figure(1,figsize=(6,4))
	plt.plot(ridgeAlpha,ridgeScoreArray,'b-', linewidth=.5)
	#plt.plot(lassoAlpha,lassoScoreArray,'r-', linewidth=.1)

	plt.grid(True) #Turn the grid on
	plt.ylabel("Accuracy") #Y-axis label
	plt.xlabel("alpha value") #X-axis label
	plt.title("Ridge CV Accuracy by Alpha Value") #Plot title
	plt.xlim(1,100) #set x axis range
	plt.ylim(.946,.948) #Set yaxis range
	plt.legend(["Ridge 3-Fold CV Accuracy"],loc="best")

	#Make sure labels and titles are inside plot area
	plt.tight_layout()

	#Save the chart
	plt.savefig("../Figures/"+"bankRidgeLasso_line_plot.pdf")
	plt.clf()



# makeRobotScatterMatrix()
# makeRobotOls()

# saveToPickle(copy.deepcopy(rbtKnnRegGrid),'robotKnnRegGrid.p')
'''
Make Exhibits
'''
#makeRobotScatterMatrix()
#makeRobotOls()
#makeBankOls()

'''
Uncomment below to make Bank scatter_matrix
'''
makeBankScatterMatrix()

'''
Uncomment below to run the KNNReg Expirament on the Robot Data
'''
rbtKnnRegGrid=robotKNNRegExp()
rbtKnnRegGrid=loadFromPickle('robotKnnRegGrid.p')
plotKnnRobot(rbtKnnRegGrid)

'''
Uncomment below to run the Ridge Lasso Expirament on the 
'''
bankRidgeLassoGrid=bankRidgeLassoExp()
bankRidgeGrid=loadFromPickle('bankRidgeLassoGrid.p')
plotBankRidge(bankRidgeGrid)


'''
Uncomment the 4 lines below to run the SVR, tree regression, and GradientBoostingRegressor 
experiments on the Robot data

'''
rbtLinearSVRegGrid=robotLinearSVRExp()
rbtSVRegGrid=robotSVRExp()
rbtTreeRegGrid=robotTreeRegExp()
rbtGBRgrid=robotGradienBoostRegxp()

