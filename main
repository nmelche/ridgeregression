import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def normalize(data):
    x_normed = (data - data.min(axis=0))/ (data.max(axis=0) - data.min(axis=0))
    return x_normed
def split(data):
    train = data.head(int(len(data)*0.8))
    test = data.tail(int(len(data) - len(train)))
    return train, test
def shuffle(data):
    data = data.sample(len(data))
    y = data[data.columns[-1]] 
    x = data
    del x[x.columns[-1]]
    y = np.array(y)
    x = np.array(x)
    return x, y
def rss(x, y, beta):
    m = y.size
    yhead = np.dot(x, beta)
    error = yhead - y
    loss = (1.0/(2*m)) * np.dot(error.T, error)
    return loss
def calc_rmse(predictions, targets):
    rmse = np.sqrt(((predictions-targets)**2).mean())
    return rmse
def turn_neg(ilist):
    neglist = [ -x for x in ilist]
    return neglist
def cross_validation(train, kfold, beta, alpha, batchsize, itera, regu, var):
    #split train set in kfold chunks
    folds = np.array_split(train, kfold)
    rmse_l_validation = []
    clist = []
    counter = 0
    for fold in folds:
        train_set = list(folds) #Put all folds in train
        train_set.pop(counter)
        train_set = pd.concat(train_set) #drop current fold
        validation_set = fold #current fold is validation set
        ilist, rmse_validation_fold = miniBGD(train_set, validation_set, beta, alpha, batchsize, itera, regu, var)
        counter += 1
        rmse_l_validation.append(np.average(rmse_validation_fold))
        clist.append(ilist)
    fold_rmse = np.average(rmse_l_validation) #get average rmse for a hyperparameter set over all kfolds
    return fold_rmse
def next_batch(x, y, batchSize):
	# loop over our dataset `x` in mini-batches of size `batchSize`
	for i in np.arange(0, x.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (x[i:i + batchSize], y[i:i + batchSize])
def miniBGD(train, test, beta, alpha, batchSize, itera, regu, var):
    ilist = []
    losslist= []
    rmse_l_test = []
    rmse_l_train = []
    grhistory = 1
    for i in range(0, itera):
        epochloss = []
        x, y = shuffle(train)
        xtest, ytest = shuffle(test)
        #iterate over our batches
        for (batchX, batchY) in next_batch(x, y, batchSize):
            yhead = np.dot(batchX, beta)
            error = batchY - yhead
            #adagrad
            g = 2*np.dot(-batchX.T, error)/len(batchX)
            grhistory = grhistory + np.multiply(g, g)
            beta = (1-2*alpha*regu)*beta-(alpha/np.sqrt(grhistory))*g
            loss = rss(batchX, batchY, beta)
            epochloss.append(loss)
        #rmseTrain
        yhead = np.dot(x, beta)
        rmse_train = calc_rmse(yhead, y)
        rmse_l_train.append(rmse_train)
        #rmseTest
        yhead_test = np.dot(xtest, beta)
        rmse_test = calc_rmse(yhead_test, ytest)
        rmse_l_test.append(rmse_test)
        ilist.append(i)
        losslist.append(np.average(epochloss))
        ilist_neg = turn_neg(ilist)
    if (var == 1):
        return beta, ilist, ilist_neg, rmse_l_train, rmse_l_test
    else:
        return ilist, rmse_l_test
                
def grid_search(train, k, beta, stepsize, batchsize, itera, lambdaa, var):
    grmse = []
    bestgrmse = []
    curr_min = 0
    for step in stepsize:
        for lam in lambdaa:
            crmse = cross_validation(train, k, beta, step, batchsize, itera, lam, 2)
            grmse.append(crmse)
            old_min = curr_min
            curr_min = min(grmse)
            if (curr_min < old_min):
                bestgrmse.append((curr_min, step, lam))
    return grmse, bestgrmse

data_ww_reg = pd.read_csv('.../winequality-red.csv', sep=";", header=0, names =['fixeda','volatileac', 'citricac', 'resudials', 'chlorides', 'fsd', 'tsd', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])    
ww_reg_norm = normalize(data_ww_reg)
ww_reg_norm.insert(0, 'bias', 1)
train, test = split(ww_reg_norm)
beta = np.zeros(len(train.T)-1)
batchsize = 50
itera = 500
k = 5
stepsize = [0.0001, 0.001, 0.01, 0.1, 0.5]
lambdaa = [0.0001, 0.001, 0.01, 0.1, 0.5]


#--------------------------------------EX1----------------------------------
#alpha 0.0001
beta1, ilist, neg_list, rmse_train, rmse_test = miniBGD(train, test, beta, 0.0001, batchsize, itera, 0.0001, 1) #best RMSE ca 0.46
beta2, ilist2, neg_list2, rmse_train2, rmse_test2 = miniBGD(train, test, beta, 0.0001, batchsize, itera, 0.1, 1)
beta3, ilist3, neg_list3, rmse_train3, rmse_test3 = miniBGD(train, test, beta, 0.0001, batchsize, itera, 5, 1)
plt.title('RMSE with Alpha: 0.001')
line1, = plt.plot(ilist, rmse_train, label="Alpha: 0.001, Lambda: 0.0001, RMSE Train")
line2, = plt.plot(neg_list, rmse_test, label="Alpha: 0.001, Lambda: 0.0001, RMSE Test")
line3, = plt.plot(ilist2, rmse_train2, label="Alpha: 0.001, Lambda: 0.1, RMSE Train")
line4, = plt.plot(neg_list2, rmse_test2, label="Alpha: 0.001, Lambda: 0.1, RMSE Test")
line5, = plt.plot(ilist3, rmse_train3, label="Alpha: 0.001, Lambda: 5, RMSE Train")
line6, = plt.plot(neg_list3, rmse_test3, label="Alpha: 0.001, Lambda: 5, RMSE Test")
plt.figure()
axes = plt.gca()
axes.set_xlim([-itera,itera])
plt.legend(handles=[line1, line2, line3, line4, line5, line6])

#alpha 0.01
beta12, ilist12, neg_list12, rmse_train12, rmse_test12 = miniBGD(train, test, beta, 0.01, batchsize, itera, 0.0001, 1) #best
beta22, ilist22, neg_list22, rmse_train22, rmse_test22 = miniBGD(train, test, beta, 0.01, batchsize, itera, 0.1, 1)
beta33, ilist33, neg_list33, rmse_train33, rmse_test33 = miniBGD(train, test, beta, 0.01, batchsize, itera, 5, 1)
plt.title('RMSE with Alpha: 0.01')
line7, = plt.plot(ilist12, rmse_train12, label="Alpha: 0.01, Lambda: 0.0001, RMSE Train")
line8, = plt.plot(neg_list12, rmse_test12, label="Alpha: 0.01, Lambda: 0.0001, RMSE Test")
line9, = plt.plot(ilist22, rmse_train22, label="Alpha: 0.01, Lambda: 0.1, RMSE Train")
line10, = plt.plot(neg_list22, rmse_test22, label="Alpha: 0.01, Lambda: 0.1, RMSE Test")
line11, = plt.plot(ilist33, rmse_train33, label="Alpha: 0.01, Lambda: 5, RMSE Train")
line12, = plt.plot(neg_list33, rmse_test33, label="Alpha: 0.01, Lambda: 5, RMSE Test")
plt.figure()
axes = plt.gca()
axes.set_xlim([-itera,itera])
plt.legend(handles=[line7, line8, line9, line10, line11, line12])

#alpha 0.5
beta13, ilist13, neg_list13, rmse_train13, rmse_test13 = miniBGD(train, test, beta, 0.5, batchsize, itera, 0.0001, 1) #best
beta23, ilist23, neg_list23, rmse_train23, rmse_test23 = miniBGD(train, test, beta, 0.5, batchsize, itera, 0.1, 1)
beta34, ilist34, neg_list34, rmse_train34, rmse_test34 = miniBGD(train, test, beta, 0.5, batchsize, itera, 5, 1)
plt.title('RMSE with Alpha: 0.5')
line13, = plt.plot(ilist13, rmse_train13, label="Alpha: 0.5, Lambda: 0.0001, RMSE Train")
line14, = plt.plot(neg_list13, rmse_test13, label="Alpha: 0.5, Lambda: 0.0001, RMSE Test")
line15, = plt.plot(ilist23, rmse_train23, label="Alpha: 0.5, Lambda: 0.1, RMSE Train")
line16, = plt.plot(neg_list23, rmse_test23, label="Alpha: 0.5, Lambda: 0.1, RMSE Test")
line17, = plt.plot(ilist34, rmse_train34, label="Alpha: 0.5, Lambda: 5, RMSE Train")
line18, = plt.plot(neg_list34, rmse_test34, label="Alpha: 0.5, Lambda: 5, RMSE Test")
plt.figure()
axes = plt.gca()
axes.set_xlim([-itera,itera])
plt.legend(handles=[line13, line14, line15, line16, line17, line18])

#compare best of each combination
beta1, ilist, neg_list, rmse_train, rmse_test = miniBGD(train, test, beta, 0.0001, batchsize, itera, 0.0001, 1)
beta12, ilist12, neg_list12, rmse_train12, rmse_test12 = miniBGD(train, test, beta, 0.01, batchsize, itera, 0.0001, 1)
beta13, ilist13, neg_list13, rmse_train13, rmse_test13 = miniBGD(train, test, beta, 0.5, batchsize, itera, 0.0001, 1)
plt.title('RMSE | Compare best combinations')
line19, = plt.plot(ilist, rmse_train, label="Alpha: 0.001, Lambda: 0.0001, RMSE Train")
line20, = plt.plot(neg_list, rmse_test, label="Alpha: 0.001, Lambda: 0.0001, RMSE Test")
line21, = plt.plot(ilist12, rmse_train12, label="Alpha: 0.01, Lambda: 0.0001, RMSE Train")
line22, = plt.plot(neg_list12, rmse_test12, label="Alpha: 0.01, Lambda: 0.0001, RMSE Test")
line23, = plt.plot(ilist13, rmse_train13, label="Alpha: 0.5, Lambda: 0.0001, RMSE Train")
line44, = plt.plot(neg_list13, rmse_test13, label="Alpha: 0.5, Lambda: 0.0001, RMSE Test")
plt.figure()
axes = plt.gca()
axes.set_xlim([-itera,itera])
plt.legend(handles=[line19, line20, line21, line22, line23, line44])
#--------------------------------END EX1-----------------------------
#---------------------------------EX2--------------------------------
grmse, bestgrmse = grid_search(train, k, beta, stepsize, batchsize, itera, lambdaa, 2)
#best set is stepsize 0.1, lambda 0.0001
x, y = np.meshgrid(stepsize, lambdaa)
z = np.array(grmse)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('Stepsize')
ax.set_ylabel('Lambdaa')
ax.set_zlabel('RMSE')
ax.view_init(azim=15)
plt.show()
#use best set
beta_best, ilist_best, neg_list_best, rmse_train_best, rmse_test_best = miniBGD(train, test, beta, 0.1, batchsize, itera, 0.0001, 1)
plt.title('RMSE of best Hyperparameter set')
line1_b, = plt.plot(ilist_best, rmse_train_best, label="Alpha: 0.1, Lambda: 0.0001, RMSE Train")
line2_b, = plt.plot(neg_list_best, rmse_test_best, label="Alpha: 0.1, Lambda: 0.0001, RMSE Test")
axes = plt.gca()
axes.set_xlim([-itera,itera])
plt.legend(handles=[line1_b, line2_b])
plt.show()
