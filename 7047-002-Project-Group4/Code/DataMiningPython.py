import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn import preprocessing



bcdata = pd.read_csv('C:/Users/Jacob Blizzard/Desktop/Data Mining/BreastCancerWisconsin.csv')
bcdata = pd.DataFrame(bcdata)

# print(bcdata.info())

xdata = bcdata[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
              'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']]
ydata = bcdata[['Class']]
# print(xdata)
# print(ydata)


xdata = xdata.fillna(xdata.mean())
print(xdata.info())

ydata = ydata.replace([2,4],[0,1])
# print(ydata)
# print(xdata.info())

train = pd.read_csv('C:/Users/Jacob Blizzard/Desktop/Data Mining/CancerTrain.csv')
test = pd.read_csv('C:/Users/Jacob Blizzard/Desktop/Data Mining/CancerTest.csv')
train = train.fillna(train.mean())
test = test.fillna(test.mean())
xtrain = train[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
              'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']]
ytrain = train[['Class']]

xtest = test[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
              'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']]

ytest = test[['Class']]
# xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.25, random_state=123)


classtree = tree.DecisionTreeClassifier()
ctree = classtree.fit(xtrain, ytrain)

dot_data = StringIO('C:/Users/Jacob Blizzard/Desktop/Data Mining/')
export_graphviz(ctree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=xdata.columns.values)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
print(Image(graph.create_png()))
graph.write_pdf('Breast Cancer.pdf')


classtree2 = tree.DecisionTreeClassifier(max_depth = 5, min_samples_leaf=2,
                                         max_leaf_nodes=10)
ctree2 = classtree2.fit(xtrain,ytrain)
dot_data2 = StringIO('C:/Users/Jacob Blizzard/Desktop/Data Mining/')
export_graphviz(ctree2, out_file=dot_data2,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=xdata.columns.values)
graph2 = pydotplus.graph_from_dot_data(dot_data2.getvalue())
graph2.write_pdf('Breast Cancer2.pdf')

# Third Tree Made
classtree3 = tree.DecisionTreeClassifier(max_leaf_nodes=7, min_samples_leaf=7)
ctree3 = classtree3.fit(xtrain, ytrain)
dot_data3 = StringIO('C:/Users/Jacob Blizzard/Desktop/Data Mining/')
export_graphviz(ctree3, out_file=dot_data3,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=xdata.columns.values)
graph3 = pydotplus.graph_from_dot_data(dot_data3.getvalue())
graph3.write_pdf('Breast Cancer3.pdf')

# In Sample Prediction For Trees
yinsamp = classtree.predict(xtrain)
yinsamp2 = classtree2.predict(xtrain)
yinsamp3 = classtree3.predict(xtrain)

# Out of Sample Prediction for Trees
ypred = classtree.predict(xtest)
ypred2 = classtree2.predict(xtest)
ypred3 = classtree3.predict(xtest)

# Confusion Matrix for Trees
print('In Sample First Tree')
print(confusion_matrix(ytrain, yinsamp))

print('In Sample Second Tree')
print(confusion_matrix(ytrain, yinsamp2))

print('In Sample Third Tree')
print(confusion_matrix(ytrain, yinsamp3))

print('First Decision Tree Testing Sample')
print(confusion_matrix(ytest, ypred))

print('Second Decision Tree Testing Sample')
print(confusion_matrix(ytest, ypred2))

print('Third Decision Tree Testing Sample')
print(confusion_matrix(ytest, ypred3))

# First Tree ROC Curve
# calculate the fpr and tpr for all thresholds of the classification
probs = ctree.predict_proba(xtest)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(ytest, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic\n Out of Sample')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Second Tree ROC Curve
# calculate the fpr and tpr for all thresholds of the classification
probs2 = ctree2.predict_proba(xtest)
preds2 = probs2[:,1]
fpr2, tpr2, threshold2 = metrics.roc_curve(ytest, preds2)
roc_auc = metrics.auc(fpr2, tpr2)

# method I: plt
plt.title('Receiver Operating Characteristic\n Out of Sample')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Third Tree ROC Curve
probs3 = ctree3.predict_proba(xtest)
preds3 = probs3[:,1]
fpr3, tpr3, threshold3 = metrics.roc_curve(ytest, preds3)
roc_auc = metrics.auc(fpr3, tpr3)

# method I: plt
plt.title('Receiver Operating Characteristic\n Out of Sample')
plt.plot(fpr3, tpr3, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# First Tree ROC Curve IN SAMPLE
# calculate the fpr and tpr for all thresholds of the classification
probs = ctree.predict_proba(xtrain)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(ytrain, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic\n In Sample')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Second Tree ROC Curve IN SAMPLE
# calculate the fpr and tpr for all thresholds of the classification
probs2 = ctree2.predict_proba(xtrain)
preds2 = probs2[:,1]
fpr2, tpr2, threshold2 = metrics.roc_curve(ytrain, preds2)
roc_auc = metrics.auc(fpr2, tpr2)

# method I: plt
plt.title('Receiver Operating Characteristic\n In Sample')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Third Tree ROC Curve
probs3 = ctree3.predict_proba(xtrain)
preds3 = probs3[:,1]
fpr3, tpr3, threshold3 = metrics.roc_curve(ytrain, preds3)
roc_auc = metrics.auc(fpr3, tpr3)

# method I: plt
plt.title('Receiver Operating Characteristic\n In Sample')
plt.plot(fpr3, tpr3, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#
# SUPPORT VECTOR MACHINES
ydatanp = np.asarray(ytrain)
ydatanp = ydatanp.squeeze()
xscale = preprocessing.scale(xtrain)
xscaletest = preprocessing.scale(xtest)

# print(ydatanp)
svmclass = svm.SVC(kernel='rbf', gamma='auto', probability=True)
svmmodel = svmclass.fit(xscale, ydatanp)
# print(svmmodel.get_params)

svmmodel.predict(xscale)
# In Sample Prediction For Trees
yinsampsvm = svmmodel.predict(xscale)
# yinsamp2 = classtree2.predict(xtrain)
# yinsamp3 = classtree3.predict(xtrain)

# Out of Sample Prediction for Trees
ypredsvm = svmmodel.predict(xscaletest)
# ypred2 = classtree2.predict(xtest)
# ypred3 = classtree3.predict(xtest)

# Confusion Matrix for SVM
print('In Sample First SVM')
print(confusion_matrix(ytrain, yinsampsvm))
print('Out of Sample First SVM')
print(confusion_matrix(ytest, ypredsvm))

# First SVM ROC Curve
probs4 = svmmodel.predict_proba(xscale)
preds4 = probs4[:,1]
fpr4, tpr4, threshold4 = metrics.roc_curve(ytrain, preds4)
roc_auc = metrics.auc(fpr4, tpr4)

# method I: plt
plt.title('Receiver Operating Characteristic\n In Sample\n SVM')
plt.plot(fpr4, tpr4, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# First SVM Tree ROC Curve
probs5 = svmmodel.predict_proba(xscaletest)
preds5 = probs5[:,1]
fpr5, tpr5, threshold5 = metrics.roc_curve(ytest, preds5)
roc_auc = metrics.auc(fpr5, tpr5)

# method I: plt
plt.title('Receiver Operating Characteristic\n Out of Sample\n SVM')
plt.plot(fpr5, tpr5, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# #Visualizing SVC
# def make_meshgrid(x, y, h=.02):
#     """Create a mesh of points to plot in
#
#     Parameters
#     ----------
#     x: data to base x-axis meshgrid on
#     y: data to base y-axis meshgrid on
#     h: stepsize for meshgrid, optional
#
#     Returns
#     -------
#     xx, yy : ndarray
#     """
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     return xx, yy
#
#
# def plot_contours(ax, clf, xx, yy, **params):
#     """Plot the decision boundaries for a classifier.
#
#     Parameters
#     ----------
#     ax: matplotlib axes object
#     clf: a classifier
#     xx: meshgrid ndarray
#     yy: meshgrid ndarray
#     params: dictionary of params to pass to contourf, optional
#     """
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out
#
# # Take the first two features. We could avoid this by using a two-dim dataset
# xscaled = preprocessing.scale(xdata)
# xdatanp = np.asarray(xscaled)
# X = xdatanp[:, [2, 5]]
# ydatanp2 = np.asarray(ydata)
# y = ydatanp2.squeeze()
#
# # we create an instance of SVM and fit out data. We do not scale our
# # data since we want to plot the support vectors
# C = 1.0  # SVM regularization parameter
# models = (svm.SVC(kernel='linear', C=C),
#           svm.SVC(kernel='rbf', gamma='auto', C=C, decision_function_shape='ovo'),
#           svm.SVC(kernel='sigmoid', gamma='auto', C=C, decision_function_shape='ovr'),
#           svm.SVC(kernel='poly', degree=3, C=C))
# models = (clf.fit(X, y) for clf in models)
#
# # title for the plots
# titles = ('SVC with linear kernel',
#           'SVC with RBF kernel',
#           'SVC with Sigmoid kernel',
#           'SVC with polynomial (degree 3) kernel')
#
# # Set-up 2x2 grid for plotting.
# fig, sub = plt.subplots(2, 2)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)
#
# for clf, title, ax in zip(models, titles, sub.flatten()):
#     plot_contours(ax, clf, xx, yy,
#                   cmap=plt.cm.coolwarm, alpha=0.8)
#     ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel('Uniformity of Cell Shape')
#     ax.set_ylabel('Bare Nuclei')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(title)
#
# plt.show()
