from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

"""
Calculate the FPR, TPR and  Threshold.
"""
# y-label is the sample, 1=Positive  0=Negative, these has 10 samples
y_label = ([1, 1, 0, 1, 1, 0, 0, 0, 1, 0])  # 非二进制需要pos_label
# y_pre is the Predicted probability which is to the sample, also has 10 numbers.
y_pre = ([0.95, 0.86, 0.7, 0.65, 0.55, 0.53, 0.52, 0.43, 0.42, 0.35])
# thersholds is the Threshold which is to the y_pre and sample
fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=1)

for i, value in enumerate(thersholds):
    print("%f %f %f" % (fpr[i], tpr[i], value))

roc_auc = auc(fpr, tpr)  # easy get auc by sklearn.metrics lib

"""
Draw a picture by plt.
"""
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
# limit the x-line and y-line for more observe the curve
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
