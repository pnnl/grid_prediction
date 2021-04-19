import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from utils.math_utils import z_inverse
from utils.math_utils import RMSE,MAE

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    print(idx)
    return idx



def draw_actual_vs_prediction(y, y_, x_stats,p_start_t,load_path,x_stat,epoch,scenario):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)
    
    x_mean = x_stat


    #ground_truth
    v = z_inverse(y, x_stats['mean'], x_stats['std'])
    x_mean = np.tile(x_mean,(v.shape[0],1))

    x_mean = x_mean.reshape((x_mean.shape[0],x_mean.shape[1],1))
    
    v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
   
    full_ground_truth = v[:,:,0]
    full_prediction = v_[:,:,0]

    if x_stat != 0:
        print("saving the prediction and ground truth results")
        np.savetxt(load_path+scenario+"_"+str(full_prediction.shape[0])+"_"+"ground_truth.csv", full_ground_truth, delimiter=",")
        np.savetxt(load_path+scenario+"_"+str(full_prediction.shape[0])+'_'+"prediction.csv", full_prediction, delimiter=",")
        print("saving is finished")
    
    ground_value = np.swapaxes(full_ground_truth,1,0)
    predict_value = np.swapaxes(full_prediction,1,0)
    print(predict_value.shape)
    
    rmse_1 = RMSE(ground_value, predict_value)
    print("rmse value is %13.8f" %rmse_1)

    mae_1 = MAE(ground_value, predict_value)
    print("mae value is %13.8f" % mae_1)


    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.title('prediction vs. ground truth in next {}'.format(predict_value.shape[1]) +'timestamps')
    plt.xlabel('future timestamps')
    plt.ylabel('frequency')
    
    
    x = pylab.linspace(0, predict_value.shape[1], predict_value.shape[1])
    
    for i in range(ground_value.shape[0]):
        plt.plot(x, ground_value[i],color='blue',label='ground truth')
        plt.plot(x, predict_value[i], '--',linewidth=2, color='red', label='prediction')
    
    handles, labels = plt.gca().get_legend_handles_labels()

    
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels,loc='best')
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    
    if x_stat ==0:
        plt.savefig(load_path+'figures/'+"results_on_"+scenario+"_{}_{}_epoch{}.png".format(p_start_t, predict_value.shape[1],epoch), dpi=600)
    else:
        plt.savefig(
            load_path + 'figures/' + "test_results_on_"+scenario+"_{}_{}_epoch{}.png".format(p_start_t, predict_value.shape[1], epoch),
            dpi=600)
        print("The testing figures is ::::::::::::")
        print("###########################")
    plt.show()


