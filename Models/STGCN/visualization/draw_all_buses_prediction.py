import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import matplotlib.ticker as ticker
from utils.math_utils import z_inverse
from utils.math_utils import RMSE


def draw_all_buses_prediction(y, y_, x_stats, p_start_t, load_path,n_pred):

    dim = len(y_.shape)


    v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])

    full_ground_truth = y
    full_prediction = v_[:, :, 0]


    predict_value = np.swapaxes(full_prediction, 1, 0)
    print(predict_value.shape)

    ground_value = np.swapaxes(full_ground_truth, 1, 0)

    cores_ground_value = ground_value[:,p_start_t:p_start_t+n_pred]
    core_ground = ground_value[:,500:p_start_t+n_pred]


    rmse_1 = RMSE(cores_ground_value, predict_value)
    print("rmse value is %13.8f" % rmse_1)
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')

    x_1 = np.arange(0, 500+n_pred, 1)
    for i in range(ground_value.shape[0]):
        plt.plot(x_1, core_ground[i], color='blue', label='Ground truth')
        plt.plot(x_1[500:500+n_pred], predict_value[i], '--', linewidth=2, color='red', label='Predicted using STGCN')


    handles, labels = plt.gca().get_legend_handles_labels()


    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    print(newLabels)
    print(newHandles)
    plt.legend(newHandles, newLabels, loc='best')
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
 
    ax.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax.set_xticklabels(['10', '15', '20', '25', '30', '35'])
    plt.show()

    return    rmse_1