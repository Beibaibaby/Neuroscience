#!/usr/bin/env python
# coding: utf-8

# ### 定义函数

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numba import jit


def RGCrf(
    
    N: int, 
    sigma: float, 
    K: float
    
) -> np.array:
    
    """
    N: receptive field width.
    sigma: Gaussian distribution standard deviation.
    K: a coefficient.
    """
    
    domain = range(-int((N-1)/2), int((N-1)/2 + 1))
    X_grid, Y_grid = np.meshgrid(domain, domain)
    Y_grid = -Y_grid
    
    
    R = (X_grid ** 2 + Y_grid ** 2) ** 0.5
    
    FR = K / (2*np.pi*sigma**2) * np.exp(- R**2 / (2*sigma**2))
    
    return FR



def GenerateBar(
    
    N:int, 
    theta:float, 
    t_len:int,
    contrast:float=10,
    bar_length:int=3, 
    bar_width:int=1,
    
):
    """
    Attributes:
    
    N:              RGC number
    theta:          bar orientation
    t_len:          total time
    bar_length:     length of the rectangular bar, unit degree, 1 degree = 72 pixel
    bar_width:      width of the rectangular bar, unit degree, 1 degree = 72 pixel
    
    """
    
    bar_length = bar_length * 60
    bar_width = bar_width * 60
    
    
    M = 60 + (N-1) * 12 + N  # image size
    rectangular = np.ones((t_len,M,M)) * contrast
    
    domain = range(-int((M-1)/2), int((M-1)/2 + 1))
    X_grid, Y_grid = np.meshgrid(domain, domain)
    Y_grid = -Y_grid
    
    
    length_ori = abs( np.cos(theta)*Y_grid - np.sin(theta)*X_grid )
    width_ori = abs( np.sin(theta)*Y_grid + np.cos(theta)*X_grid )
    field = (length_ori <= bar_length/2) * (width_ori <= bar_width/2)  # inside 'field' is the area of a rectangular bar, whose gray value will be 0.
    
    rectangular[:, field] = 0
    
    return rectangular



def SpatialTemporalFilter(
    
    N: int,
    image: np.array,
    s_kernel: np.array,
    t_len: int,
    dt: float,
    tau: float,   
    
):
    
    s_filtered = np.zeros((t_len, N, N))
    
    for t in range(t_len):
        image_t = image[t,:,:]    # at time point t
        for i in range(N):
            for j in range(N):    # the [i, j](row i and column j. total 21 rows and 21 columns) RGC
                i_ = 30 + 13 * i  # [i_,j_] is the coordinate of center of the region of interest in the image.
                j_ = 30 + 13 * j
                ROI = image_t[i_ - 30:i_ + 31, j_ - 30:j_ + 31]  # region of interest for spatial convolution of the [i,j] RGC
                s_filtered[t, i, j] = np.sum(ROI * s_kernel)  # doing spatial convolution
    
    st_filtered = np.zeros((t_len, N, N))
    
    for t in range(t_len):
        if t == 0:
            st_filtered[0,:,:] = s_filtered[0,:,:]
        else:
            st_filtered[t,:,:] = st_filtered[t-1,:,:] * np.exp(-dt / tau) + s_filtered[t,:,:] * dt / tau  # doing temporal convolution
               
    return st_filtered



def ON_OFF_RGCResponse(
    
    center_R: np.array, 
    surround_R: np.array,
    baseline: np.array,
    tau_c:float,
    tau_s:float,
    delta:int=int(3/0.25),
    dt:float=0.25
    
):
    """
    ON-center RGC and OFF-center RGC response to bar stimulus
    
    Attributes:
    center_R:        center receptive field response to bar stimulus. 3D array.
    center_U_R:      center receptive field response to uniform background. 3D array.
    baseline:        ON-center RGC response to uniform background.
    delta:           3 msec delay between center and surround field responses. float.
    
    """

    t_len = center_R.shape[0]
    
    # ON-center response (center minus surround)
    ON_R = np.zeros(center_R.shape)
    ON_R[delta:,:,:] = center_R[delta:,:,:] - surround_R[:t_len-delta,:,:]

    # OFF-center response (surround minus center plus baseline)
    OFF_R = np.zeros(center_R.shape)
    OFF_R[delta:,:,:] = - center_R[delta:,:,:] + surround_R[:t_len-delta,:,:]
    
    for i in range(t_len - delta):
        ON_R[i, :, :] -= min(0, np.min(ON_R[i, :, :])) # at each time point, if min ON-center response is less than 0, then reduce the min response.
        OFF_R[i, :, :] += 2 * baseline[i] # at each time point, add the baseline to OFF-center response
        
    return ON_R, OFF_R


def IntraCorticalCurrent(
    
    IntraCor_delay:np.array, 
    connect:np.array, 
    firings:np.array, 
    t:int, 
    g:np.array, 
    h:np.array, 
    tau:float, 
    g_max:float, 
    N_cortical:int, 
    N_col=21

):
    
    """
    Attributes:
    
    
    IntraCor_delay:   A np.array that records the delay of each intracortical connection.
                      size (N_col * N_cortical * N), N is the number of input cortical cells.
    
    connect:          A np.array that records the connections between cortical cells.
                      size (N_col * N_cortical * 2).
                      connect[i, j, 0] tells which function column(in total 21 columns) does each input cell come from;
                      connect[i, j, 1] tells the exact id(0~84 for excitatory input cell and 0~21 for inhibitory) of each input cell inside its functional column.
    
    firings:          A np.array that records the firings of each cortical cell at each time point.
                      size (t_len=1001 * N_col * N_cortical).
    
    t:                time point.
    g:                intracortical input conductance matrix at last time point (time point t-1)
    h:                dynamic programming variable at last time point (time point t-1)
    
    """
    
    dt = 0.25
    
    for i in range(N_col):
        for j in range(N_cortical):  # at time t, the intracortical input conductance into cortical cell j at functional column i.
            
            delay = IntraCor_delay[i, j, :]   # the delay between the [i,j] cell and each cortical cell connected to it.
            
            cols = connect[i, j, 0]
            cellids = connect[i, j, 1]
       
            input_times = np.maximum((t-delay), 0)
            
            input_cells = sum(firings[input_times, cols, cellids])
            
            g[i, j] = np.exp(-dt/tau) * (
                g[i, j] + g_max * (np.e/tau) * (
                    h[i, j] + dt * input_cells
                )
            )
            
            h[i, j] = np.exp(-dt/tau) * (h[i, j] + dt * input_cells)
    
    return g, h



def AHPCurrent(
    
    firing, 
    g, 
    h, 
    g_max

):
    
    
    dt = 0.25
    tau = 2
    
    g = np.exp(-dt/tau) * (
        g + g_max * (np.e/tau) * (h + dt * firing)
    )
    
    h = np.exp(-dt/tau) * (h + dt * firing)
    
    return g, h
    

def LGNCurrent(
    
    on_connect:np.array, 
    off_connect:np.array, 
    on_spikes:np.array, 
    off_spikes:np.array, 
    N_cortical:int,
    N_columns=21, 
    t_len=1001

):
    
    """
    Attributes:
    
    on_connect:     A np.array that records the connection between on-lgn and cortical cells.
                    size (N_columns * N_cortical * 2)
                    on_connect[i, j, 0] tells the row index of each on-lgn connected with cortical cell [i,j]
                    on_connect[i, j, 1] tells the column index of each on-lgn connected with cortical cell [i,j]
                    (on-lgn cells is arranged in 21 rows * 21 columns)
                 
    off_connect:    A np.array that records the connection between off-lgn and cortical cells.
                    size (N_columns * N_cortical * 2)
                  
    on_spikes:      A np.array that records the spiking situation of each on-lgn cell at each time point.
                    size (t_len=1001 * 21 * 21)
    
    off_spikes:     A np.array that records the spiking situation of each off-lgn cell at each time point.
                    size (t_len=1001 * 21 * 21)
    
    
    """

    
    
    """
    on_g[t,i,j]: 时刻t，功能柱i，第j个皮层细胞接受的LGN输入电导
    """
    on_g = np.zeros((t_len, N_columns, N_cortical))
    off_g = np.zeros((t_len, N_columns, N_cortical))
    on_h = np.zeros((t_len, N_columns, N_cortical))
    off_h = np.zeros((t_len, N_columns, N_cortical))
    
    dt = 0.25
    
    if N_cortical == 84:  # 兴奋性细胞，delay分布参数
        mean = 10
        var = 5 
    elif N_cortical == 21:  # 抑制性细胞，delay分布参数
        mean = 5
        var = 3 
        
    tau_peak = 1     # msec
    g_max = 3        # nS(西门子)
        
    for i in range(N_columns):
        for j in range(N_cortical):
            
            on_rows = on_connect[i, j, 0]  # 每个皮层细胞，ON-LGN的所在行
            on_cols = on_connect[i, j, 1]  # 每个皮层细胞，ON-LGN的所在列
            
            off_rows = off_connect[i, j, 0]
            off_cols = off_connect[i, j, 1]
            
            
            delay_on = normal(loc=mean/dt, scale=(var/dt)**0.5, size=(len(on_rows))).astype('int')  # ON-LGN 到皮层细胞的delay
            delay_on[delay_on < 0] = 0
            delay_off = normal(loc=mean/dt, scale=(var/dt)**0.5, size=(len(on_rows))).astype('int')  # OFF-LGN 到皮层细胞的delay
            delay_off[delay_off < 0] = 0
            
            for t in range(1, t_len):
                
                on_input_times = np.maximum((t-delay_on), 0)
                off_input_times = np.maximum((t-delay_off), 0)
                
                on_input = sum(on_spikes[on_input_times, on_rows, on_cols])
                off_input = sum(off_spikes[off_input_times, off_rows, off_cols])
                
                on_g[t, i, j] = np.exp(-dt/tau_peak) * ( 
                    on_g[t-1, i, j] + 
                    g_max * (np.e/tau_peak) * (on_h[t-1, i, j] + dt * on_input) 
                )
                off_g[t, i, j] = np.exp(-dt/tau_peak) * ( 
                    off_g[t-1, i, j] + 
                    g_max * (np.e/tau_peak) * (off_h[t-1, i, j] + dt * off_input) 
                )
                on_h[t, i, j] = np.exp(-dt/tau_peak) * (on_h[t-1, i, j] + dt * on_input)
                off_h[t, i, j] = np.exp(-dt/tau_peak) * (off_h[t-1, i, j] + dt * off_input)
                
        
    return np.array(on_g + off_g)




def IntraCorticalConnect(
    
    N_cortical:int, 
    N_E:int, 
    N_I:int, 
    N_columns=21

):
    
    """
    problem: if N_I or N_E is too small, then some columns may contribute 0 excitatory or inhibitory spikes,
    especially for N_I, where the distribution is wide-spread.
    
    """
    
    from scipy.stats import norm
    from random import shuffle
    
    sigma_E = 7.5  # 兴奋性细胞 在临近功能柱 中的数量的概率分布标准差
    sigma_I = 60  # 抑制性细胞 在临近功能柱 中的数量的概率分布标准差
    
    prob_list_E = []; prob_list_I = []
    
    for i in range(-1, 2): # 兴奋性细胞来自3根功能柱
        d = i * 15 - 7.5
        cdf_sup = norm.cdf(d+15, loc=0, scale=sigma_E)
        cdf_inf = norm.cdf(d, loc=0, scale=sigma_E)
        prob_list_E.append(cdf_sup - cdf_inf) # 兴奋性细胞 落在该功能柱的概率
    for i in range(-4, 5): # 抑制性细胞来自9根功能柱
        d = i * 15 - 7.5
        cdf_sup = norm.cdf(d+15, loc=0, scale=sigma_I)
        cdf_inf = norm.cdf(d, loc=0, scale=sigma_I)
        prob_list_I.append(cdf_sup - cdf_inf) # 抑制性细胞 落在该功能柱的概率
    
    # 概率归一化
    prob_list_E = np.array(prob_list_E) / sum(prob_list_E)
    prob_list_I = np.array(prob_list_I) / sum(prob_list_I)
    
    number_E = [0 for _ in range(3)] # 落在临近功能柱的兴奋性细胞的数量
    number_I = [0 for _ in range(9)] # 落在临近功能柱的抑制性细胞的数量
    
    for i in range(N_E):
        column = np.random.choice(list(range(-1, 2)), p=prob_list_E.ravel())
        number_E[column + 1] += 1
    for i in range(N_I):
        column = np.random.choice(list(range(-4, 5)), p=prob_list_I.ravel())
        number_I[column + 4] += 1
        
        
    E_connect = []; I_connect = [] # 21根功能柱，每根柱 N_cortical 个皮层细胞，每个细胞收到的兴奋性连接和抑制性连接。
    
    for i in range(N_columns):
        
        E_connect.append([]); I_connect.append([])
        E_cols = [i-1, i, i+1] # 每根功能柱中，皮层细胞从周边总共 3根 功能柱（包括自身功能柱）中获得兴奋性连接，因此 E_cols 长度3
        I_cols = [(i+_) for _ in range(-4, 5)] # 从总共 9根 功能柱中获得抑制性连接，因此 I_cols 长度9
        
        for _ in range(3):
            if E_cols[_] < 0:
                E_cols[_] += 12
            if E_cols[_] >= 21:
                E_cols[_] -= 12
                
        for _ in range(9):
            if I_cols[_] < 0:
                I_cols[_] += 12
            if I_cols[_] >= 21:
                I_cols[_] -= 12
                
        for j in range(N_cortical):
            
            E_connect[i].append([])
            I_connect[i].append([])
            
            Columns_E = []; Columns_I = []
            Neurons_E = []; Neurons_I = []
            
            for k in range(3): 
                
                col_id = E_cols[k]
                neurons_id = list(range(j)) + list(range(j+1, 84))
                shuffle(neurons_id)
                Columns_E += [col_id] * number_E[k]
                Neurons_E += neurons_id[:number_E[k]]
                
            for k in range(9):
                
                col_id = I_cols[k]
                if j <= 20:
                    neurons_id = list(range(j)) + list(range(j+1, 21))
                else:
                    neurons_id = list(range(21))
                shuffle(neurons_id)
                Columns_I += [col_id] * number_I[k]
                Neurons_I += neurons_id[:number_I[k]]
            
            E_connect[i][j].append(Columns_E)
            E_connect[i][j].append(Neurons_E)
            
            I_connect[i][j].append(Columns_I)
            I_connect[i][j].append(Neurons_I)
    
    return np.array(E_connect), np.array(I_connect)
    
    
            
                
            
    
    
    
def LGN2CorticalConnect(
    
    N_cortical:int, 
    N_lgn:int, 
    N_columns=21

):
    
    from random import choice, shuffle
    
    half_width = 3
    half_length_list = [3, 5, 7]
    
    orilist = 15 * np.arange(21) - 150
    orilist[orilist > 90] = orilist[orilist > 90] - 180
    orilist[orilist < -90] = orilist[orilist < -90] + 180
    orilist = orilist / 180 * np.pi
    
    Xgrid, Ygrid = np.meshgrid(range(21), range(21))
    Xgrid = (Xgrid + 1) - 11
    Ygrid = 11 - (Ygrid + 1)
    
    ON_lgn_cor = []; OFF_lgn_cor = []
    
    for i in range(N_columns):
        
        ON_lgn_cor.append([])
        OFF_lgn_cor.append([])
        ori = orilist[i]
        
        for j in range(N_cortical):
            
            ON_lgn_cor[i].append([])
            OFF_lgn_cor[i].append([])
            
            half_length = choice(half_length_list)
            
            length_range = abs( np.cos(ori)*Ygrid - np.sin(ori)*Xgrid );
            width_range = abs( np.sin(ori)*Ygrid + np.cos(ori)*Xgrid );
            
            off_range = np.where((length_range <= half_length) * (width_range <= half_width))  # off-lgn 在长宽的范围之内
            off_rows_range = off_range[0]
            off_columns_range = off_range[1]
            
            on_range = np.where((length_range <= half_length) * (width_range >= half_width) * (width_range <= 7))  # on-lgn 在长的范围内，宽的范围外，但在三倍宽的范围内
            on_rows_range = on_range[0]
            on_columns_range = on_range[1]
            
            # randomly choose N_lgn/2 ON LGN cells.
            on = list(range(len(on_rows_range)))
            shuffle(on)
            on = on[:int(N_lgn/2)]
            
            # randomly choose N_lgn/2 OFF LGN cells.
            off = list(range(len(off_rows_range)))
            shuffle(off)
            off = off[:int(N_lgn/2)]
            
            on_rows = on_rows_range[on]
            on_columns = on_columns_range[on]
            
            off_rows = off_rows_range[off]
            off_columns = off_columns_range[off]
            
            ON_lgn_cor[i][j].append(on_rows.tolist())
            ON_lgn_cor[i][j].append(on_columns.tolist())
            OFF_lgn_cor[i][j].append(off_rows.tolist())
            OFF_lgn_cor[i][j].append(off_columns.tolist())
    
    return np.array(ON_lgn_cor), np.array(OFF_lgn_cor)

    


def ONBaseline(
    
    size:int, 
    rf_c, 
    rf_s, 
    tau_c, 
    tau_s, 
    t_len, 
    dt:float=0.25, 
    contrast:float=10, 
    delta:int=int(3/0.25)
    
):
    """
    Attributes:
    size:          size of the uniform background to generate.
    delta:         a 3 msec delay between center and surround field responses
    contrast:      the contrast of the image
    
    """
    
    uni_bg = np.ones((1, size, size)) * contrast  # generate a uniform background
    
    center_U_R = SpatialTemporalFilter(1, uni_bg, rf_c, 1, dt, tau_c)  # center response to uniform background at time 0
    surround_U_R = SpatialTemporalFilter(1, uni_bg, rf_s, 1, dt, tau_s)  # surround response to uniform background at time 0
    
    center_U_R = [center_U_R[0, 0, 0]] # center response to uniform background at time 0
    surround_U_R = [surround_U_R[0, 0, 0]] # surround response to uniform background at time 0
    
    # temporal convolution of center and surround responses to uniform background.
    for i in range(1, t_len):
        center_U_R.append(center_U_R[-1] * np.exp(-dt / tau_c) + center_U_R[0] * dt / tau_c)
        surround_U_R.append(surround_U_R[-1] * np.exp(-dt / tau_s) + surround_U_R[0] * dt / tau_s)
        
    # calculate Baseline: defined as the response of an ON cell to uniform background stimuli.
    baseline = [
        center_U_R[i + delta] - 
        surround_U_R[i] 
        for i in range(t_len - delta)
    ]
    
    return baseline


def OrientationSelection(
    ON_R,
    OFF_R,
    on2E_connect,
    on2I_connect,
    off2E_connect,
    off2I_connect,
):
    
    global N_LGN, N_cor_E, N_cor_I, N_E2E, N_E2I, N_I2E, N_I2I
    global t_len, dt
    global ref_tau, tau_E, g_E_max, tau_I, g_I_max, g_AHP_E_max, g_AHP_I_max
    global E_excit, E_inhibit, E_leak, E_AHP, Cm_E, Cm_I
    
    
    k = 1/1000

    # on-rgc to on-lgn delay, following normal distribution
    RGC_LGN_delay_ON = normal(loc=3/dt, scale=(1/dt)**0.5, size=(N_LGN,N_LGN)) # 均值 3/dt 个dt；标准差 1/dt 个dt。
    RGC_LGN_delay_ON = RGC_LGN_delay_ON.astype('int') # 取整
    RGC_LGN_delay_ON[RGC_LGN_delay_ON < 0] = 0

    # off-rgc to off-lgn delay, following normal distribution
    RGC_LGN_delay_OFF = normal(loc=3/dt, scale=(1/dt)**0.5, size=(N_LGN,N_LGN)) # 均值 3/dt 个dt；标准差 1/dt 个dt。
    RGC_LGN_delay_OFF = RGC_LGN_delay_OFF.astype('int') # 取整
    RGC_LGN_delay_OFF[RGC_LGN_delay_OFF < 0] = 0

    # for on-lgn [i,j] at time t, the probability of firing is  k*dt*ON_R[t-delay, i, j]
    ON_spikes = np.random.uniform(low=0, high=1, size=(t_len, N_LGN, N_LGN))
    # for off-lgn [i,j] at time t, the probability of firing is  k*dt*ON_R[t-delay, i, j]
    OFF_spikes = np.random.uniform(low=0, high=1, size=(t_len, N_LGN, N_LGN))

    for t in range(t_len):
        for i in range(N_LGN):
            for j in range(N_LGN):

                ON_delay = RGC_LGN_delay_ON[i, j]
                OFF_delay = RGC_LGN_delay_OFF[i, j]

                if t < ON_delay:
                    ON_spikes[t, i, j] = 0
                else:
                    if ON_spikes[t, i, j] <= k * dt * ON_R[t - ON_delay, i, j]:
                        ON_spikes[t, i, j] = 1
                    else:
                        ON_spikes[t, i, j] = 0

                if t < OFF_delay:
                    OFF_spikes[t, i, j] = 0
                else:

                    if OFF_spikes[t, i, j] <= k * dt * OFF_R[t - OFF_delay, i, j]:
                        OFF_spikes[t, i, j] = 1
                    else:
                        OFF_spikes[t, i, j] = 0


    """
    LGN to cortical conductance
    """
    LGN2Ecortical_current = LGNCurrent(on2E_connect, off2E_connect, ON_spikes, OFF_spikes, N_cor_E, t_len=t_len)
    LGN2Icortical_current = LGNCurrent(on2I_connect, off2I_connect, ON_spikes, OFF_spikes, N_cor_I, t_len=t_len)


    """
    voltage calculating
    """

    vmat_E = (-65) * np.ones((t_len, 21, N_cor_E))
    vmat_I = (-65) * np.ones((t_len, 21, N_cor_I))
    vmat_E[0, :, :] = (-65) * np.ones((21, N_cor_E))
    vmat_I[0, :, :] = (-65) * np.ones((21, N_cor_I))

    v_thresh_E = np.ones((21, N_cor_E)) * (-55) # baseline spike threshold value is -55 mV
    lastfire_E = np.zeros((21, N_cor_E))

    v_thresh_I = np.ones((21, N_cor_I)) * (-55)
    lastfire_I = np.zeros((21, N_cor_I))

    firings_E = np.zeros((t_len, 21, N_cor_E))
    firings_I = np.zeros((t_len, 21, N_cor_I))

    IntraCor_delay_EE = np.maximum(normal(loc=3/dt, scale=(1/dt)**0.5, size=(21, N_cor_E, N_E2E)).astype('int'), 0)
    IntraCor_delay_IE = np.maximum(normal(loc=3/dt, scale=(1/dt)**0.5, size=(21, N_cor_E, N_I2E)).astype('int'), 0)
    IntraCor_delay_EI = np.maximum(normal(loc=3/dt, scale=(1/dt)**0.5, size=(21, N_cor_I, N_E2I)).astype('int'), 0)
    IntraCor_delay_II = np.maximum(normal(loc=3/dt, scale=(1/dt)**0.5, size=(21, N_cor_I, N_I2I)).astype('int'), 0)




    g_EE = np.zeros((21, N_cor_E)); h_EE = np.zeros((21, N_cor_E))
    g_IE = np.zeros((21, N_cor_E)); h_IE = np.zeros((21, N_cor_E))
    g_EI = np.zeros((21, N_cor_I)); h_EI = np.zeros((21, N_cor_I))
    g_II = np.zeros((21, N_cor_I)); h_II = np.zeros((21, N_cor_I))
    g_AHP_E = np.zeros((21, N_cor_E)); h_AHP_E = np.zeros((21, N_cor_E))
    g_AHP_I = np.zeros((21, N_cor_I)); h_AHP_I = np.zeros((21 ,N_cor_I))


    for t in range(1, t_len - 1):
        """
        第t轮: ( vmat[t,:,:]和v_thresh已经在 t-1 时刻算好 )
        1.判断 t 时刻哪些 cortical cell 电势超过threshold, 记录在firings_E和firings_I相应位置
        2.更新 t+1 时刻的threshold
        3.更新 t+1 时刻的last fire time
        4.计算 t 时刻的电流，更新 t+1 时刻的vmat
        """

        """
        firing of excitatory cortical cells, update v_thresh_E
        """
        firings_E_cells = (vmat_E[t,:,:] > v_thresh_E)
        firings_E[t, firings_E_cells] = 1

        lastfire_E[firings_E_cells] = t

        Abs_ref_E_cells = ((t - lastfire_E) < Abs_ref_E) & (lastfire_E != 0)
        Rel_ref_E_cells = ((t - lastfire_E) > Abs_ref_E) & (lastfire_E != 0)

        v_thresh_E[Abs_ref_E_cells] = -45
        v_thresh_E[Rel_ref_E_cells] = (v_thresh_E[Rel_ref_E_cells] + 55) * np.exp(-dt / ref_tau) - 55





        """
        firing of inhibitory cortical cells, update v_thresh_I
        """
        firings_I_cells = (vmat_I[t,:,:] > v_thresh_I)
        firings_I[t, firings_I_cells] = 1

        lastfire_I[firings_I_cells] = t

        Abs_ref_I_cells = ((t - lastfire_I) < Abs_ref_I) & (lastfire_I != 0)
        Rel_ref_I_cells = ((t - lastfire_I) > Abs_ref_I) & (lastfire_I != 0)

        v_thresh_I[Abs_ref_I_cells] = -45
        v_thresh_I[Rel_ref_I_cells] = (v_thresh_I[Rel_ref_I_cells] + 55) * np.exp(-dt / ref_tau) - 55



        """
        conductance updating: dynamic programming
        """

        g_EE, h_EE = IntraCorticalCurrent(
            IntraCor_delay_EE, E2E_connect, firings_E, t, 
            g_EE, h_EE, 
            tau_E, g_E_max, N_cor_E
        )

        g_IE, h_IE = IntraCorticalCurrent(
            IntraCor_delay_IE, I2E_connect, firings_I, t, 
            g_IE, h_IE, 
            tau_I, g_I_max, N_cor_E
        )

        g_EI, h_EI = IntraCorticalCurrent(
            IntraCor_delay_EI, E2I_connect, firings_E, t, 
            g_EI, h_EI, 
            tau_E, g_E_max, N_cor_I
        )

        g_II, h_II = IntraCorticalCurrent(
            IntraCor_delay_II, I2I_connect, firings_I, t, 
            g_II, h_II, 
            tau_I, g_I_max, N_cor_I
        )

        g_AHP_E, h_AHP_E = AHPCurrent(firings_E[t,:,:], g_AHP_E, h_AHP_E, g_AHP_E_max)
        g_AHP_I, h_AHP_I = AHPCurrent(firings_I[t,:,:], g_AHP_I, h_AHP_I, g_AHP_I_max)



        """
        current updating
        """

        current_EE = (LGN2Ecortical_current[t,:,:] + g_EE) * (vmat_E[t,:,:] - E_excit)
        current_IE = g_IE * (vmat_E[t,:,:] - E_inhibit)

        current_EI = (LGN2Icortical_current[t,:,:] + g_EI) * (vmat_I[t,:,:] - E_excit)
        current_II = g_II * (vmat_I[t,:,:] - E_inhibit)

        current_leak_E = g_leak_E * (vmat_E[t,:,:] - E_leak)
        current_leak_I = g_leak_I * (vmat_I[t,:,:] - E_leak)

        current_AHP_E = g_AHP_E * (vmat_E[t,:,:] - E_AHP)
        current_AHP_I = g_AHP_I * (vmat_I[t,:,:] - E_AHP)

        dv_Eneurons = - (current_EE + current_IE + current_leak_E + current_AHP_E) * dt / Cm_E / 1000
        dv_Ineurons = - (current_EI + current_II + current_leak_I + current_AHP_I) * dt / Cm_I / 1000

        vmat_E[t+1, :, :] = vmat_E[t, :, :] + dv_Eneurons
        vmat_I[t+1, :, :] = vmat_I[t, :, :] + dv_Ineurons
        
        vmat_E[ t+1, Abs_ref_E_cells ] = v_ref
        vmat_I[ t+1, Abs_ref_I_cells ] = v_ref

        vmat_E[t, firings_E_cells] = 20
        vmat_I[t, firings_I_cells] = 20
        
        
    return vmat_E, vmat_I, firings_E, firings_I, LGN2Ecortical_current, LGN2Icortical_current



# ### 方向选择性模拟

# In[4]:


get_ipython().run_cell_magic('time', '', '\n\norilist = 15 * np.arange(21) - 150\norilist[orilist > 90] = orilist[orilist > 90] - 180\norilist[orilist < -90] = orilist[orilist < -90] + 180\norilist = orilist / 180 * np.pi\n\n"""\nTime parameters\n"""\ndt = 0.25\nt_len = 501\n\n"""\nNumber of different neurons\n"""\nN_RGC = 21  # ON and OFF both 21 * 21\nN_LGN = 21  # ON and OFF both 21 * 21\n\nN_cols = 21\nN_cor_E = 84\nN_cor_I = 21\n\nN_LGH2E = 24\nN_LGH2I = 16\n\nN_E2E = 36\nN_I2E = 24\n\nN_E2I = 56\nN_I2I = 8\n\n"""\ncenter and surround receptive fields parameters\n"""\nK_center = 17\nK_surround = 16\nsigma_center = 10.6\nsigma_surround = 31.8\nRF_width = 61\n\n# center receptive field & surround receptive field\nrf_center = RGCrf(N=RF_width, sigma=sigma_center, K=K_center)\nrf_surround = RGCrf(N=RF_width, sigma=sigma_surround, K=K_surround)\n\n\n"""\nelectrical properties\n"""\nE_excit = 0\nE_inhibit = -70\nE_AHP = -90\nE_leak = -65\nt_spike = 1 / dt\n\nAbs_ref_E = 3 / dt\nAbs_ref_I = 1.6 / dt\nv_ref = -65\nref_tau = 10\n\ng_E_max = 3  # excitatory synapses\ng_I_max = 5  # inhibitory synapses\ng_leak_E = 22.5\ng_leak_I = 18\ng_AHP_E_max = 40\ng_AHP_I_max = 20\n\n\ntau_center = 10\ntau_surround = 20\ntau_E = 2  # excitatory synapses\ntau_I = 1  # inhibitory synapses\ntau_AHP = 2\n\nCm_E = 0.45 # nF(法拉第), excitatory neurons\nCm_I = 0.18 # inhibitory neurons\n\n\n\n"""\nLGN to cortical connections\n"""\non2E_connect, off2E_connect = LGN2CorticalConnect(N_cor_E, N_LGH2E)  # ON/OFF-LGN to excitatory cortical cells\non2I_connect, off2I_connect = LGN2CorticalConnect(N_cor_I, N_LGH2I)  # ON/OFF-LGN to inhibitory cortical cells\n\n"""\nIntracortical connections\n"""\nE2E_connect,I2E_connect = IntraCorticalConnect(N_cortical=N_cor_E, N_E=N_E2E, N_I=N_I2E,)  # excitatory/inhibitory cortical cells to excitatory cortical cells\nE2I_connect, I2I_connect = IntraCorticalConnect(N_cortical=N_cor_I, N_E=N_E2I, N_I=N_I2I,)  # excitatory/inhibitory cortical cells to inhibitory cortical cells\n\n\n\n\n"""\nStimulus: oriented dark bars flashed onto a uniform light background.\n"""\n# contrast = 1\ndelta = int(3/0.25)\n\n# uniform background response\n\nORI_SELECTION_E = []   # stores the firings of all excitatory cells of each functional column at each specific barwidth, contrast, orientation and turn.\nORI_SELECTION_I = []   # stores the firings of all inhibitory cells of each functional column at each specific barwidth, contrast, orientation and turn.\nbar_width_list = [1]   # barwidth ranges in \'bar_width_list\'\ncontrast_list = [1]  # contrast ranges in \'contrast_list\'\n# orientation is set to be 0°\n\n\nfor bar_w in range(len(bar_width_list)):  # each specific barwidth\n    bar_width = bar_width_list[bar_w]\n    ORI_SELECTION_E.append([])\n    ORI_SELECTION_I.append([])\n    \n    for c in range(len(contrast_list)):  # each specific contrast\n        contrast = contrast_list[c]\n        ORI_SELECTION_E[bar_w].append([])\n        ORI_SELECTION_I[bar_w].append([])\n        \n        baseline = ONBaseline(\n            RF_width, \n            rf_center, \n            rf_surround, \n            tau_center, \n            tau_surround, \n            t_len, \n            dt, \n            contrast, \n            delta\n        )\n        \n        \n        for ori in range(1):  # each specific orientation\n            orientation = orilist[10]\n            ORI_SELECTION_E[bar_w][c].append([])\n            ORI_SELECTION_I[bar_w][c].append([])\n            \n            # generate bar stimuli\n            bar_stimu = GenerateBar(\n                N_RGC, \n                orientation, \n                t_len=t_len, \n                contrast=contrast, \n                bar_width=bar_width\n            )\n            \n            # size of the image\n            size = bar_stimu.shape[1]\n\n\n            """\n            RGC response to the bar stimulus.\n            """\n\n\n            # center and surround receptive field response to bar stimulus\n            center_R = SpatialTemporalFilter(N_RGC, bar_stimu, rf_center, t_len, dt, tau_center)\n            surround_R = SpatialTemporalFilter(N_RGC, bar_stimu, rf_surround, t_len, dt, tau_surround)\n\n\n            # ON-center and OFF-center RGC responses.\n            ON_R, OFF_R = ON_OFF_RGCResponse(\n                center_R,  \n                surround_R,\n                baseline,\n                tau_center, tau_surround,\n                delta\n            )\n            \n            turn = 0\n            \n            while turn < 3:\n                \n                E_result_turn = []\n                I_result_turn = []\n\n\n\n                vmat_E, vmat_I, firings_E, firings_I, ON_spikes, OFF_spikes = OrientationSelection(\n                    ON_R,\n                    OFF_R,\n                    on2E_connect,\n                    on2I_connect,\n                    off2E_connect,\n                    off2I_connect\n                )\n                \n               \n            \n\n                for column in range(21):\n\n                    excit_firings = firings_E[:, column, :].sum()\n                    inhibit_firings = firings_I[:, column, :].sum()\n\n                    E_result_turn.append(excit_firings)\n                    I_result_turn.append(inhibit_firings)\n                    \n                if E_result_turn[10] == max(E_result_turn):\n                    if I_result_turn[10] == max(I_result_turn):\n                        ORI_SELECTION_E[bar_w][c][ori].append(E_result_turn)\n                        ORI_SELECTION_I[bar_w][c][ori].append(I_result_turn)\n                        \n                        turn += 1\n                        \n\n')


# ### 结果可视化

# In[5]:


"""
Visualize the result

"""


ORI_SELECTION_E = np.array(ORI_SELECTION_E)
ORI_SELECTION_I = np.array(ORI_SELECTION_I)

ORI_SELECTION = (ORI_SELECTION_E + ORI_SELECTION_I) / 105 * 8

ORI_SELECTION_average = []
for bw in range(len(bar_width_list)):  #  specific bar width parameter
    for c in range(len(contrast_list)):  # specific contrast parameter
        for ori in range(1):  # specific orientation parameter
            average = np.zeros((21))
            for t in range(3):  # each turn
                average += ORI_SELECTION[bw, c, ori, t] / 5  # average on each turn
            ORI_SELECTION_average.append(average)
            
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot()

            ax.plot(
                average,
                color='black',
                linewidth=1
            )
            ax.scatter(x=range(21), y=average, s=10, color='black')

            plt.xticks(range(21), orilist/np.pi*180, rotation='vertical')
            plt.xlabel('Functional column’s favoriate orientation', fontsize=15)
            plt.ylabel('Mean Response ( spikes / sec)', fontsize=15)








# In[ ]:




