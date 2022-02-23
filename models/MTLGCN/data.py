import numpy as np
import pandas as pd
import datetime

def load_pems08_volume_data():
    adj1 = pd.read_csv(r'../../data/PeMSD8/adj_pemsd8_1st_0-1.csv', header=None).values
    adj1 = np.array(adj1)
    adj2 = pd.read_csv(r'../../data/PeMSD8/adj_pemsd8_2nd_0-1.csv', header=None).values
    adj2 = np.array(adj2)
    adj3 = pd.read_csv(r'../../data/PeMSD8/adj_pemsd8_3rd_0-1.csv', header=None).values
    adj3 = np.array(adj3)
    adj4 = pd.read_csv(r'../../data/PeMSD8/adj_pemsd8_4th_0-1.csv', header=None).values
    adj4 = np.array(adj4)
    adj5 = pd.read_csv(r'../../data/PeMSD8/adj_pemsd8_5th_0-1.csv', header=None).values
    adj5 = np.array(adj5)
    volume = pd.read_csv(r'../../data/PeMSD8/pems08_volume.csv', header=None)
    volume = np.array(volume, dtype=np.float32)
    return volume, adj1, adj2, adj3, adj4, adj5

def load_urban_speed_data():
    adj1 = pd.read_csv(r'../../data/Urban/0-1_adj_1th.csv', header=None).values
    adj1 = np.array(adj1)
    adj2 = pd.read_csv(r'../../data/Urban/0-1_adj_2th.csv', header=None).values
    adj2 = np.array(adj2)
    adj3 = pd.read_csv(r'../../data/Urban/0-1_adj_3th.csv', header=None).values
    adj3 = np.array(adj3)
    adj4 = pd.read_csv(r'../../data/Urban/0-1_adj_4th.csv', header=None).values
    adj4 = np.array(adj4)
    adj5 = pd.read_csv(r'../../data/Urban/0-1_adj_5th.csv', header=None).values
    adj5 = np.array(adj5)
    spd = pd.read_csv(r'../../data/Urban/speedSeries.csv', header=None).values
    return spd, adj1, adj2, adj3, adj4, adj5

def z_score(x, mean, std):
    return (x - mean) / std

def process(data, date_path):
    data = np.array(data, dtype=np.float32)
    print("data.shape", np.array(data).shape)

    #加载日期
    date_list = pd.read_csv(date_path, header=None)
    date_list = np.array(date_list)
    print(date_list.shape)
    print(date_list[0])

    mean = data.mean()
    std = data.std()
    data = z_score(data, mean, std)

    data = np.reshape(data, [data.shape[0], data.shape[1], 1])
    print("data.shape", np.array(data).shape)

    dataset_week = []
    dataset_daily = []
    dataset_hour = []
    dataset_target = []

    start_idx = 288*7
    for i in range(start_idx,data.shape[0]-12):
        if datetime.datetime.strptime(date_list[i][0], '%Y-%m-%d %H:%M:%S').weekday() == 0:
            daily = data[i - 288 * 3:i - 288 * 3 + 12, :, :]
        elif datetime.datetime.strptime(date_list[i][0], '%Y-%m-%d %H:%M:%S').weekday() == 5:
            daily = data[i - 288 * 6:i - 288 * 6 + 12, :, :]
        else:
            daily = data[i - 288 * 1:i - 288 * 1 + 12, :, :]

        weekly = data[i-288*7:i-288*7+12,:,:]
        hourly = data[i-12:i,:,:]
        target = data[i:i+12,:,:]
        dataset_week.append(weekly)
        dataset_daily.append(daily)
        dataset_hour.append(hourly)
        dataset_target.append(target)
    dataset = np.array([dataset_week,dataset_daily,dataset_hour,dataset_target])
    print(np.array(dataset).shape)

    spilt_idx1 = int(dataset.shape[1] * 0.6)
    spilt_idx2 = int(dataset.shape[1] * 0.8)

    #train_set = dataset[:, :spilt_idx2, :, :]
    train_set = dataset[:,:spilt_idx1,:,:]
    valid_set = dataset[:,spilt_idx1:spilt_idx2,:,:]
    test_set = dataset[:,spilt_idx2:,:,:]

    print("train_set:", train_set.shape)
    print("valid_set:", valid_set.shape)
    print("test_set:", test_set.shape)

    return train_set, valid_set, test_set, mean, std
