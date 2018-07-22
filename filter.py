def low_filter(data, init_data):
    last_data=init_data
    for i in range(np.size(data)):
        data[i] = data[i] - low_filter_weight * last_data
        last_data = data[i]
    return data


