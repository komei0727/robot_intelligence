def load_mnist(dl_list, dataset_dir):
    import gzip
    import numpy as np

    #dl_list = [
     #   'train-images-idx3-ubyte.gz',
      #  'train-labels-idx1-ubyte.gz',
       # 't10k-images-idx3-ubyte.gz',
        #'t10k-labels-idx1-ubyte.gz'
    #]

    #dataset_dir = '/Users/komei0727/workspace/robot_intelligence/mnist/data'

#train_img
    file_path = dataset_dir + '/' + dl_list[0]
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    train_img = data.reshape(-1,784)

#train_label
    file_path = dataset_dir + '/' + dl_list[1]
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    train_label = data

#test_img
    file_path = dataset_dir + '/' + dl_list[2]
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    test_img = data.reshape(-1,784)

#test_label
    file_path = dataset_dir + '/' + dl_list[3]
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    test_label = data

    dataset = [train_img,train_label,test_img,test_label]

    return dataset
