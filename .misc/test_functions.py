import h5py
import numpy as np

def test_h5(path="./data/training_MFCAD++.h5"):
    f = h5py.File(path, "r")
    print(len(list(f.keys())))
    print("-------")
    for key in list(f.keys()):
        group = f.get(key)
        A_2_shape = np.array(group.get("A_2_shape"))
        print(A_2_shape)
    f.close()

def get_items(path="./data/",filename="training_MFCAD++.h5"):
    """Load dataset with only adjacency matrix information."""
    hf = h5py.File(path+filename, 'r')
    with open(f"./.misc/debug_txts/debug_{filename}.txt", 'w') as writer: 
        for key in list(hf.keys()):
            group = hf.get(key)

            V_1 = np.array(group.get("V_1"))
            V_2 = np.array(group.get("V_2"))
            labels = np.array(group.get("labels"), dtype=np.int16)

            A_1_idx = np.array(group.get("A_1_idx"))
            A_1_values = np.array(group.get("A_1_values"))
            A_1_shape = np.array(group.get("A_1_shape"))
            #A_1_sparse = tf.SparseTensor(A_1_idx, A_1_values, A_1_shape)
            #A_1 = tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")

            A_2_idx = np.array(group.get("A_2_idx"))
            A_2_values = np.array(group.get("A_2_values"))
            A_2_shape = np.array(group.get("A_2_shape"))
            #A_2_sparse = tf.SparseTensor(A_2_idx, A_2_values, A_2_shape)
            #A_2 = tf.Variable(tf.sparse.to_dense(A_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_2")

            A_3_idx = np.array(group.get("A_3_idx"))
            A_3_values = np.array(group.get("A_3_values"))
            A_3_shape = np.array(group.get("A_3_shape"))
            #A_3_sparse = tf.SparseTensor(A_3_idx, A_3_values, A_3_shape)
            #A_3 = tf.Variable(tf.sparse.to_dense(A_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_3")

            #yield [V_1, A_1, V_2, A_2, A_3], labels

            #print(f"No: {key}")
            #print(f"V_1 shape: {V_1.shape}")
            #print(f"A_1 shape: {A_1_shape}")
            #print(f"V_2 shape: {V_2.shape}")
            #print(f"A_2 shape: {A_2_shape}")
            #print(f"A_3 shape: {A_3_shape}")
            #print("--------------------")
            
            writer.write(f"No: {key}\n")
            writer.write(f"V_1 shape: {V_1.shape}\n")
            writer.write(f"A_1 shape: {A_1_shape}\n")
            writer.write(f"V_2 shape: {V_2.shape}\n")
            writer.write(f"A_2 shape: {A_2_shape}\n")
            writer.write(f"A_3 shape: {A_3_shape}\n")
            writer.write("--------------------\n")
            


    hf.close()

def print_items(path="./data/training_MFCAD++.h5"):
    """Load dataset with only adjacency matrix information."""
    hf = h5py.File(path, 'r')
    for key in list(hf.keys()):
        group = hf.get(key)
        V_1 = np.array(group.get("V_1"))
        V_2 = np.array(group.get("V_2"))
        labels = np.array(group.get("labels"), dtype=np.int16)
        A_1_idx = np.array(group.get("A_1_idx"))
        A_1_values = np.array(group.get("A_1_values"))
        A_1_shape = np.array(group.get("A_1_shape"))
        #A_1_sparse = tf.SparseTensor(A_1_idx, A_1_values, A_1_shape)
        #A_1 = tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")
        A_2_idx = np.array(group.get("A_2_idx"))
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))
        #A_2_sparse = tf.SparseTensor(A_2_idx, A_2_values, A_2_shape)
        #A_2 = tf.Variable(tf.sparse.to_dense(A_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_2")
        A_3_idx = np.array(group.get("A_3_idx"))
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))
        #A_3_sparse = tf.SparseTensor(A_3_idx, A_3_values, A_3_shape)
        #A_3 = tf.Variable(tf.sparse.to_dense(A_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_3")
        #yield [V_1, A_1, V_2, A_2, A_3], labels
        print(f"No: {key}")
        #print(f"V_1 shape: {V_1.shape}")
        #print(f"A_1 shape: {A_1_shape}")
        #print(f"V_2 shape: {V_2.shape}")
        #print(f"A_2 shape: {A_2_shape}")
        #print(f"A_3 shape: {A_3_shape}")
        #print("--------------------")
            
    print(len(list(hf.keys())))

    hf.close()

def get_items_gr_then(path="./data/",filename="training_MFCAD++.h5",comp_n=10000):
    """Load dataset with only adjacency matrix information."""
    hf = h5py.File(path+filename, 'r')
    with open(f"./.misc/debug_txts/debug_grt_{filename}.txt", 'w') as writer:
        for key in list(hf.keys()):
            group = hf.get(key)

            V_1 = np.array(group.get("V_1"))
            V_2 = np.array(group.get("V_2"))
            labels = np.array(group.get("labels"), dtype=np.int16)

            A_1_idx = np.array(group.get("A_1_idx"))
            A_1_values = np.array(group.get("A_1_values"))
            A_1_shape = np.array(group.get("A_1_shape"))
            #A_1_sparse = tf.SparseTensor(A_1_idx, A_1_values, A_1_shape)
            #A_1 = tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")

            A_2_idx = np.array(group.get("A_2_idx"))
            A_2_values = np.array(group.get("A_2_values"))
            A_2_shape = np.array(group.get("A_2_shape"))
            #A_2_sparse = tf.SparseTensor(A_2_idx, A_2_values, A_2_shape)
            #A_2 = tf.Variable(tf.sparse.to_dense(A_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_2")

            A_3_idx = np.array(group.get("A_3_idx"))
            A_3_values = np.array(group.get("A_3_values"))
            A_3_shape = np.array(group.get("A_3_shape"))
            #A_3_sparse = tf.SparseTensor(A_3_idx, A_3_values, A_3_shape)
            #A_3 = tf.Variable(tf.sparse.to_dense(A_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_3")

            #yield [V_1, A_1, V_2, A_2, A_3], labels

            if V_2.shape > (comp_n,0):
                writer.write(f"No: {key}\n")
                writer.write(f"V_1 shape: {V_1.shape}\n")
                writer.write(f"A_1 shape: {A_1_shape}\n")
                writer.write(f"V_2 shape: {V_2.shape}\n")
                writer.write(f"A_2 shape: {A_2_shape}\n")
                writer.write(f"A_3 shape: {A_3_shape}\n")
                writer.write("--------------------\n")
            


    hf.close()

#die nicht noch nicht batched-te datei browsen
def browe_war_h5():
    hf = h5py.File("../hierarchical-brep-graphs-main/generated_in_vm/train.h5", 'r')
    for key in list(hf.keys()):
        group = hf.get(key)
        print(group)
        print(group.keys())
        print("--------------------------")
        #V_1 = np.array(group.get("V_1"))
        #V_2 = np.array(group.get("V_2"))
        
#die nicht noch nicht batched-te datei browsen
def browe_war_h5_nth(n):
    hf = h5py.File("../hierarchical-brep-graphs-main/generated_in_vm/train.h5", 'r')
    key = list(hf.keys())[n]
    group = hf.get(key)
    print(group)
    print("--------------------------")
    for key2 in list(group.keys()):
        print(f"{key2} mit {len(list(group.get(key2)))} elementen")
        print( np.array(group.get(key2)))
        print("---")


def dataloader_adj(file_path):
    """Load dataset with only adjacency matrix information."""
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)

        V_1 = []#tf.Variable(np.array(group.get("V_1")), dtype=tf.dtypes.float32, name="V_1")
        V_2 = np.array(group.get("V_2"))
        labels = np.array(group.get("labels"), dtype=np.int16)

        A_1_idx = np.array(group.get("A_1_idx"))
        A_1_values = np.array(group.get("A_1_values"))
        A_1_shape = np.array(group.get("A_1_shape"))
        A_1_sparse = []#tf.SparseTensor(A_1_idx, A_1_values, A_1_shape)
        A_1 = None#tf.Variable(tf.sparse.to_dense(A_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_1")

        A_2_idx = np.array(group.get("A_2_idx"))
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))
        A_2_sparse = []#tf.SparseTensor(A_2_idx, A_2_values, A_2_shape)
        A_2 = []#tf.Variable(tf.sparse.to_dense(A_2_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_2")

        A_3_idx = np.array(group.get("A_3_idx"))
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))
        A_3_sparse = []#tf.SparseTensor(A_3_idx, A_3_values, A_3_shape)
        A_3 = []#tf.Variable(tf.sparse.to_dense(A_3_sparse, default_value=0.), dtype=tf.dtypes.float32, name="A_3")

        yield [V_1, A_1, V_2, A_2, A_3], labels

    hf.close()

'''
with open("./.misc/debug_txts/V2shapes.txt", 'w') as writer:
    for step, (x_batch_train, y_batch_train) in enumerate(dataloader_adj("data/training_MFCAD++.h5")):
        V_1, A_1, V_2, A_2, A_3 = x_batch_train
        writer.write(f"step: {step} \t V2:\n")
        writer.write(str(list(V_2)))
        writer.write("\n-------------\n")'''
get_items("./data/","training_MFCAD++.h5")
#get_items_gr_then("./data/","training_MFCAD++.h5",10000)
#browe_war_h5_nth(0)


