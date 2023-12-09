import h5py
import numpy as np
import tensorflow as tf

def test_h5(path="./data/training_MFCAD++.h5"):
    f = h5py.File(path, "r")
    print(len(list(f.keys())))
    print("-------")
    for key in list(f.keys()):
        group = f.get(key)
        A_2_shape = np.array(group.get("A_2_shape"))
        print(A_2_shape)
    f.close()

def write_shapes(path="./data/",filename="training_MFCAD++.h5"):
    """Load dataset with only adjacency matrix information."""
    hf = h5py.File(path+filename, 'r')
    with open(f"./.misc/debug_txts/debug_{filename}.txt", 'w') as writer: 
        for key in list(hf.keys()):
            group = hf.get(key)

            V_1 = np.array(group.get("V_1"))
            V_2 = np.array(group.get("V_2"))
            A_1_shape = np.array(group.get("A_1_shape"))
            A_2_shape = np.array(group.get("A_2_shape"))
            A_3_shape = np.array(group.get("A_3_shape"))
            
            writer.write(f"No: {key}\n")
            writer.write(f"V_1 shape: {V_1.shape}\n")
            writer.write(f"A_1 shape: {A_1_shape}\n")
            writer.write(f"V_2 shape: {V_2.shape}\n")
            writer.write(f"A_2 shape: {A_2_shape}\n")
            writer.write(f"A_3 shape: {A_3_shape}\n")
            writer.write("--------------------\n")
            


    hf.close()

def write_shapes_grt(path="./data/",filename="training_MFCAD++.h5",comp_n=10000):
    """Load dataset with only adjacency matrix information."""
    hf = h5py.File(path+filename, 'r')
    with open(f"./.misc/debug_txts/debug_grt_{filename}.txt", 'w') as writer:
        for key in list(hf.keys()):
            group = hf.get(key)

            V_1 = np.array(group.get("V_1"))
            V_2 = np.array(group.get("V_2"))
            A_1_shape = np.array(group.get("A_1_shape"))
            A_2_shape = np.array(group.get("A_2_shape"))
            A_3_shape = np.array(group.get("A_3_shape"))

            if V_2.shape > (comp_n,0):
                writer.write(f"No: {key}\n")
                writer.write(f"V_1 shape: {V_1.shape}\n")
                writer.write(f"A_1 shape: {A_1_shape}\n")
                writer.write(f"V_2 shape: {V_2.shape}\n")
                writer.write(f"A_2 shape: {A_2_shape}\n")
                writer.write(f"A_3 shape: {A_3_shape}\n")
                writer.write("--------------------\n")
    hf.close()

def print_h5_keys():
    hf = h5py.File("../hierarchical-brep-graphs-main/generated_in_vm/train.h5", 'r')
    for key in list(hf.keys()):
        group = hf.get(key)
        print(group)
        print(group.keys())
        print("--------------------------")
        
def print_h5_keys(n):
    hf = h5py.File("../hierarchical-brep-graphs-main/generated_in_vm/train.h5", 'r')
    key = list(hf.keys())[n]
    group = hf.get(key)
    print(group)
    print("--------------------------")
    for key2 in list(group.keys()):
        print(f"{key2} mit {len(list(group.get(key2)))} elementen")
        print( np.array(group.get(key2)))
        print("---")

def print_checkpoint_data(checkpoint_name):
    reader = tf.train.load_checkpoint(f"./checkpoint/{checkpoint_name}")
    shape_from_key = reader.get_variable_to_shape_map()
    dtype_from_key = reader.get_variable_to_dtype_map()
    return

#print_checkpoint_data("MF_CAD++_residual_lvl_7_edge_MFCAD++_units_512_date_2021-07-27_epochs_100.ckpt")
print_checkpoint_data("edge_lvl_6_units_512_epochs_10_date_2023-12-04.ckpt")

