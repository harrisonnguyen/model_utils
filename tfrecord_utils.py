import numpy as np # linear algebra
import tensorflow as tf
import os


"""
methods to write to tfrecord
"""
def write_tfrecord(file_name, data_array,feature_list,data_list,directory):
    instances = data_array[0].shape[0]
    #str_num = str(n).zfill(2)
    #str_num = str(n)
    # write the data
    if not os.path.exists(directory):
        os.makedirs(directory)
    writer = tf.python_io.TFRecordWriter(os.path.join(directory,file_name +".tfrecords"))

        #iterate over each example
    for i in range(instances):
        temp_dict = {}
        for j in range(0,len(feature_list)):
            if data_list[j] == 'float':
                temp_dict[feature_list[j]] = tf.train.Feature(float_list=tf.train.FloatList(value = data_array[j][i].astype(float)))
            elif data_list[j] == 'int':
                temp_dict[feature_list[j]] = tf.train.Feature(int64_list=tf.train.Int64List(value = [data_array[j][i]]))
            elif data_list[j] =='str':
                temp_dict[feature_list[j]] = tf.train.Feature(bytes_list=tf.train.BytesList(value = [data_array[j][i]]))

        #construct the example proto object
        example = tf.train.Example(
                    features = tf.train.Features(
                        feature = temp_dict))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    #writer.close()

    
def tfrecord_parser(serialized_example,feature_list,feature_type,feature_size):
    """Parses a single tf.Example into image and label tensors.
       e.g feature_list = ['mfcc','melspec','length','label','id']
            feature_type = ['float','float','int','int','str']
            serialized_example = some_file.tfrecord
    
    """
    # One needs to describe the format of the objects to be returned
    
    features_dict = {}
    for i in range(len(feature_list)):
        if feature_type[i] == 'float':
            features_dict[feature_list[i]] = tf.FixedLenFeature([feature_size], tf.float32)
        elif feature_type[i] == 'int':
            features_dict[feature_list[i]] = tf.FixedLenFeature([], tf.int64)
        elif feature_type[i] =='str':    
            features_dict[feature_list[i]] = tf.FixedLenFeature([],tf.string)
        else:
            raise ValueError('Not Valid data type')
    features = tf.parse_single_example(
                                        serialized_example,
                                        features=features_dict)
    return  [features[ele] for ele in feature_list]
    
def create_tfrecord_queue(batch_size,parser_fn,queue_size,n_epochs=None):
    # filenames ffor validation/training
    filenames = tf.placeholder(tf.string, shape=[None])
    batch_ph = tf.placeholder_with_default(BATCH_SIZE,())
    dataset = tf.data.TFRecordDataset(filenames)
    
    if n_epochs is None:
        # Repeat the input indefinitely.
        dataset = dataset.repeat()
    else:
         dataset = dataset.repeat(n_epochs)
         
    #convert byte string to something meaningful
    # some parser functions thaty uses lambda x: parser_fn(x,...,,)
    dataset = dataset.map(parser_fn)

    # shuffler
    dataset = dataset.shuffle(buffer_size=queue_size)

    dataset = dataset.batch(tf.cast(batch_ph,tf.int64))
    iterator = dataset.make_initializable_iterator()
    iterator_next = iterator.get_next()
    
    return filenames, batch_ph,iterator,iterator_next
    
    

