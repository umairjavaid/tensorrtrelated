import ctypes
import pycuda.autoinit
import pycuda.driver as cuda

import cv2
import numpy as np
import time
import tensorrt as trt

import os

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
batch_size = 1

class VGG13:
    
    def __init__(self, engine_file):
        self.engine_file = engine_file
        self.engine, self.context, self.h_input, self.h_output, self.d_input, self.d_output, self.stream, self.output_shape = self.set_up(self.engine_file)
    
    def preprocess_img(self, frame):
        org_shape = 64
        frame = cv2.resize(frame, (org_shape, org_shape), interpolation = cv2.INTER_AREA)
        frame = np.asarray(frame, dtype=np.float32)
        frame = frame.transpose(2, 0, 1)
        mean = 255 * np.array([0.485, 0.456, 0.406])
        std = 255 * np.array([0.229, 0.224, 0.225])
        frame[0] = (frame[0] - mean[0]) / std[0]
        frame[1] = (frame[1] - mean[1]) / std[1]
        frame[2] = (frame[2] - mean[2]) / std[2]
        return np.ascontiguousarray(frame)

    def set_up(self, engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()

        for binding in engine:
            if engine.binding_is_input(binding): 
                h_input = None
                input_shape = engine.get_binding_shape(binding)
                input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  
                d_input = cuda.mem_alloc(input_size)
            else:  # and one output
                output_shape = engine.get_binding_shape(binding)
                h_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
                d_output = cuda.mem_alloc(h_output.nbytes)

        stream = cuda.Stream()
        return engine, context, h_input, h_output, d_input, d_output, stream, output_shape

    def infer(self, input_):
        input_ = self.preprocess_img(input_)
        input_ = np.array(input_, dtype=np.float32, order='C')
        cuda.memcpy_htod_async(self.d_input, input_, self.stream)
        self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.argmax()

model = VGG13("/workspace/vgg13/vgg13.engine")

def images_dir_inference():
    
    # images_path = os.getcwd() + '/Validated-dataset/Face_Position_Dataset/val/Front/'
    images_path = '/workspace/vgg13/val/Side/'
    #write_path = os.getcwd() + '/inference_results/orig-dataset/Side/'

    inference = 'Side'

    mapper=['Front', 'Side']
    frontal_counter = 0
    side_counter = 0
    accuracy = 0

    list = os.listdir(images_path) # dir is your directory path
    number_files = len(list)

    for img in os.listdir(images_path):
        img_path = images_path + img
        img_name = img.split('.')[0]

        print(img_path)
        image = cv2.imread(img_path)
        
        # t1=time.time()
        #output = inferenceFrontalFaceModel(image, model)
        # t2=time.time()
        # print(t2-t1)
        output = model.infer(image)
        
        output= mapper[output]
        print(output)

        if output == 'Front':
            frontal_counter = frontal_counter + 1
        elif output == 'Side':
            side_counter = side_counter + 1

        cv2.putText(image, output, (50, 50), 2, 1, (0, 0, 255))

        #output_img = write_path + img_name + '.jpg'
        #cv2.imwrite(output_img, image)

    print("Total images: ", number_files)
    print("Frontal predictions: ", frontal_counter)
    print("Side predictions: ", side_counter)

    if inference == 'Front':
        accuracy = (frontal_counter / number_files) * 100
    else:
        accuracy = (side_counter / number_files) * 100 

    print("Accuracy for {} is: {}".format(inference, round(accuracy, 2)))

images_dir_inference()
