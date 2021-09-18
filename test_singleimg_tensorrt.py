import ctypes
import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np

import tensorrt as trt
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

import cv2 
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
batch_size = 1

#read tensorrt
def set_up(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            h_input = None
            input_shape = engine.get_binding_shape(binding)
            print("input_shape: ", input_shape)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float16).itemsize  # in bytes
            d_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            print("output_shape: ", output_shape)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            h_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float16)
            d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return engine, context, h_input, h_output, d_input, d_output, stream, output_shape

engine_file = "/workspace/genderClassification/gender_model.engine"
img1 = "/workspace/genderClassification/1.jpg"
img2 = "/workspace/genderClassification/2.jpg"
engine, context, h_input, h_output, d_input, d_output, stream, output_shape = set_up(engine_file)

data_transforms = transforms.Compose([      
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def preprocess_image(img_path):
    # transformations for the input data
    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    #print("data_transforms(image=input_img): ",data_transforms(image=input_img))
    input_data = data_transforms(input_img).unsqueeze_(0)
    return input_data

def postprocess(output_data):
    # get class names
    with open("labels.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    print("torch.nn.functional.softmax(output_data, dim=1): ",torch.nn.functional.softmax(output_data, dim=1))
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    print("confidences: ",confidences)
    # find top predicted classes
    print("output_data: ",output_data)
    _, indices = torch.sort(output_data, descending=True)
    print("output_data sorted: ",output_data)
    print("indices: ", indices)
    print("indices[0]: ",indices[0])
    print("indices[0][0]: ",indices[0][0])
    print("confidences[indices[0][0]]: ", confidences[indices[0][0]])
    print("confidences[indices[0][1]]: ", confidences[indices[0][1]])
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 50:

        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1

def infer(img_path, engine, device_input, stream, context, device_output, host_output, output_shape):
    img = preprocess_image(img_path).numpy()
    print("img.shape: ", img.shape)
    print("np.array(img).shape: ", np.array(img).shape)
    host_input = np.array(preprocess_image(img_path).numpy(), dtype=np.float16, order='C')
    print("host_input.shape: ", host_input.shape)
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    print("host_output: ",host_output)
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    print("output_data: ",output_data)
    postprocess(output_data)

infer(img2, engine, d_input, stream, context, d_output, h_output, output_shape)













