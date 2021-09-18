import ctypes
import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np

import tensorrt as trt
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
torch.manual_seed(torch.initial_seed())

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
batch_size = 1

test_path = '/workspace/gender/our_data_cropped'

data_transforms = transforms.Compose([      
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_data = torchvision.datasets.ImageFolder(
    root = test_path,
    transform = data_transforms
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1,
    num_workers=0,
    shuffle=True
)

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

def infer(host_input, engine, device_input, stream, context, device_output, host_output, output_shape):
    #img = preprocess_image(img_path).numpy()
    #print("img.shape: ", img.shape)
    #print("np.array(img).shape: ", np.array(img).shape)
    #host_input = np.array(preprocess_image(img_path).numpy(), dtype=np.float16, order='C')
    print("host_input.shape: ", host_input.shape)
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()
    print("host_output: ",host_output)
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    print("output_data: ",output_data)
    return postprocess(output_data)

def postprocess(output_data):
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    print("confidences: ",confidences)
    print("confidences.argmax(): ", confidences.argmax())
    return(confidences.argmax())
    # find top predicted classes
    #_, indices = torch.sort(output_data, descending=True)


# def test_model(engine, device_input,  stream, context, device_output, host_output, output_shape, dataloader):
#     val_acc_history = []
#     running_corrects = 0
#     for inputs, labels in dataloader:
#         #print("inputs: ", inputs)
#         print("inputs.numpy().shape: ", inputs.numpy().shape)
#         inputs = np.array(inputs.numpy(), dtype=np.float16, order='C')
#         preds = infer(inputs, engine, d_input, stream, context, d_output, h_output, output_shape)
#         running_corrects += torch.sum(preds == labels.data)
#     #acc = running_corrects.double() / len(dataloader.dataset)
#     #print("Model accuracy: {:.4f}".format(acc))

engine_file = "/workspace/genderClassification/gender_model.engine"
engine, context, h_input, h_output, d_input, d_output, stream, output_shape = set_up(engine_file)
#test_model(engine, context, h_output, d_input, d_output, stream, output_shape, test_loader)

val_acc_history = []
running_corrects = 0
for inputs, labels in test_loader:
    inputs = np.array(inputs.numpy(), dtype=np.float16, order='C')
    preds = infer(inputs, engine, d_input, stream, context, d_output, h_output, output_shape)
    running_corrects += torch.sum(preds == labels.data)

acc = running_corrects.double() / len(test_loader.dataset)
print("Model accuracy: {:.4f}".format(acc))