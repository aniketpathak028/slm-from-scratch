
import os
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

model_fp32 = 'model.onnx'
model_int8 = 'model_quant.onnx'

print("Quantizing model to int8...")
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QUInt8,
)

print(
    f"Original size: {os.path.getsize(model_fp32) / (1024*1024):.2f} MB"
)
print(f"Quantized size: {os.path.getsize(model_int8) / (1024*1024):.2f} MB")