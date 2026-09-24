import torch
from model import GPT, GPTConfig

config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,
    bias=True,
)

model = GPT(config)
model.load_state_dict(
    torch.load("best_model_params.pt", map_location=torch.device("cpu"))
)
model.eval()

class ONNXGPTWrapper(torch.nn.Module):

  def __init__(self, gpt_model):
    super().__init__()
    self.gpt = gpt_model

  def forward(self, idx):
    logits, _ = self.gpt(idx)
    return logits


wrapped_model = ONNXGPTWrapper(model)

dummy_input = torch.randint(0, 50257, (1, 8), dtype=torch.long)

print("Exporting model to ONNX format...")
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    dynamo=False
)

print("Export complete! 'model.onnx' generated successfully.")