# Model_Deploy
一些深度学习模型部署的代码



## 目前已完成

- [x] YOLOv8/11的目标检测
- [x] Zero-DCE
- [ ] SAHI



## 待做

- [ ] YOLO11的实例分割



## 注意事项

1.在导出ONNX的时候尽量不要用动态维度，动态维度会导致模型占用显存明显变大



## 模型格式转换

### YOLO-Detect

1.onnx

```python
from ultralytics import YOLO
model = YOLO("xxx.pt")
model.export(format="onnx",
             opset=11, 
             simplify=True,
             imgsz=1024,
             dynamic=False,
             half=True)
```

2.engine

```bash
trtexec --onnx=xxx.onnx --saveEngine=xxx.engine --fp16
```



### Zero-DCE

1.onnx（这里将输入张量设置为固定大小，因为测试时发现如果不固定将占用非常大的显存）

```python
model = enhance_net_nopool()
model.load_state_dict(
    torch.load("E:/DeepLearning/Zero-DCE-improved/src/snapshots/pre-train.pth")
)
model.eval()

fix_input = torch.randn(1, 3, 1024, 1024)  # 示例输入，调整为所需大小
torch.onnx.export(
    model,
    fix_input,
    "zero-dce_1024-1024.onnx",
    input_names=["input"],
    output_names=["enhance_image_1", "enhance_image", "r"],
    opset_version=11,
)
```

2.engine

```bash
trtexec --onnx=xxx.onnx --saveEngine=xxx.engine --fp16
```



### YOLO-Seg

1.onnx

```python
from ultralytics import YOLO
model = YOLO("xxx.pt")
model.export(format="onnx",
             opset=11, 
             simplify=True,
             imgsz=1024,
             dynamic=False,
             half=True)
```

2.engine

```bash
trtexec --onnx=xxx.onnx --saveEngine=xxx.engine --fp16
```



## 参考

- [YOLO部署](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [ZeroDCE权重](https://github.com/Aiemu/Zero-DCE-improved)
