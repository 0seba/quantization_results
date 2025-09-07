|                                   | Quantization Method        |         | MMLU-Pro |             |         | MMLU-Redux |            |         | MMLU-Redux-V2.0 |             |         | GPQA    |            |         | GPQA-Diamond |           |
|-----------------------------------|----------------------------|---------|----------|-------------|---------|------------|------------|---------|-----------------|-------------|---------|---------|------------|---------|--------------|-----------|
|                                   |                            | Acc     | Maj      | # 11801 @ 1 | Acc     | Maj        | # 8862 @ 3 | Acc     | Maj             | # 16944 @ 3 | Acc     | Maj     | # 2240 @ 5 | Acc     | Maj          | # 990 @ 5 |
| Qwen3-4B-Instruct-2507 [Reported] |                            | 69.6    |          |             | 84.2    |            |            |         |                 |             | 62.0    |         |            |         |              |           |
| Qwen3-4B-Thinking-2507 [Reported] |                            | 74.0    |          |             | 86.1    |            |            |         |                 |             | 65.8    |         |            |         |              |           |
| Qwen3-4B-Instruct-2507            |                            | 71.84@1 |          | 11733       | 81.23@3 | 82.77@3    | 8862       | 83.19@3 | 83.85@3         | 16925       |         |         |            |         |              |           |
| Qwen3-4B-Thinking-2507            |                            | 72.51@1 |          | 11800       |         |            |            | 83.93@3 | 85.22@3         | 16943       | 58.84@5 | 62.28@5 | 2240       | 63.43@5 | 67.68@5      | 923       |
| Qwen3-Instruct-2507               | Gemlite-4Bits-per-channel* | 70.93@1 |          | 11801       | 81.27@3 | 81.89@3    | 8862       | 83.36@3 | 83.85@3         |             |         |         |            | 59.6@5  | 61.62@5      | 990       |
|                                   | GPTQ-4Bits-per-channel     | 65.32@1 |          | 11801       | 77.48@3 | 78.94@3    | 8856       | 79.99@3 | 81.06@3         | 16942       |         |         |            | 45.56@5 | 46.46@5      | 988       |
|                                   | AutoRound-Best             | 67.09@1 |          | 11801       | 78.64@3 | 79.55@3    | 8862       | 81.02@3 | 82.24@3         | 16944       |         |         |            | 50.2@5  | 52.02@5      | 990       |
|                                   | AutoRound-RTN              | 61.44@1 |          | 11801       | 75.93@3 | 77.62@3    | 8862       | 78.77@3 | 80.076@3        | 16944       | 43.71@5 | 47.1@5  | 2240       | 34.95@5 | 43.94@5      | 990       |

\* not quantized, missing some sglang config


|                        | Quantization Method        |                | MMLU-Pro      |                |               |                |
|------------------------|----------------------------|----------------|---------------|----------------|---------------|----------------|
|                        |                            | Level 1 (5307) | Level 2 (663) | Level 3 (1645) | Level 4 (941) | Level 5 (3476) |
| Qwen3-4B-Instruct-2507 |                            | 100%           | 100%          | 100%           | 0%            | 23.42%         |
| Qwen3-4B-Thinking-2507 |                            | 100%           | 100%          | 100%           | 100%          | 0%             |
| Qwen3-Instruct-2507    | Gemlite-4Bits-per-channel* | 100%           | 100%          | 72.4%          | 41.45%        | 23.56%         |
|                        | GPTQ-4Bits-per-channel     | 100%           | 100%          | 37.63%         | 39.63%        | 21.52%         |
|                        | AutoRound-Best             | 100%           | 100%          | 50.03%         | 37.94%        | 22.07%         |
|                        | AutoRound-RTN              | 100%           | 0.0%          | 54.47%         | 33.9%         | 20.97%         |


- [AutoRoundBest](https://huggingface.co/seba/Qwen3-4B-Instruct-2507-AutoRound-Best-Channel): `auto-round --model Qwen/Qwen3-4B-Instruct-2507 --bits 4 --iters 1000 --output_dir Qwen3-4B-Instruct-2507-AutoRound-Best --format auto_gptq --group_size -1  --nsamples 512 --scale_dtype bf16 --model_dtype bf16 --enable_torch_compile --data_type int  --seqlen 4096 `
- [AutoRoundRTN](https://huggingface.co/seba/Qwen3-4B-Instruct-2507-AutoRound-RTN-Channel): `auto-round --model Qwen/Qwen3-4B-Instruct-2507 --bits 4 --iters 0 --output_dir Qwen3-4B-Instruct-2507-AutoRound-RTN --format auto_gptq --group_size -1`
- [GPTQ](https://huggingface.co/seba/Qwen3-4B-Instruct-2507-GPTQ-4-bits-Channel)
```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
model_id = "Qwen/Qwen3-4B-Instruct-2507"
quant_path = "Qwen3-4B-Instruct-2507-GPTQ"
calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(1024))["text"]
quant_config = QuantizeConfig(bits=4, group_size=-1)
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=32)
# model.save(quant_path)
```
