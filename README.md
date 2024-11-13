---
license: apache-2.0
pipeline_tag: time-series-forecasting
tags:
  - time series
  - forecasting
  - pretrained models
  - foundation models
  - time series foundation models
  - time-series
---

# Chronos-Bolt-Base

Chronos-Bolt is a family of pretrained time series forecasting models which can be used for zero-shot forecasting. Chronos-Bolt models are based on the [T5 architecture](https://arxiv.org/abs/1910.10683) and are available in the following sizes.


<div align="center">

| Model                                                                         | Parameters | Based on                                                               |
| ----------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| [**chronos-bolt-tiny**](https://huggingface.co/autogluon/chronos-bolt-tiny)   | 9M         | [t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)   |
| [**chronos-bolt-mini**](https://huggingface.co/autogluon/chronos-bolt-mini)   | 21M        | [t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)   |
| [**chronos-bolt-small**](https://huggingface.co/autogluon/chronos-bolt-small) | 48M        | [t5-efficient-small](https://huggingface.co/google/t5-efficient-small) |
| [**chronos-bolt-base**](https://huggingface.co/autogluon/chronos-bolt-base)   | 205M       | [t5-efficient-base](https://huggingface.co/google/t5-efficient-base)   |

</div>


## Usage with AutoGluon

> [!WARNING]  
> Chronos-Bolt models will be available in the next stable release of AutoGluon, so the following instructions will only work once AutoGluon 1.2 has been released.

The recommended way of using Chronos for production use cases is through [AutoGluon](https://auto.gluon.ai/stable/index.html), which features effortless fine-tuning, ensembling with other statistical and machine learning models for time series forecasting as well as seamless deployments on AWS with SageMaker.
Check out the AutoGluon Chronos [tutorial](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html).

A minimal example showing how to perform zero-shot inference using Chronos-Bolt with AutoGluon:

```
pip install autogluon
```

```python
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

df = TimeSeriesDataFrame("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")

predictor = TimeSeriesPredictor(prediction_length=48).fit(
    df,
    hyperparameters={
        "Chronos": {"model_path": "amazon/chronos-bolt-base"},
    },
)

predictions = predictor.predict(df)
```

## Usage with inference library

Alternatively, you can install the package in the GitHub [companion repo](https://github.com/amazon-science/chronos-forecasting).
This is intended for research purposes and provides a minimal interface to Chronos models.
Install the library by running:

```
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

A minimal example showing how to perform inference using Chronos-Bolt models:

```python
import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# Chronos-Bolt models generate quantile forecasts, so forecast has shape
# [num_series, num_quantiles, prediction_length].
forecast = pipeline.predict(
    context=torch.tensor(df["#Passengers"]), prediction_length=12
)
```

## Citation

If you find Chronos or Chronos-Bolt models useful for your research, please consider citing the associated [paper](https://arxiv.org/abs/2403.07815):

```
@article{ansari2024chronos,
  author  = {Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  title   = {Chronos: Learning the Language of Time Series},
  journal = {arXiv preprint arXiv:2403.07815},
  year    = {2024}
}
```

## License

This project is licensed under the Apache-2.0 License.
