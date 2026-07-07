# Green AI: Enerji Verimli Derin Öğrenme

Derin öğrenme modellerinde enerji tüketimini azaltmaya yönelik optimizasyon tekniklerinin (quantization, yapısal küçültme ve early stopping), farklı model mimarileri ve veri türleri üzerindeki etkisini karşılaştırmalı olarak inceleyen bir TÜBİTAK 2209-A projesi.

Çalışma kapsamında üç farklı görev ve modalite ele alınmıştır:

- **Görüntü sınıflandırma** — MobileNetV2 + CIFAR-10
- **Metin sınıflandırma** — DistilBERT + SST-2
- **Zaman serisi tahmini** — LSTM + NASA Sunspot Zaman Serisi

Her model için baseline, quantization (INT8), yapısal küçültme (pruning / hidden-size reduction) ve early stopping deneyleri ayrı ayrı yürütülmüş; toplam 13 deneyin sonuçları tek bir veri setinde birleştirilip etkileşimli bir dashboard üzerinden karşılaştırılabilir hâle getirilmiştir.


## Notebook'lar

| Notebook | Açıklama |
|---|---|
| `01_mobilenetv2_cifar10_optimization.ipynb` | MobileNetV2 + CIFAR-10 üzerinde baseline, quantization, pruning (Dense-only ve full-model) deneyleri |
| `02_distilbert_sst2_optimization.ipynb` | DistilBERT + SST-2 üzerinde baseline, dynamic quantization, structured attention-head pruning, early stopping deneyleri |
| `03_LSTM_optimization.ipynb` | LSTM + NASA Güneş Lekesi Zaman Serisi üzerinde baseline, dynamic quantization, early stopping, hidden-size reduction deneyleri |
| `visualization.ipynb` | Birleştirilmiş sonuç verisi (`all_results.csv`) üzerinden karşılaştırma grafiklerinin üretimi (doğruluk, model boyutu, enerji, çıkarım süresi, cross-model quantization etkisi) |

Notebook'lar yukarıdaki sırayla çalıştırılmalıdır; `visualization.ipynb`, önceki üç notebook'un ürettiği sonuç dosyalarına bağımlıdır.

## Sonuç Verisi

Tüm deneylerin sonuçları `results/all_results.csv` dosyasında birleştirilmiştir. Her satır bir deneyi temsil eder ve şu sütunları içerir: `model`, `deney`, `doğruluk`/`f1` (sınıflandırma modelleri için) veya `mae`/`rmse` (LSTM için), `model_boyutu_mb`, `çıkarım_süresi`, `enerji`.

## Dashboard

Sonuçlar Streamlit tabanlı, Plotly ile görselleştirilmiş etkileşimli bir dashboard üzerinden incelenebilir. Dashboard altı sekmeden oluşmaktadır: **Doğruluk**, **Model Boyutu**, **Enerji**, **Çıkarım Hızı**, **LSTM** ve **Deney Karşılaştırıcı** (ayrıca ham verinin görüntülendiği bir **Veri Tablosu** sekmesi bulunmaktadır).

### Çalıştırma

```bash
pip install -r requirements.txt
cd dashboard
streamlit run app.py
```

## Ortam ve Araçlar

- Python 3.x, TensorFlow, PyTorch, HuggingFace Transformers
- Enerji ölçümü: [CodeCarbon](https://github.com/mlco2/codecarbon) (`OfflineEmissionsTracker`)
- Görselleştirme: Streamlit, Plotly
- Tüm deneyler CPU üzerinde (Windows, GPU izleme devre dışı) yürütülmüştür.


## Proje Bilgisi

TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destek Programı kapsamında yürütülmüştür.

**Proje Yürütücüsü:** Bahar Erol
**Danışman:** Prof. Dr. Zafer CÖMERT