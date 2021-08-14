# Abstractive Text Summarization using Encoder-Decoder Architecture

# Task
Given a document, the model would generate an abstarct summary while retaining the context of the document.

# Dataset
We use Amazon Fine Food Reviews dataset [data_link](https://www.kaggle.com/snap/amazon-fine-food-reviews), which has columns for food reviews and their corresponding summary for around 500000 data points.

# Baseline
Encoder-Decoder with Bahdanau Attention : We have referred and adapted the code for the baseline model from https://bastings.github.io/annotated_encoder_decoder/ (Baseline folder)

# Novelity
- **Self Attention**: To capture the dependencies between various encoder states, we introduce Self-Attention on top of Encoder outputs.
- **Attention on Decoder**: To generate contextually better sequences, we introduce Bahdanau attention on already generated tokens, while predicting the next token.
- **Multihead Attention**: Taking inspiration from Transformers which uses Multihead-attention, we introduce multiple Bahdanau attentions on Encoder states.
- **Variation to Multihead Attention**: We introduce a variation to multihead attention by combining one Bahdanau attention model and one Luong attention model.
- **Metric on Cosine Similarity**: Summary is subjective and Bleu score which does exact match, doesn’t do justice. We introduce an equivalent which computes cosine similarity between embeddings of output and target tokens instead.

# Structure of code
- *Each other folder (apart from Baseline) contains a variant architecture from above.*

- `Training.ipynb` contains common training code for all the variants

- `utils.py` contains Helper functions used by all the variants

- Please copy `Training.py` and `utils.py` to the folder of the intented variant before running the code

- Please download the data from [data_link](https://www.kaggle.com/snap/amazon-fine-food-reviews) and move to folder of intented variant

# Tools
- Python Version: 3.9.5
- Torch Version: 1.8.1 + cu111
- Torchtext Version: 0.9.1
- Numpy Version: 1.18.5
- Pandas Version: 1.2.4
- [BLEU Score](https://pypi.org/project/bleu/) Version: 0.3
- [ROUGE Scores](https://pypi.org/project/rouge/) Version: 1.0.0

# Brief Results
| Model        | BLEU Score           | Our Metric  |
| ------------- |:-------------:| -----:|
| BASELINE      | 22.86 | 0.482 |
| MULTIHEAD ATTENTION   | **25.41**      |   **0.516** |

# Sample Results
The following output sequences are generated for an input sentence (109 words) of a customer review on `coffee`.

`Target`: ”nice stuff”

`Baseline output`: ”dreamfields addition for the morning”

`Self-Attention output`: ”my little break to brew”

