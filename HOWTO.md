# How to train a model with the `llm-foundry`? 🤔

This document is a high-level overview of the steps involved in training a large language model (LLM) using the LLM Foundry. It is intended to provide a general understanding of the process and the various components involved.

## Step 1: Data Collection and Preprocessing

The first step in training an LLM is to collect and preprocess the data that will be used for training. This involves gathering a large corpus of text data, which can come from various sources. Two of the most common sources are:

* Public datasets available on the Hugging Face Hub, which can be easily accessed and used for training (e.g., [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [C4](https://huggingface.co/datasets/allenai/c4)).
* Dumps from [Common Crawl](https://commoncrawl.org/get-started).

In short, it is much easier to use the Hugging Face Hub datasets, as they are already preprocessed and ready for training. However, if you want to use the Common Crawl dumps, you will need to preprocess the data yourself. This involves cleaning the data, removing any unwanted content, and formatting it in a way that is suitable for training.

### The Easy Way: Hugging Face Hub Datasets

In [`utils/`](utils/), you can find scripts and tools to easily download datasets from the Hugging Face Hub (e.g., [`utils/download.py`](utils/download.py)). Scripts like [`data/preprocess.py`](data/preprocess.py) can help you format/enrich the data in a way that is suitable for your use case.

> **Note**: Almost every Python script in the LLM Foundry is accompanied by a SLURM bash script (e.g., [`utils/download.sh`](utils/download.sh)) that can be used to run the script on a SLURM-managed cluster—which happens to be the case for almost all clusters we have access to. However, you can also run the Python scripts locally, without using SLURM, by simply executing them with Python (e.g., `python utils/download.py`). Don't forget that the SLURM bash scripts are **templates** and not ready-to-use scripts. You will need to modify them according to your specifications (e.g., cluster partition, number of nodes, all paths, etc.) before running them.

### The Hard Way: Common Crawl Dumps

If you wish to use the Common Crawl dumps, you will need to preprocess the data yourself. You can find some scripts and tools for preprocessing the Common Crawl dumps on [`data/cc/`](data/cc/). Here is a high-level overview of the steps involved in preprocessing the Common Crawl dumps:

* Download the Common Crawl dumps that you want to use for training. This usually involves getting the addresses of all the WARC files of the specific crawl dump you want to use (see [`data/cc/warc_paths_get.sh`](data/cc/warc_paths_get.sh)), and then downloading them (see [`data/cc/warc_files_download.sh`](data/cc/warc_files_download.sh)).
* Extract the text data from the WARC files. Our data pipelines reproduce the FineWeb2 approach using a library called [datatrove](https://github.com/huggingface/datatrove), which allows you to build complex data pipelines for processing large datasets in a very lego-block-like fashion. You can find an example of how to use datatrove to extract text data from the Common Crawl dumps in [`data/cc/process_cc_dump_with_quality_filters.py`](data/cc/process_cc_dump_with_quality_filters.py). This text extraction process involves several steps, such as:
    * Reading the WARC files (`WarcReader`).
    * Filtering pages from blacklisted domains (`URLFilter`).
    * Extracting text from the pages (`Trafilatura`).
    * Performing language identification (LID) and keeping only the documents in the desired language(s) (`LanguageFilter`).
    * Running heuristic quality filters to remove low-quality documents (see [`data/filters/quality_filters.py`](data/filters/quality_filters.py)).
    * Removing duplicates (see [`data/filters/minhash.py`](data/filters/minhash.py)).

> **Note**: Preprocessing Common Crawl dumps can be very computationally expensive and time-consuming, especially if you are working with a large number of dumps or a limited number of cores to process the data. Also, when working with low-resource languages, it is important to be careful with the LID step. In [`data/cc/process_cc_dump_with_quality_filters.py`](data/cc/process_cc_dump_with_quality_filters.py), you can find an example of how to use datatrove to preprocess the Common Crawl dumps with a 2-round LID approach. Quality filtering and deduplication must also respect the language of the documents. In [`data/cc/.configs`](data/cc/.configs) you will find some examples of language-specific configurations for the preprocessing of the Common Crawl dumps (imported from the FineWeb2 pipeline).

### Training Annotators

Heuristics can only get you so far. If you want to train a really good model, you will need to have good data. And if you want to have good data, you will need to have good annotators (i.e., models that can annotate your data with labels that are relevant for your use case). For example, if you want to train a model that can perform well on knowledge-intensive tasks, you will need to have a good annotator that can annotate your data with respect to how "knowledge-rich" that data is. If you want to train a model that can perform well on reasoning tasks, you will need to have a good annotator that can annotate your data with respect to how reasoning-intensive that data is. And so on and so forth.

For this, one of the most scalable approaches is the LLM-as-a-Judge approach, which consists in using a large language model (e.g., Qwen3-32B) as an annotator to annotate your data with respect to the labels that are relevant for your use case. Many studies have shown that this approach can be very effective, and basically all models we trained under Polyglot relied on this approach to stratify the data and create training curricula (e.g., [Tucano](https://arxiv.org/abs/2603.03543), [LilMoo](https://arxiv.org/abs/2603.03508), and [LilTii](https://huggingface.co/blog/Polygl0t/liltii)).

However, it is inefficient to use the LLM-as-a-Judge approach to annotate the entirety of your data. A more efficient approach is to use the LLM-as-a-Judge approach to annotate a small subset of your data, and then use that annotated subset to train a smaller model (e.g., a 100-350M parameter encoder) that can then be used as an annotator to annotate the rest of your data. Here is a high-level overview of the steps involved in this process:

* Sample a small subset of your data (e.g., 100K documents), but make sure it is stratified according to the different sources your data comes from (e.g., Common Crawl, C4, FineWeb2, etc.).
* Annotate that small subset of your data using the LLM-as-a-Judge approach (see [`synthetic/generate.py`](synthetic/generate.py), since our synthetic data generation pipelines are also used for the annotation of the data).
* Train a smaller model with the annotated subset of your data (see [`data/filters/train_annotator.py`](data/filters/train_annotator.py)).
* If the results are good enough (i.e., the model can at least be used as a binary classifier to separate the good documents from the bad ones), use that smaller model as an annotator to annotate the rest of your data (see [`data/filters/run_annotator.py`](data/filters/run_annotator.py)).

This approach can be used to annotate your data with respect to any dimension that is relevant for your use case (e.g., educational value, toxicity, STEM content, code quality, etc.), as long as the judge LLM can annotate a small subset of your data with respect to that dimension, and as long as you are able to train a smaller model that can mimic the judge LLM's annotations with good enough performance.

### Making your own Data

If you have the resources to do so, you can also create your own data from scratch synthetically with the help of LLMs. This is a very interesting approach, since it allows you to create data that is tailored to your specific use case, and that has the characteristics that you want (e.g., a certain level of complexity, a certain distribution of topics, etc.).

> **Note**: In low-resource settings, you might not find any generator that is good enough to be used as a synthetic data generator for your specific use case. This is something you will have to evaluate on a case-by-case basis. If you find that the existing generators are not good enough for your use case, you can try using API models instead.

In [`synthetic/`](synthetic/) you can find scripts and tools to help you generate synthetic data with the help of LLMs. Here is a high-level overview of the steps involved in generating synthetic data with the help of LLMs:

* Find a seed of documents that are relevant for your use case (e.g., 10K documents). These can be, for example, wikipedia articles, or documents from the small subset of your data that you know are good for your use case.
* Select a generator that you will use to generate the synthetic data. This can be, for example, a model from the Hugging Face Hub (e.g., Qwen3-32B), or an API model (e.g., GPT-4). Depending on the task (e.g., generate summaries, generate GAQs, etc.) you can use smaller models (e.g., 3B parameter models) as generators. See *"[The Synthetic Data Playbook: Generating Trillions of the Finest Tokens](https://huggingface.co/spaces/HuggingFaceFW/finephrase)"* for some cool lessons on how to optimize the generation of synthetic data with the help of LLMs.
* Scripts like [`synthetic/generate.py`](synthetic/generate.py) and [`synthetic/generate_datatrove.py`](synthetic/generate_datatrove.py) can be used to create data generation pipelines. [`synthetic/generate_cai.py`](synthetic/generate_cai.py) can be used to create synthetic data with a Constitutional AI approach, which can be very useful to create alignment data for post-training.

## Step 2: Tokenization

The next step in training an LLM is to tokenize the data. This involves converting the raw text data into a format that can be fed into the model for training. Like in the last step, there are two ways of doing this:

* **Easy way**: Use a pre-trained tokenizer that is available on the Hugging Face Hub (e.g., [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/tokenizer.json)).
* **Hard way**: Train your own tokenizer from scratch.

Both approaches have pros and cons. The easy way is, well, easier, and it can be a good option if there already exists a good pre-trained tokenizer that is suitable for your use case. One simple way to evaluate a tokenizer is by seeing how well it compresses your data. If the tokenizer is able to compress your data well (i.e., it can represent your data with a small number of tokens), this will help you spend less compute on training and inference. See [`tokenizer/tokenizer_eval.py`](tokenizer/tokenizer_eval.py) for an example of how to evaluate a tokenizer. The hard way is, well, harder, but it can be a good option if there is no good pre-trained tokenizer that is suitable for your use case (e.g., because you are working with a low-resource language, or because your data has very specific characteristics that are not well captured by the existing pre-trained tokenizers).

> **Note**: Faulty tokenizers are silent killers. If your tokenizer has a weird bug that causes it to split the text in a very bad way, this can have a very negative impact on the performance of your model, and it can be very hard to debug. So, if you decide to train your own tokenizer, make sure to evaluate it **very carefully** before using it for training your model. By **"very carefully"**, I mean that you should not only evaluate it with the standard metrics (e.g., fertility), but also do a qualitative analysis of the tokenization output to make sure it is splitting the text in a reasonable way.

### I choose to train my own tokenizer, now what?

If you choose to train your own tokenizer, you can use the scripts and tools available on [`tokenizer/`](tokenizer/) to do so. Here is a high-level overview of the steps involved in training your own tokenizer:

* Prepare a portion of your dataset that is representative of the data you will be using for training. This portion does not need to be very large (e.g., 1M documents), but it should be representative of the different sources and characteristics of your data. If you will train your model on English, Spanish and Arabic data, make sure to include documents in those three languages in the portion of the dataset you will use to train your tokenizer.
* Choose your backend for training the tokenizer. The most common backends are [SentencePiece](https://github.com/google/sentencepiece) and [Tokenizers](https://github.com/huggingface/tokenizers).
* Regardless of the backend you choose, we have scripts that can help you train your tokenizer:
    * For SentencePiece, you can use [`tokenizer/train_tokenizer_sentencepiece.py`](tokenizer/train_tokenizer_sentencepiece.py).
    * For Tokenizers, you can use [`tokenizer/train_tokenizer_tokenizers.py`](tokenizer/train_tokenizer_tokenizers.py).

> **Note**: If you choose to train your own tokenizer, make sure to think about things like the vocabulary size (e.g., 32K, 64K, 128K), the special tokens you will need (e.g., padding token, end of sequence token, unknown token, etc.), and the different pre-tokenization and post-tokenization rules you will need to apply to make sure your tokenizer is splitting the text in a reasonable way. When all qualitative tests pass, make sure to evaluate the tokenizer quantitatively with the standard metrics (see [`tokenizer/tokenizer_eval.py`](tokenizer/tokenizer_eval.py)), since these metrics can be used to compare different tokenizers in terms of efficiency and performance, which will tell you how much you are gaining (or losing) by using your own tokenizer instead of an off-the-shelf option.

### I have a tokenizer, now what?

After you have a tokenizer (either a pre-trained one, or one that you trained yourself), you can use it to tokenize your data and prepare it for training. Tokenizing is nothing more than applying the tokenizer to your data (i.e., converting the raw text data into sequences of token IDs) and packing it into the appropriate context windows for training. In the foundry, we perform all of this **offline** for pretraining. This means that we tokenize the data and pack it into the appropriate context windows before training the model, and then we save the tokenized and packed data on disk, so that it can be efficiently loaded during training. This introduces an extra step in the process, but it allows us to simplify the data loading process during training, which can be very beneficial when working with large datasets and large models 

> ***"Whatever can be done offline, should be done offline."*** Not always true, but a good principle to follow in general if you are aiming for simplicity and efficiency.

Here is an overview of the steps involved in tokenizing your data and preparing it for training:

* Use your tokenizer to tokenize your data (see [`data/tokenization/run_tokenization.py`](data/tokenization/run_tokenization.py)). This script will take your raw text data, apply the tokenizer to it, and convert it into sequences of token IDs.
* After tokenizing your data, you will need to pack it into the appropriate context windows for training (see [`data/tokenization/pack.py`](data/tokenization/pack.py)). This script will take the tokenized data and pack it into the appropriate context windows (e.g., 4096 tokens), which will be used for training the model. See the script for more details on different packing strategies (e.g., simple concatenation vs. best-fit decreasing).
* After tokenizing and packing your data, it is a good practice to run a decontamination step to make sure that there are no documents in your training data that are too similar to the evaluation data (e.g., MMLU, HumanEval, etc.), since this can lead to an overestimation of the performance of your model on the evaluation data. You can find a script for decontamination on [`data/tokenization/decontaminate.py`](data/tokenization/decontaminate.py).
* After tokenizing, packing and decontaminating your data, you can now create a small validation set that you will use during training to monitor the loss/perplexity of your model on unseen data (see [`data/tokenization/make_validation_split.py`](data/tokenization/make_validation_split.py)). In LLM-land, this is not actually what we care about. We care about the performance of our model on downstream tasks (e.g., GSM8K, HellaSwag, etc.). Regardless, it is still a good practice to create a small validation set that you can use to monitor the training process and make sure that your model is learning something.

## Step 3: Define your Evaluation Harness

Before you train anything, it is good practice to know how you will evaluate your model. If you don't have evaluations, you don't have anything to optimize for, and you have no business training a model. So, before you start training, make sure you have an evaluation harness in place that you can use to assess the performance of your model on downstream tasks.

In the foundry, we use EleutherAI's [LM Evaluation Harness](https://github.com/Polygl0t/lm-evaluation-harness) as our evaluation framework, and our fork has several benchmarks that we have implemented ourselves for the languages we worked on. [Here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) and [here](https://github.com/Polygl0t/workshop-exercises/tree/main/day3/pretraining/solutions/exercise6) are resources for how to add evaluations to the harness. But before going out and adding a bunch of evaluations, make sure to do some research and see if there is something already implemented that you can use (see the [task list](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks#tasks) in the harness). There may be evaluations available outside the harness, and in those cases we recommend you port them to the harness. This will make it easier for you to evaluate your model, and it also gives you a minimal baseline for reproducibility by making it easier for other people to evaluate their models on the same tasks and compare results.

To see examples of how to use the evaluation harness, check out the scripts in [`evals/`](evals/).

## Step 4: Train your Model (Pretraining)

You have data. You have a tokenizer. You have evaluations. Now you are ready to train your model. The training process involves several steps, like defining your model architecture, setting up your training loop, defining your optimization strategy, and so on and so forth. In the foundry, we have scripts and tools that can help you with all of these steps (see [`distributed/`](distributed/)). Here is a high-level overview of the steps involved in training your model:

* The foundry is optimized to train decoder-only transformer models. It is also optimized for certain architectures which have support for the optimized kernels we use (e.g., FlashAttention, Liger, Causal-Conv1D, etc.). Please see [`distributed/README.md`](distributed/README.md) for more details on the model architectures that are currently supported/recommended.
* In the foundry, you define your model architecture with a configuration file. This is the same `config.json` that every model in the Hugging Face Hub has. You can find some examples of configuration files in [`distributed/README.md#example-architecture-configs`](distributed/README.md#example-architecture-configs).
* All other hyperparameters related to the training process (e.g., learning rate, batch size, number of training steps, etc.) are defined in the [`specifications.yaml`](distributed/specifications.yaml) file. You can learn more about all the different hyperparameters that you can set in [`distributed/specifications.py`](distributed/specifications.py).
* We have helper scripts to help you define things like learning rate and batch size, via some heuristics found in the literature (all hail DeepSeek). See [`utils/compute_hyperparams.py`](utils/compute_hyperparams.py) for more details. It is also a good approach to mirror values from existing papers/models that are similar to the one you want to train (especially if you don't have time to ablate these things yourself). For example, if you want to train a 7B parameter model, you can look at the training hyperparameters used by other 7B parameter models (e.g., Qwen2.5-7B, LLaMA-2-7B, etc.) and use similar values for your model. However, nothing beats a good hyperparameter ablation, so if you have the time and resources to do it, it is always a good idea to ablate different hyperparameter values to see what works best for your specific use case.
* The foundry relies on raw PyTorch for pretraining. It will enable you to launch either as a stand-alone/single GPU training job, a distributed training job via DDP (Distributed Data Parallel), or a distributed training job via FSDP2 (Fully Sharded Data Parallel). The training scripts are located in [`distributed/`](distributed/), and you can find examples of how to launch training jobs with these scripts:
    * [`distributed/train_ddp.sh`](distributed/train_ddp.sh) for DDP training.
    * [`distributed/train_fsdp.sh`](distributed/train_fsdp.sh) for FSDP2 training.

> **Note**: DDP is suited for training models that you can fit in a single GPU, since it relies on only data parallelism and a single `all_reduce` on the gradients. FSDP2 is suited for training larger models that cannot fit in a single GPU, since it relies on model parallelism as well (model/gradient/optimizer sharding, and a lot of `all_gather` and `reduce_scatter` gymnastics). According to our scaling tests, both our DDP and FSDP2 implementations scale almost linearly up to 256 GPUs (training a 7B parameter dense model). See [this](https://arxiv.org/html/2603.03543v1#A8) section of our Tucano paper for more details on the scaling performance of our implementations.

### What about merging and transplantation?

Merging and transplantation are two techniques that can be used to improve the performance of your model by leveraging the knowledge learned by other models. **Merging** consists in merging the weights of two or more models to create a new model that combines the knowledge learned by the original models. It is useful for when you have several models that share the same tokenizer and architecture, but were trained on different data, and you want to merge them to create a new model that has knowledge of all the data. **Transplantation** consists in transplanting certain parts of a donor model into a recipient model. For example, tokenizer transplantation consists in transplanting the vocabulary/embeddings of a donor model into a recipient model, which can be useful (and cost-effective) in continual pretraining scenarios (see *"[Tucano 2 Cool: Better Open Source LLMs for Portuguese](https://arxiv.org/abs/2603.03543)"* for an example of tokenizer transplantation that worked well).

To work with transplantation and merging techniques, you can use the scripts available on [`merge/`](merge/).

## Step 5: Post-Training and Alignment

After you have pretrained your model, if you want to go further than having just a generic base model, you can perform some post-training and alignment techniques to further improve the performance of your model on downstream tasks, or to align it with human preferences. This makes the model more steerable and more useful for end-users. Some examples of post-training and alignment techniques are:

* **Supervised fine-tuning (SFT)**: This consists in fine-tuning your model on a specific task or set of tasks with supervised data. For example, if you want to improve the performance of your model on question answering tasks, you can fine-tune it on a question answering dataset (see [`alignment/sft_trainer.py`](alignment/sft_trainer.py)).
* **Direct preference optimization (DPO)**: This consists in fine-tuning your model with a DPO objective, which optimizes the model via contrastive pairs of better and worse model outputs ranked by humans or by a reward model (see [`alignment/dpo_trainer.py`](alignment/dpo_trainer.py)). The same type of data used to perform DPO can be used to train reward models (see [`alignment/reward_trainer.py`](alignment/reward_trainer.py)), which can then be used to perform either reinforcement learning or serve as a preference-based sampler during inference.
* **Group Relative Policy Optimization (GRPO)**: This consists in fine-tuning your model with verifiable rewards. For this, we use [`alignment/gym`](alignment/gym) to procedurally generate samples with verifiable group identities, and then we use those samples to fine-tune the model with a GRPO objective.

## FAQ (Frequently Asked Questions)

1. **How long does it take to train a model?** 

- The time it takes to train a model depends on several factors, such as the size of the model, the size of the dataset, the computational resources available, and the training hyperparameters. For example, training a 7B parameter model on a dataset of 1 trillion tokens with 256 A100 GPUs can take around 2 weeks. However, these numbers can vary significantly depending on the specific circumstances.

2. **Are the SLURM bash scripts ready to use?** 
- No, the SLURM bash scripts provided in the LLM Foundry are templates and not ready-to-use scripts. You will need to modify them according to your specifications (e.g., cluster partition, number of nodes, all paths, etc.) before running them. Paths are the most common thing you will need to change, since the directory structure of your cluster/workspace will probably be different from the one we set in the bash script templates.
