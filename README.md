# GLUScope

This is the code for the tool demonstrated at <https://sjgerstner.github.io/gluscope>.
The website itself is in the ```docs/``` folder of this repository.
The goal is to visualise any given MLP neuron of a Transformer language model, especially via text examples that strongly activate it.

Contrary to previous work, this tool is specifically adapted to gated activation functions (GLU variants) like SwiGLU, which are largely used in recent open-weights LLMs.
In particular, each such neuron can have quite different types of activations, including strong negative ones, or activations with negative gate values.

I was strongly inspired by Neel Nanda's <https://neuroscope.io/>, hence the name.
See also <https://github.com/neelnanda-io/Neuroscope>
(but I preferred to write my own code).

BUT I only actually made a visualisation for a few neurons. Let's be honest: for most neurons nobody will ever care.

## How to use

Please let me know if you encounter any bugs!
Also, if you have any ideas on making the code more efficient, I'm more than happy to learn from you!

### Environment
Same as in <https://github.com/sjgerstner/RW_functionalities>.

### Text dataset

I used the text dataset that I published at <https://huggingface.co/datasets/sjgerstner/dolma-small>.
I generated it with:

```[bash]
python a_dataset.py --save_to dolma-small
```

If you want to create another dataset with different design choices, run the script with other options (see the argparse part of the code).

### Activation dataset

I published an activation dataset for OLMo-7B-0424 on dolma-small, at <https://huggingface.co/datasets/sjgerstner/OLMo-7B-0424-hf_neuron-activations>.
I generated it with:

```[bash]
python b_activations.py --model allenai/OLMo-7B-0424-hf --refactor_glu
```

The ```--refactor_glu``` flag causes a specific weight processing to happen when loading the model. See our paper for details (link upcoming).

This script takes about five days to run on a single NVIDIA A100-SXM4-80GB GPU.

It saves the summary data (the same as in the dataset) as a dictionary of tensors called ```summary_refactored.pt``` (or just ```summary.pt``` if the ```--refactor_glu``` flag was not specified).
You can optionally convert this to a Huggingface-compatible dataset with the script ```summary_dict_to_hf_dataset.py```.
I'm excited if you upload such a dataset to the Huggingface hub!

With the default options, the ```b_activations.py``` script additionally caches all the residual stream activations, which amounts to a whopping 25TB of data (which I didn't upload).
This makes later recomputing faster.
If you don't want this to happen, use the ```--no_cache``` flag.
To be precise, I cache the direct inputs to the MLP, i.e. after pre-LayerNorm. My code assumes a pre-norm architecture, so otherwise please tweak the code or use the ```--no_cache``` flag.

The script allows a lot of other options, see the argparse part of the code.
I'm excited if you publish similar datasets for other models!

### Recomputing and visualizing

To recompute relevant activations and make visualizations, e.g., for neurons 5.10602 and 31.9634, I run:

```[bash]
python d_recompute_vis.py --model allenai/OLMo-7B-0424-hf --refactor_glu --neurons 5.10602 31.9634 #for example
```

With the default options, the script automatically uses the data available on the machine, for example the cached residual streams if they are present.
When using the cache, it takes about 15 minutes per neuron, so it doesn't really make sense to do it for every single one.
When not using the cache (because it is not available, or if the ```--from_scratch``` option is enabled), it takes much longer. After all, the full model has to run on about 200 texts...

Again, other options are possible, see the argparse part of the code.

If you want to contribute a neuron page:

* fork this repo
* run the ```d_recompute_vis.py``` with the appropriate options as described above
* open a pull request.
