# Enabling Small Language Models for Text-to-Process Extraction: Balancing Accuracy and Efficiency through Distant Supervision

This is the companion repository for our CAiSE 2026 submission "Enabling Small Language Models for
Text-to-Process Extraction: Balancing Accuracy and Efficiency through Distant Supervision".

![](resources/figures/power/llama3_1_70b-md.png)

Power draw curve for LLama 3.1:70b

## Install instructions

We recommend creating an empty environment with Python==3.10
Dependencies can then be installed with 

`pip install -r requirements.txt`

## Reproducing our results

You can calculate the results of all SLMs by the following.

```shell
cd src
python -m eval.visualize
```

You can calculate the results of all LLMs by the following.

```shell
cd src
python -m eval.llm-metrics
```

To print the energy consumption run 

```shell
cd src
python -m efficiency
```

## Energy consumption results

You can explore how different LLMs and SLMs use energy over the procedure of extracting models from text.
Figures for these power draw curves are located in `resources/figures/power`

The raw data for power draw is located as `power.csv` next to the results of LLMs and SLMs in `resources/results/`.

The utility used to measure power draw is located in `src/power.py`. 
We used this code during SLM-based and LLM-based text-to-process extraction. 

## Prompts

All prompts used in this work are located in `resources/prompts`.
They are used in `src/description.py` and the LLM-annotators in `src/annotate`.

## Generating data

We used SAP-SAM in validating our approach. 
While this collection of process models is well suited to that, given its size, it also prohibits derivative works, which a generated dataset would fall under.
See their licence at https://github.com/signavio/sap-sam (make sure to read the one for the dataset, not the code!)

If you want to create your own dataset, you either have to provide models in the same format as SAP-SAM, or download the SAP-SAM dataset.
In the future, we will extend our code to work with other input formats as well. 
If you are interested and want to see your data format supported, feel free to reach out via mail, or create an issue in this repository!

Should you opt to work with the SAP-SAM dataset, download it and place the csv files in `resources/models/raw`.

You can then run `python -m selection` to select the models according to the selection criteria in our paper.
Selected process models are stored in `resources/models/selected`

Run `python -m cnl` to generate SBVR descriptions for selected process models.
You can find the described models in `resources/models/described`

Finally, to run the generation of annotated process descriptions, run

`python -m batches`

This will convert ALL selected models to training data. 
It requires an OpenAI account and roughly 60$ budget for our settings. 
Since we utilize OpenAI's batching API the annotation process can run for up to 4 days.
This is a worst case scenario, realistically the dataset is created after just 6-12 hours.

Remember to create and set your API key in `.env`, run `cp .env.template .env` and fill al required fields.

You can find the annotated documents in `resources/docs`, all requests to and answers from OpenAI in `batches`.

# Future work

We plan to extend this approach to many more modeling languages, such as ER, UML, or DCR.
If you see an application in your field, we would love if you reached out!
