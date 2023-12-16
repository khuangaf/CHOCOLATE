# Do LVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning 

<div align="center">
Kung-Hsiang Huang†, Mingyang Zhou*, Hou Pong Chan‡,
Yi R. Fung†, Zhenhailong Wang†, Lingyu Zhang*, Shih-Fu Chang*, Heng Ji†

</div>
<div align="center">
<strong>University of Illinois Urbana-Champaign†</strong>

<strong>Columbia University*</strong>
<strong>University of Macau‡</strong>
</div>

<div align="center">
<hr>
</div>

This repository holds the CHOCOLATE Benchmark dataset, which is used for assessing the factuality of chart captioning systems. The dataset includes results from 6 different models applied to two distinct datasets. These includes:

* LVLM: GPT-4V, Bard (before Gemini)
* LLM-based Pipeline: DePlot + GPT-4
* Fine-tuned Model: ChartT5, MatCha, UniChart

Annotations are conducted on the VisText and Chart-to-Text (pew split) datasets. This ensures a wide range of data and types of factual errors.

Results are shown in the below figure and table. We found that all captioning models often generate captions that are factually inconsistent with the input chart. In fact, even for highly capable **LVLMs, their non-factual rate is a whopping 81.27%**.

<img src="./error_distribution.png"  class="center">

<div align="center">
<table>
  <tr>
    <th style="font-weight: bold;"></th>
    <th style="font-weight: bold;" colspan="2">CHOCOLATE-LVLM</th>
    <th style="font-weight: bold;" colspan="2">CHOCOLATE-LLM</th>
    <th style="font-weight: bold;" colspan="2">CHOCOLATE-FT</th>
  </tr>
  <tr>
    <td></td>
    <td># Factual</td>
    <td># Non-factual</td>
    <td># Factual</td>
    <td># Non-factual</td>
    <td># Factual</td>
    <td># Non-factual</td>
  </tr>
  <tr>
    <td>Sentence</td>
    <td>1,683</td>
    <td>1,270</td>
    <td>518</td>
    <td>469</td>
    <td>360</td>
    <td>1,023</td>
  </tr>
  <tr>
    <td>Caption</td>
    <td>74</td>
    <td>321</td>
    <td>27</td>
    <td>169</td>
    <td>112</td>
    <td>484</td>
  </tr>
</table>
</div>


## Spotlights

* CHOCOLATE - The first chart caption factuality benchmark for  
* CHOCOLATE is used to establish the Chart Caption Factual Error Correction task.
* Comming soon
    - [x] The CHOCOLATE benchmark
    - [ ] ChartVE metric
    - [ ] C2T model
    - [ ] Evaluation scripts
          

## The CHOCOLATE Benchmark 

We release the data for the CHOCOLATE benchmark at `data/chocolate.json`.

### Data Structure

Each instance in the json file corresponds to an annotation for a generated caption. Below, we illustrate the fields within each instance:

* **sentences**: A list of caption sentences.
* **labels**: A list of list, where the outer list correspond to sentence and the inner list corresponds to the errors within each sentence.
* **model**: A string that represents the model producing the caption.
* **dataset**: A string that represents which dataset the chart was sampled from.
* **image_path**: An URL to the chart image.
* **_id**: A unique identifier for this instance.


