# DefectTwin

It has two main applications:
1. **deftstudio_app.py**: Provides a way to simulate defects from your prompt or reference images, as well as a general chat function.
2. **deft_realtime_app.py**: A real-time defect inspection tool using video streams.

DefT also includes a colab notebook to generate ready-to-use dataset to fine-tune LLM on any given context:

**Synthetic data generation pipeline**:
[Synthetic Data Generation Pipeline](https://github.com/turna1/DefectTwin/blob/main/defect_texture__fine_tunellm_with_synthetic_data.ipynb)
## Setup and Usage on Hugging Face

### Prerequisites
- API keys for OpenAI, Gemini, and Replicate.

### Instructions

1. **Get API Keys**:
   - [OpenAI](https://platform.openai.com/signup): Sign up and obtain your API key.
   - [Gemini](https://www.geminisecurity.com/signup): Sign up and obtain your API key.
   - [Replicate](https://replicate.com/signup): Sign up and obtain your API key.

2. **Clone the Repository**: Upload the necessary files (`deftstudio_app.py`, `deft_realtime_app.py`, `app.py`, `requirements.txt`) to your Hugging Face space.

3. **Install Dependencies**: Ensure all required libraries are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Applications**: Use the relevant Python files to start the applications:
   ```bash
   python deftstudio_app.py
   python deft_realtime_app.py
   ```

## Data Source

The datasets used to build DefT can be found [here](https://github.com/turna1/GenAI-For-Goods/tree/DATASETS-TO-BUILD-RAG-LLM-RAILWAY-DEFECT).

## Tools
1. [LLM-based Railway Defect Sample Generator](https://huggingface.co/spaces/Rahatara/trainingDefectGgenerator)

2. [RailDefectInspector](https://huggingface.co/spaces/Rahatara/RailDefectInspector)

3. [Defect Studio](https://huggingface.co/spaces/Rahatara/LLM_Defect_Analyst)


## Contact

If you are not a developer and would like to request a demo, please contact: [rferd068@uottawa.ca](mailto:rferd068@uottawa.ca).

## Citation

For academic or research purposes, please cite the following paper:

### DefectTwin: [When LLM Meets Digital Twin for Railway Defect Inspection] (https://arxiv.org/abs/2409.06725)
Ferdousi, Rahatara, et al. "DefectTwin: When LLM Meets Digital Twin for Railway Defect Inspection." arXiv preprint arXiv:2409.06725 (2024).


@article{ferdousi2024defecttwin,
  title={DefectTwin: When LLM Meets Digital Twin for Railway Defect Inspection},
  author={Ferdousi, Rahatara and Hossain, M Anwar and Yang, Chunsheng and Saddik, Abdulmotaleb El},
  journal={arXiv preprint arXiv:2409.06725},
  year={2024}
}

