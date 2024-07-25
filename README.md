# DefectTwin

# DefectTwin

DefT has two main applications:
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

## Contact

If you are not a developer and would like to request a demo, please contact: [rferd068@uottawa.ca](mailto:rferd068@uottawa.ca).

## Citation

For academic or research purposes, please cite the following paper:

### DefectTwin: LLM-Driven Digital Twins in Consumer Electronics for Visual Railway Defect Inspection [Link Soon]

**Abstract**  
Digital Twin (DT) mimics objects, processes, or systems. Consumer Electronics (CE) in transportation, healthcare, and product quality maintenance benefit from DT visualization and predictive maintenance. However, employing DTs in CE requires precision, multimodal data processing, and zero-shot generalizability for cross-platform compatibility. Advances in Large Language Model (LLM) have revolutionized traditional ML-driven systems by solving complex natural language problems. Motivated by these advancements, we present DefectTwin (DefT), an LLM-integrated DT approach for visual defect inspection in railway components and consumer electronics. DefT uses a multimodal and multimodel (M²) LLM-based AI inferencing pipeline to analyze visual railway defects, achieve precision and zero-shot generalizability, and incorporate a Quality-of-Experience (QoE) feedback loop. The pipeline generates synthetic datasets to fine-tune LLMs, enabling context enforcement for multimodal decoders. DefT also includes a multimodal processor unit for advanced visualization of defect characteristics, such as mapping defect textures onto 3D models. Users interact with DefT through a multimodal interface, consuming the outcomes of information and predictive twins for defect analysis and predictive simulation. We achieved a precision of 0.93 for in-domain image defect identification in DefectTwin (DefT), which outperformed current models through rigorous testing on data from the Canadian Pacific Railway (CPR) and open-source. DefT also demonstrated proficient performance for text, image, and video stream evaluations. To the best extent of our knowledge, DefectTwin is the first LLM-integrated DT for railway defect inspection. We anticipate that DefT paves the way for integrating LLM-based DT in CE, enhancing interoperability, efficiency, cost reduction, and proactive maintenance across diverse applications.

**Keywords**: Digital Twin, Large Language Models, Consumer Electronics, Visual Railway Defect Inspection, Predictive Maintenance, Multimodal Data Processing, Zero-shot Generalizability, Quality-of-Experience Feedback Loop.
