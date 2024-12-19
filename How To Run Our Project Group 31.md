# How To Run Our Project Group 31

## 1. System Requirements

Before getting started, make sure you have the following software installed on your system:

- Python 3.11.12 or higher
- pip (Python package manager)
- Docker

## 2. Pull and Run Docker Image

This project should be run in a specific docker image.

Pull the docker image:

```
docker pull karmaresearch/wdps2
```

Run the docker image:

```shel
docker run -ti karmaresearch/wdps2
```

upload the file to the docker image:

```shell
docker cp D:\WorkSpace\WDP\group31_solution.py your_docker_image_id:/home/user

docker cp D:\WorkSpace\WDP\requirements.txt your_docker_image_id:/home/user
```

## 3. Install Dependencies

Install all the dependencies listed in the `requirements.txt` file:

```
pip install -r requirements.txt
```

And make sure you download all the model that the code need

```
python -m spacy download en_core_web_sm
```

## 4. Configure LLM Model File

This project requires a `.gguf` LLM model file. Follow these steps to configure it:

### 4.1 Model File Location

Place your `llama-2-7b.Q4_K_M.gguf` model file in a specific directory within the project (e.g., `models/`).

### 4.2 Configure Model Path

In the global_multi_demo.py file, line 212, specify the path to the model file. For example, if your model file is in the ‘model’ folder, you can configure it like this:

```
llm_model = Llama("models/llama-2-7b.Q4_K_M.gguf", verbose=False)
```

## 5. Start the Project

Once the dependencies are installed and the model file is configured, you can start the project.

```
python group31_solution.py
```
