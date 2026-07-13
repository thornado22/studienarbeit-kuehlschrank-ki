# Kühlschrank KI

This is a project created by Ruth Fröhlich, Hannah Schmidt and Nelly Thoröe as part of the T3101 Module of the DHBW Ravensburg.

## Run Application
This project uses docker to build the frontend and backend. This requires that [docker desktop](https://www.docker.com/products/docker-desktop/) is installed on the system and running.

### Start and build application

Execute this from the root directory of the project:

```
docker compose up --build
```

The build process takes some time as the models and large libraries need to be installed. The images will take around 5GB of space. Once both containers are built, the frontend is running at: [localhost port 4200](http://localhost:4200/)

### Start application

If the images are already built, the application can be started using:
```
docker compose up
```

### Stop application

The application can be stopped using:
```
docker compose down
```
or alternatively with `ctrl + C` in the terminal or by deleting the containers in docker desktop.

### Alternative without Docker

The application can also be run without Docker. In this case, the backend and frontend must be installed and started separately.

#### Backend (Windows)

Open a terminal in the project root and run:

```
cd backend

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
pip install "fastapi[standard]"
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu

fastapi dev api.py
```

#### Frontend (Windows)

Open a second terminal and run:

```
cd frontend

npm install
ng serve
```

## Configure Models
The `backend/project_config.py` defines:
- the API key for Groq (the current key has limited tokens and is used for testing purposes)
- the version of the prompt used for the LLM
- the CNN model used for classifying the images

A different CNN or prompt can be selected by changing the path.

Due to .pth files being very large, only the weights of the Test 3 model are included. The `backend/computerVision/images/test_images` folder contains 4 sample images.


## Notes

Before handing in this project, we moved an deleted files in the computerVision folder associated with tests and graphics. Paths and references were not updated, wich may cause errors if any file is run, that is not needed for the prototype. The files were left as a reference to our project work.
