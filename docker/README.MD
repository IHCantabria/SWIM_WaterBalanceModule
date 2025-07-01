# Python Development Environment with SSH, VSCode, Jupyter, and pystan

![Docker Image Version (latest by date)](https://img.shields.io/docker/v/arannentriki/pystan_env?style=flat-square)
![Docker Pulls](https://img.shields.io/docker/pulls/arannentriki/pystan_env?style=flat-square)
![Docker Stars](https://img.shields.io/docker/stars/arannentriki/pystan_env?style=flat-square)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg?style=flat-square)
![License](https://img.shields.io/github/license/Amine-RT/pystan_docker?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/Amine-RT/pystan_docker?style=flat-square)


This Docker image is based on `python:3.10.0-bullseye` and is designed to provide a comprehensive development environment with the following features:

- **Python 3.10**: A modern version of Python for all your scripting and development needs.
- **pystan**: Pre-installed for your specific library requirements.
- **matplotlib and scipy**: Additional pre-installed scientific libraries.
- **SSH Access**: Secure shell access is enabled, allowing you to connect remotely to the container.
- **VSCode (code-server)**: A web-based version of Visual Studio Code for a full-featured code editor accessible from your browser.
- **Jupyter Notebook**: An interactive computing environment that enables you to create and share documents containing live code, equations, visualizations, and narrative text.
- **Pre-loaded Notebooks**: Includes a `Notesbook` directory in the root user's home directory, allowing you to start with pre-defined Jupyter Notebooks.

## Features

1. **SSH Server**:
   - Accessible on port `22`.
   - Allows remote connection to the container.
   - Default root password: `rootpassword` (this should be changed for security purposes).

2. **VSCode (code-server)**:
   - Accessible on port `8080`.
   - Provides a robust, web-based code editor with extensions support.

3. **Jupyter Notebook**:
   - Accessible on port `8888`.
   - No password or token required for easy access.
   - Ideal for data analysis, machine learning, and interactive coding.

4. **Pre-installed Python Packages**:
   - `pystan`, `matplotlib`, `scipy`, and other dependencies specified in `requirements.txt` are pre-installed.

5. **Pre-loaded Notebooks**:
   - A `Notesbook` directory is copied to `/root/Notesbook`, ready for your use.

## Usage

### Building the Image

To build the Docker image, ensure you have Docker installed and run the following command in the directory containing your `Dockerfile` and `requirements.txt`:

```sh
docker build -t python-swim-ssh-vscode-jupyter .
```

### Running the Container

Run the Docker container with the following command:

```sh
docker run -d -p 2222:22 -p 8080:8080 -p 8888:8888 --name dev-container  -v %cd%:/root python-swim-ssh-vscode-jupyter
```

### Connecting via SSH

To connect to the container via SSH, use the following command:

```sh
ssh root@localhost -p 2222
```
Each time you recreate the container, it generates a new **SSH host key**.  
If you had previously connected, SSH will detect that the key has changed and will display a warning about a possible _man-in-the-middle_ attack, blocking the connection for security reasons.
```sh
ssh-keygen -R [localhost]:2222
```

Use the password `salva`.

### Accessing VSCode

Open your browser and navigate to `http://localhost:8080`. You should see the code-server (VSCode) interface.

### Accessing Jupyter Notebook

Open your browser and navigate to `http://localhost:8888`. You should see the Jupyter Notebook interface.

## Customization

### Changing the Root Password

For security, it is recommended to change the default root password by modifying the Dockerfile or changing it manually after connecting via SSH:

```sh
passwd
```

### Adding Additional Python Packages

Update the `requirements.txt` file to include any additional packages you need and rebuild the Docker image.

### Pre-loading More Files

Modify the Dockerfile to copy additional directories or files into the container as needed.