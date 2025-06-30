# Start from the python:3.10.0-bullseye image
FROM python:3.10.0-bullseye

# Set environment variables to prevent Python from writing pyc files to disc and buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install SSH and other necessary packages
RUN apt-get update && \
    apt-get install -y openssh-server sudo vim wget curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set root password
RUN echo 'root:salva' | chpasswd

# Configure SSH
RUN mkdir /var/run/sshd
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo 'UsePAM yes' >> /etc/ssh/sshd_config

# Install code-server (VSCode) and Jupyter Notebook
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN pip install --no-cache-dir jupyter
RUN pip install git+https://github.com/navass11/pysheds

# Expose the SSH port, code-server port, and Jupyter Notebook port
EXPOSE 22
EXPOSE 8080
EXPOSE 8888

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /root

# Start the SSH, code-server, and Jupyter Notebook services
CMD service ssh start && \
    code-server --bind-addr 0.0.0.0:8080 & \
    jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' & \
    tail -f /dev/null

# Grant necessary permissions
RUN chmod +x /root
