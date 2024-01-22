---
title: "Building Docker Image"
date:   2023-12-30 22:00:00
categories: [docker]
tags: [docker,docs,linux,windows]    
image:
  path: /assets/imgs/headers/docker.webp
---

## Introduction:
In the realm of software development, Docker stands as a revolutionary force, providing an efficient solution to the challenges of packaging, distributing, and running applications. Its inception was rooted in the need for consistent and reproducible environments, ultimately revolutionizing containerization.

## Key Advantages:

**Consistency Across Environments:**
Docker ensures applications work consistently across different environments, eliminating the "it works on my machine" challenge.

**Isolated and Portable Containers:**
Using isolated, portable containers, Docker simplifies application creation and deployment, enhancing security and dependency management.

**Portability for Any Environment:**
Docker containers encapsulate everything needed for an application to run, facilitating seamless deployment across various environments.

**Resource Efficiency:**
Docker containers, sharing the same OS kernel, offer superior resource efficiency, resulting in faster startups and optimized resource utilization.

**Scalability:**
Docker's architecture supports easy scaling of applications, ensuring optimal performance and responsiveness to increased workloads.

**DevOps Integration:**
Docker plays a crucial role in DevOps, promoting collaboration between development and operations through streamlined deployment pipelines.

**Microservices Architecture:**
Aligned with microservices, Docker allows independent development, deployment, and scaling of services, fostering agility in complex systems.


In the world of containerization, building Docker images, working with containers, and orchestrating multi-container applications with Docker Compose are essential skills. This guide will walk you through the basics and introduce you to Docker Compose for managing complex setups.

## Prerequisites

<br>

Make sure you have [Docker installed](https://docs.docker.com/engine/).

<br>

## Creating a Dockerfile

Start by creating a Dockerfile. Use the following commands:

```shell
sudo touch Dockerfile
sudo nano Dockerfile
```

Paste the content:

```shell
FROM alpine
CMD ["echo", "Hello I did it!"]
```

This Dockerfile uses the alpine image as a base, echoing "Hello I did it!" when the container starts.

## Building a Docker Image
Build the image:

```shell
docker build -t my-hello-image .
```

## Creating and Running a Container

Creating and Running a Container

```shell
docker run my-hello-image
```

Listing Containers

```shell
docker ps
docker ps -a
```

Stopping and Removing a Container
```shell
docker stop <CONTAINER_ID or CONTAINER_NAME>
docker rm <CONTAINER_ID or CONTAINER_NAME>
```
Docker Container Lifecycle:
Containers follow a lifecycle: docker create, docker start, docker stop, and docker rm. The workflow is to create a container from an image, start it, stop it when done, and optionally remove it.

## Share Local Directory with Docker Container:
Create a Local Directory:

Before sharing, create a local directory on your host machine. For example, let's create a directory named my_local_data.

```shell
mkdir my_local_data
Run Docker Container with Volume Mount:
```
Use the -v or --volume option to mount the local directory into the container. Replace [local_path] with the path to your local directory and [container_path] with the path inside the container.

```shell
docker run -v [local_path]:[container_path] [image_name]
```
Example:

```shell
docker run -v /path/to/my_local_data:/app/data my_image
```

Verify the Volume Mount:

Inside the container, check if the mounted directory is accessible.

```shell
docker exec -it [container_id or container_name] ls [container_path]

# Example:

docker exec -it my_container ls /app/data
```

Share Data Bidirectionally:

Changes made in the container are reflected on the local machine and vice versa. This bidirectional sharing is useful for development or data persistence scenarios.
Example: Running a Node.js Application with Shared Volume:
Assuming you have a Node.js application in your local directory (my_local_data), here's how you can run it in a Docker container:

```shell
# Create a local directory
mkdir my_local_data

# Create a simple Node.js app in the local directory
echo "console.log('Hello from Docker!');" > my_local_data/app.js

# Run the Docker container with volume mount
docker run -v $(pwd)/my_local_data:/app node:alpine node /app/app.js
This example runs a Node.js app in a Docker container, and any changes made to app.js in the local directory are immediately reflected inside the container.
```

## Docker Compose
Docker Compose simplifies the process of defining and running multi-container Docker applications. Create a docker-compose.yml file:

```yaml
version: '3'
services:
  my-hello-app:
    image: my-hello-image
```

Run the application with Docker Compose:

```shell
docker-compose up
```

Docker Compose will create and start the specified services. Press Ctrl+C to stop.


## Advanced Docker Compose
Scaling Services
```yaml
version: '3'
services:
  my-hello-app:
    image: my-hello-image
  my-another-app:
    image: my-another-image
```

Scale services:

```shell
docker-compose up --scale my-hello-app=2
```

**Environment Variables**
Pass environment variables:

```yaml
version: '3'
services:
  my-hello-app:
    image: my-hello-image
    environment:
      - MY_VARIABLE=value
```

## Conclusion
As technology evolves, Docker remains at the forefront of modern software development practices. Its utility extends beyond convenience—it fundamentally transforms the way we conceive, build, and deploy applications. Whether you're a developer aiming for consistency, an operations professional pursuing scalability, or part of a collaborative DevOps environment, Docker is a powerful tool that has reshaped the landscape of containerization.








