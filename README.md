# About

This repository contains code and tests for a set of containerized flood inundation map (FIM) evaluation jobs that are meant to be composed together to form a FIM evaluation workflow. Each job can be developed and run independently of other jobs or used in conjunction with a job orchestrator to run evaluations at scale. The intended target for the orchestrator is HashiCorp Nomad and the jobs have been designed to make them easy to run as parameterized jobs on a Nomad cluster.

Two Docker images are built from this repo:

| Image | Jobs |
|-------|------|
| `ghcr.io/ngwpc/auto-eval-jobs:latest` | `hand_inundator`, `fim_mosaicker` |
| `ghcr.io/ngwpc/auto-eval-jobs-gval:latest` | `agreement_maker`, `depth_evaluator` |

Images are built and pushed automatically on push to `main`. No registry authentication is required to pull.

A more thorough description of the inputs and outputs of each job as well as the intended behavior of a job can be found in the jobs' [interfaces](/interfaces/interfaces.md) descriptions. A job interface is a formal specification of a job's inputs, outputs, and arguments specified using [json-schema](https://json-schema.org/). At the moment the interface yaml files serve as a guide for developers when (re)implementing jobs and for understanding the possible ways that jobs can interact through their inputs and outputs. In the future they could also be used to validate the data accepted and produced by each job.

We also provide another document in the interfaces directory listing [job conventions](/interfaces/job_conventions.md) to use when implementing jobs. These are conventions developers are expected to follow when implementing jobs and include implementation guidelines on how inputs and outputs should be handled by the job, rules for job entrypoint argument names, and job logging.

# Setup

We provide Dockerfiles for each image as well as two Docker compose files. The first Docker compose file contains services for running the jobs in an interactive shell for development and debugging. The second Docker compose file specifies services that run a job's test suite on startup.

## Configuring environment

Depending on which job is being run, some configuration of the container's environment may be necessary. Currently, the most important environment variables are the credentials that jobs need to read and write data to AWS S3 buckets.

The docker compose services follow the following policy for environment variables:

- Variables that need to be dynamically updated (like AWS credentials with a fixed, short expiration date) will be referenced in the "environment" block of the docker compose service definition and will be read from the user's local shell environment. To populate your Docker containers with these credentials you would set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, and `AWS_DEFAULT_REGION` in your host shell's environment before starting the containers.
- Fixed variables will be referenced from a .env file in an env_file block. An example .env file is located at [example.env](example.env). Copy it to get started:

```bash
cp example.env .env
```

The `.env` file also controls GDAL performance settings (`GDAL_CACHEMAX`, `GDAL_NUM_THREADS`), output formats, and processing defaults. The defaults are tuned for production use — adjust `GDAL_CACHEMAX` and `DASK_CLUST_MAX_MEM` if running on a smaller machine.

## Using Docker Compose dev services to interactively run and debug jobs

Here is an example for how to enter into an interactive shell for the fim_mosaicker job:

```
# start all Docker Compose dev services
docker compose up -d

# enter into a shell for developing fim_mosaicker
docker compose exec mosaic-dev bash
```

You only need to run the `docker compose up -d` command once and then after that you can enter into shells for any job there is a service for. There are Docker Compose dev services for all four jobs: `mosaic-dev` (fim_mosaicker), `inundate-dev` (hand_inundator), `make-agreement-dev` (agreement_maker), and `calculate-metrics-dev` (depth_evaluator).

Once you have entered the shell for a job container, the container's '/app' directory will contain code, tests, and mock data for that job. The bash shell can be exited by entering the `exit` command from bash.

Once you are done developing a job all the dev services can be shutdown with:

```
docker compose down
```

## Running tests

Tests for each job can also be run as Docker Compose services. To run all tests at once:

```
docker compose -f docker-compose-tests.yml up --abort-on-container-exit
```

To run tests for a specific job:

```
docker compose -f docker-compose-tests.yml run --rm mosaic-test
docker compose -f docker-compose-tests.yml run --rm inundate-test
docker compose -f docker-compose-tests.yml run --rm make-agreement-test
docker compose -f docker-compose-tests.yml run --rm calculate-metrics-test
```

Each docker service in the tests docker compose file runs tests for a different job.

# License

