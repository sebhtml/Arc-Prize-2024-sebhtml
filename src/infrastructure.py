import runpod
import yaml
import os


def get_api_key(api_key_file: str):
    if not os.path.isfile(api_key_file):
        print(f"File {api_key_file} does not exist")
        return None

    with open(api_key_file, "r") as stream:
        try:
            values = yaml.safe_load(stream)
            return values['RUNPOD_API_KEY']
        except yaml.YAMLError as exc:
            print(exc)

    return None


def get_current_pod_id():
    """
        See https://docs.runpod.io/pods/references/environment-variables
    """
    try:
        return os.environ['RUNPOD_POD_ID']
    except:
        return None


def terminate_pod(api_key_file: str):
    """
    See https://github.com/runpod/runpod-python?tab=readme-ov-file#endpoints
    """
    api_key = get_api_key(api_key_file)
    if api_key == None:
        return

    current_pod_id = get_current_pod_id()
    if current_pod_id == None:
        return

    runpod.api_key = api_key

    # Get all my pods
    pods = runpod.get_pods()
    for pod in pods:
        pod_id = pod['id']
        if pod_id == current_pod_id:
            # Terminate the pod
            runpod.terminate_pod(pod_id)
