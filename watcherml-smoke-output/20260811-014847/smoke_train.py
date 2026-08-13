def main(config, max_steps=None):
    if config["batch_size"] > 16:
        raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
    steps = max_steps if max_steps is not None else config["training_steps"]
    return {"validation_loss": 0.4, "steps_completed": steps}
